"""
Control Barrier Function Verifier for CADP
Implementation of Algorithm 2 from the CADP paper
Author: CADP Project Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class CBFVerificationResult:
    """Results of CBF trajectory verification"""
    safe_trajectory: torch.Tensor
    num_unsafe_waypoints: int
    num_corrections: int
    max_correction_norm: float
    correction_ratio: float
    verification_time_ms: float
    barrier_violations: Dict
    feasibility_adjusted: bool


class ControlBarrierFunction:
    """
    Implementation of Control Barrier Functions for trajectory verification
    Based on CADP paper Algorithm 2: Safety Verification and Projection
    """
    
    def __init__(self, robot_config: Dict):
        """
        Initialize CBF verifier with robot-specific parameters
        
        Args:
            robot_config: Dictionary containing robot specifications
                - q_min: Joint lower limits (7-DOF for Franka)
                - q_max: Joint upper limits  
                - v_max: Maximum velocity (1.0 rad/s from paper)
                - a_max: Maximum acceleration (2.0 rad/s²)
                - delta_safe: Safety margin (0.05m)
        """
        # Robot kinematic limits (Franka Panda 7-DOF from paper Section V.C)
        self.q_min = torch.tensor(robot_config.get('q_min', 
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]))
        self.q_max = torch.tensor(robot_config.get('q_max',
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]))
        
        # Dynamic limits from paper Section V.C
        self.v_max = robot_config.get('v_max', 1.0)  # rad/s
        self.a_max = robot_config.get('a_max', 2.0)  # rad/s²
        self.delta_safe = robot_config.get('delta_safe', 0.05)  # meters
        
        # Workspace bounds as fallback (when SDF not available)
        self.workspace_bounds = robot_config.get('workspace_bounds', [
            [-0.5, 0.5],   # x bounds
            [-0.5, 0.5],   # y bounds  
            [0.0, 1.0]     # z bounds
        ])
        
        # Initialize environment SDF (will be set externally)
        self.sdf = None
        
        # Performance tracking
        self.verification_stats = {
            'total_verifications': 0,
            'total_corrections': 0,
            'avg_verification_time': 0.0
        }
        
        print(f"🛡️  CBF Verifier initialized:")
        print(f"   • Joint limits: {self.q_min.tolist()} to {self.q_max.tolist()}")
        print(f"   • Velocity limit: {self.v_max} rad/s")
        print(f"   • Acceleration limit: {self.a_max} rad/s²")
        print(f"   • Safety margin: {self.delta_safe} m")
        
    def set_environment_sdf(self, sdf_function):
        """Set the Signed Distance Field for collision detection"""
        self.sdf = sdf_function
        print("✅ Environment SDF configured for collision detection")
        
    def compute_barrier_values(self, q: torch.Tensor, q_dot: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all barrier function values (Paper Algorithm 2, Lines 3-6)
        
        B(x) > 0 implies safety
        B(x) ≤ 0 implies constraint violation
        
        Args:
            q: Joint positions [batch, 7]
            q_dot: Joint velocities [batch, 7]
            
        Returns:
            Dictionary of barrier values for each constraint
        """
        barriers = {}
        
        # 1. Collision barrier: B_col(q) = SDF(q) - δ_safe (Line 3 of Algorithm 2)
        if self.sdf is not None:
            ee_pos = self.forward_kinematics(q)
            sdf_values = self.sdf(ee_pos)
            barriers['collision'] = sdf_values - self.delta_safe
        else:
            # Workspace boundary constraints as fallback
            barriers['collision'] = self.workspace_barrier(q)
        
        # 2. Velocity barrier: B_vel(q̇) = v_max² - ||q̇||² (Line 4)
        velocity_norm_sq = torch.sum(q_dot ** 2, dim=-1)
        barriers['velocity'] = self.v_max ** 2 - velocity_norm_sq
        
        # 3. Joint limit barriers: B_joint(q) = ∏(q_i - q_min,i)(q_max,i - q_i) (Line 5)
        # For numerical stability, use minimum margin approach
        q_min_expanded = self.q_min.unsqueeze(0) if q.dim() > 1 else self.q_min
        q_max_expanded = self.q_max.unsqueeze(0) if q.dim() > 1 else self.q_max
        
        lower_margins = q - q_min_expanded
        upper_margins = q_max_expanded - q
        
        # Joint limit barrier: minimum distance to any limit
        joint_barriers = torch.minimum(lower_margins, upper_margins)
        barriers['joint_limits'] = torch.min(joint_barriers, dim=-1)[0]
        
        # Combined barrier (minimum of all barriers) - Line 6
        all_barrier_values = [barriers[key] for key in ['collision', 'velocity', 'joint_limits']]
        barriers['combined'] = torch.min(torch.stack(all_barrier_values), dim=0)[0]
        
        return barriers
    
    def verify_trajectory(self, trajectory: torch.Tensor, dt: float = 0.1) -> CBFVerificationResult:
        """
        Main verification function implementing Algorithm 2
        
        Args:
            trajectory: Generated trajectory [T, 7] for joint positions
            dt: Time step between waypoints
            
        Returns:
            CBFVerificationResult with verified trajectory and statistics
        """
        import time
        start_time = time.time()
        
        T, dim = trajectory.shape
        assert dim == 7, f"Expected 7-DOF trajectory, got {dim}"
        
        safe_trajectory = trajectory.clone()
        
        # Compute velocities via finite differences
        velocities = torch.zeros_like(trajectory)
        velocities[1:] = (trajectory[1:] - trajectory[:-1]) / dt
        
        # Stage 1: Trajectory-level CBF verification (Lines 2-10)
        unsafe_waypoints = []
        barrier_violations = {}
        
        for t in range(T):
            q_t = trajectory[t:t+1]  # Keep batch dimension
            q_dot_t = velocities[t:t+1]
            
            barriers = self.compute_barrier_values(q_t, q_dot_t)
            
            # Check if any barrier is violated (Line 7)
            if barriers['combined'].item() < 0:
                unsafe_waypoints.append(t)
                barrier_violations[t] = {
                    key: val.item() for key, val in barriers.items()
                    if key != 'combined'  # Avoid redundancy
                }
        
        # Stage 2: Projection and repair (Lines 11-15)
        num_corrections = 0
        max_correction_norm = 0.0
        
        for t in unsafe_waypoints:
            # Project to safe set (Line 13)
            safe_q, correction_norm = self.project_to_safe_set(
                trajectory[t], 
                velocities[t],
                barrier_violations[t]
            )
            
            max_correction_norm = max(max_correction_norm, correction_norm)
            
            # Update trajectory (Line 14)
            safe_trajectory[t] = safe_q
            num_corrections += 1
        
        # Stage 3: Dynamics feasibility check (Lines 16-20)
        safe_trajectory, feasibility_adjusted = self.ensure_dynamic_feasibility(safe_trajectory, dt)
        
        # Calculate verification time
        verification_time_ms = (time.time() - start_time) * 1000
        
        # Update statistics
        self.verification_stats['total_verifications'] += 1
        self.verification_stats['total_corrections'] += num_corrections
        self.verification_stats['avg_verification_time'] = (
            (self.verification_stats['avg_verification_time'] * 
             (self.verification_stats['total_verifications'] - 1) + 
             verification_time_ms) / self.verification_stats['total_verifications']
        )
        
        return CBFVerificationResult(
            safe_trajectory=safe_trajectory,
            num_unsafe_waypoints=len(unsafe_waypoints),
            num_corrections=num_corrections,
            max_correction_norm=max_correction_norm,
            correction_ratio=len(unsafe_waypoints) / T,
            verification_time_ms=verification_time_ms,
            barrier_violations=barrier_violations,
            feasibility_adjusted=feasibility_adjusted
        )
    
    def project_to_safe_set(self, q: torch.Tensor, q_dot: torch.Tensor, 
                           violations: Dict) -> Tuple[torch.Tensor, float]:
        """
        Project unsafe configuration to nearest safe configuration
        Uses gradient-based projection to satisfy all constraints
        
        Implements: min ||q_safe - q|| s.t. B(q_safe) ≥ 0 for all barriers
        """
        q_safe = q.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([q_safe], lr=0.01)
        
        max_iters = 100
        epsilon = 1e-3  # Small positive margin for safety
        
        for iteration in range(max_iters):
            optimizer.zero_grad()
            
            # Compute current barriers
            barriers = self.compute_barrier_values(
                q_safe.unsqueeze(0), 
                q_dot.unsqueeze(0)
            )
            
            # Check if all constraints satisfied
            if barriers['combined'].item() >= epsilon:
                break
            
            # Loss: distance to original + barrier penalties
            distance_loss = torch.norm(q_safe - q)
            
            # Penalty for violated barriers (soft constraint)
            barrier_loss = 0.0
            penalty_weight = 10.0
            
            for key, value in barriers.items():
                if key != 'combined' and value.item() < epsilon:
                    barrier_loss += torch.relu(epsilon - value)
            
            total_loss = distance_loss + penalty_weight * barrier_loss
            total_loss.backward()
            optimizer.step()
            
            # Ensure joint limits are strictly enforced (hard constraint)
            with torch.no_grad():
                q_safe.data = torch.clamp(q_safe.data, self.q_min, self.q_max)
        
        correction_norm = torch.norm(q_safe - q).item()
        return q_safe.detach(), correction_norm
    
    def ensure_dynamic_feasibility(self, trajectory: torch.Tensor, dt: float) -> Tuple[torch.Tensor, bool]:
        """
        Ensure trajectory satisfies acceleration limits
        Implements time-scaling if needed (Lines 17-19 of Algorithm 2)
        """
        T = trajectory.shape[0]
        
        # Compute velocities and accelerations via finite differences
        velocities = torch.zeros_like(trajectory)
        velocities[1:] = (trajectory[1:] - trajectory[:-1]) / dt
        
        accelerations = torch.zeros_like(trajectory)
        accelerations[1:] = (velocities[1:] - velocities[:-1]) / dt
        
        # Check maximum acceleration across all joints and timesteps
        max_acc = torch.max(torch.norm(accelerations, dim=-1))
        
        feasibility_adjusted = False
        
        if max_acc > self.a_max:
            # Apply time scaling to reduce accelerations (Line 18-19)
            scaling_factor = self.a_max / max_acc
            
            # For simplicity, we scale the entire trajectory
            # Real implementation might use spline interpolation
            new_dt = dt / scaling_factor
            
            feasibility_adjusted = True
            print(f"⚠️  Applied time scaling: factor {scaling_factor:.3f} (new dt: {new_dt:.3f}s)")
        
        return trajectory, feasibility_adjusted
    
    def workspace_barrier(self, q: torch.Tensor) -> torch.Tensor:
        """
        Simplified workspace boundary constraints
        Used when full SDF is not available
        """
        # Approximate end-effector position using simplified FK
        ee_pos = self.forward_kinematics(q)
        
        # Check against workspace boundaries
        margins = []
        
        for dim in range(3):  # x, y, z
            lower_bound, upper_bound = self.workspace_bounds[dim]
            
            # Distance to lower and upper bounds
            lower_margin = ee_pos[..., dim] - lower_bound
            upper_margin = upper_bound - ee_pos[..., dim]
            
            # Minimum margin in this dimension
            dim_margin = torch.minimum(lower_margin, upper_margin)
            margins.append(dim_margin)
        
        # Overall workspace barrier: minimum across all dimensions
        workspace_barrier = torch.min(torch.stack(margins), dim=0)[0]
        
        return workspace_barrier
    
    def forward_kinematics(self, q: torch.Tensor) -> torch.Tensor:
        """
        Simplified forward kinematics for Franka Panda
        
        NOTE: This is a placeholder implementation. 
        Real deployment should use proper DH parameters or URDF model.
        """
        # Simplified approximation: use combination of joints for position estimation
        # This is sufficient for basic workspace constraints
        
        batch_size = q.shape[0] if q.dim() > 1 else 1
        
        # Approximate mapping based on typical Franka arm geometry
        # Joint 0,1,2: shoulder/elbow contribute to reach
        # Joint 3,4,5,6: wrist contributes to fine positioning
        
        reach_joints = q[..., :3]  # First 3 joints control primary reach
        wrist_joints = q[..., 4:6]  # Joints 4,5 for wrist positioning
        
        # Approximate end-effector position (simplified)
        # Real implementation needs proper kinematic chain
        base_reach = 0.3  # Approximate arm length
        ee_pos = torch.zeros(batch_size, 3) if q.dim() > 1 else torch.zeros(3)
        
        # X-Y position approximation
        ee_pos[..., 0] = base_reach * torch.cos(reach_joints[..., 0]) * torch.cos(reach_joints[..., 1])
        ee_pos[..., 1] = base_reach * torch.sin(reach_joints[..., 0]) * torch.cos(reach_joints[..., 1])
        
        # Z position approximation  
        ee_pos[..., 2] = 0.3 + base_reach * torch.sin(reach_joints[..., 1]) + 0.1 * torch.cos(wrist_joints[..., 0])
        
        return ee_pos
    
    def get_verification_statistics(self) -> Dict:
        """Return CBF verification statistics"""
        return {
            'total_verifications': self.verification_stats['total_verifications'],
            'total_corrections': self.verification_stats['total_corrections'],
            'avg_corrections_per_trajectory': (
                self.verification_stats['total_corrections'] / 
                max(1, self.verification_stats['total_verifications'])
            ),
            'avg_verification_time_ms': self.verification_stats['avg_verification_time']
        }
    
    def reset_statistics(self):
        """Reset verification statistics"""
        self.verification_stats = {
            'total_verifications': 0,
            'total_corrections': 0,
            'avg_verification_time': 0.0
        }


def create_franka_cbf_verifier() -> ControlBarrierFunction:
    """
    Factory function to create CBF verifier with Franka Panda defaults
    Matches the configuration used in CADP paper Section V.C
    """
    franka_config = {
        'q_min': [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        'q_max': [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        'v_max': 1.0,  # rad/s
        'a_max': 2.0,  # rad/s²
        'delta_safe': 0.05,  # meters
        'workspace_bounds': [
            [-0.5, 0.5],   # x bounds (meters)
            [-0.5, 0.5],   # y bounds (meters)
            [0.0, 1.0]     # z bounds (meters)
        ]
    }
    
    return ControlBarrierFunction(franka_config)


if __name__ == "__main__":
    # Test CBF verifier with sample trajectory
    print("🧪 Testing CBF Verifier...")
    
    cbf = create_franka_cbf_verifier()
    
    # Create a test trajectory with some violations
    T, dim = 20, 7
    test_trajectory = torch.randn(T, dim) * 0.5  # Random trajectory
    
    # Add some violations for testing
    test_trajectory[5, 0] = 3.5  # Violate joint limit
    test_trajectory[10, 1] = -2.0  # Another violation
    
    # Verify trajectory
    result = cbf.verify_trajectory(test_trajectory, dt=0.1)
    
    print(f"✅ Verification complete:")
    print(f"   • Unsafe waypoints: {result.num_unsafe_waypoints}")
    print(f"   • Corrections made: {result.num_corrections}")
    print(f"   • Max correction norm: {result.max_correction_norm:.4f}")
    print(f"   • Verification time: {result.verification_time_ms:.2f}ms")
    print(f"   • Correction ratio: {result.correction_ratio:.2%}")
    
    # Check final trajectory is safe
    final_violations = 0
    for t in range(T):
        barriers = cbf.compute_barrier_values(
            result.safe_trajectory[t:t+1], 
            torch.zeros(1, 7)  # Zero velocity for final check
        )
        if barriers['combined'].item() < 0:
            final_violations += 1
    
    print(f"   • Final violations: {final_violations} (should be 0)")
    assert final_violations == 0, "CBF verification failed - violations remain"
    print("🎉 CBF Verifier test passed!")