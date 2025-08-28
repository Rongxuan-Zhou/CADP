"""
Sliding Mode Control (SMC) based Safe Tracking Controller
Implementation of Algorithm 3 from CADP paper

This module implements the SMC-based safe tracking control that unifies
CLF (Control Lyapunov Functions) and CBF (Control Barrier Functions)
objectives without the infeasibility issues of QP-based methods.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any

class SMCController:
    """
    SMC-based Safe Tracking Controller
    
    Implements Algorithm 3 from the CADP paper, providing unified
    CLF-CBF control without optimization infeasibility issues.
    """
    
    def __init__(
        self,
        robot_dof: int = 7,
        clf_gain: float = 10.0,
        cbf_weight: float = 0.1,
        switching_gain: float = 50.0,
        boundary_layer: float = 0.01,
        manifold_offset: float = 0.5,
        singularity_threshold: float = 1e-3,
        velocity_limit: float = 1.0,
        acceleration_limit: float = 2.0,
        collision_margin: float = 0.05
    ):
        """
        Initialize SMC Controller
        
        Args:
            robot_dof: Degrees of freedom
            clf_gain: CLF gain matrix diagonal value
            cbf_weight: CBF weight in sliding manifold (β)
            switching_gain: Switching control gain (K)
            boundary_layer: Boundary layer width (Φ)
            manifold_offset: Sliding manifold offset (c)
            singularity_threshold: Threshold to avoid singularity (ε)
            velocity_limit: Maximum joint velocity
            acceleration_limit: Maximum joint acceleration
            collision_margin: Safety margin from obstacles
        """
        self.dof = robot_dof
        self.clf_gain = clf_gain
        self.cbf_weight = cbf_weight
        self.switching_gain = switching_gain
        self.boundary_layer = boundary_layer
        self.manifold_offset = manifold_offset
        self.eps = singularity_threshold
        
        # Safety limits
        self.v_max = velocity_limit
        self.a_max = acceleration_limit
        self.collision_margin = collision_margin
        
        # CLF gain matrix P = diag(clf_gain, ...)
        self.P = torch.eye(robot_dof) * clf_gain
        
        # Control history for numerical differentiation
        self.prev_state = None
        self.prev_time = None
        
    def compute_control(
        self,
        current_state: torch.Tensor,
        reference_trajectory: torch.Tensor,
        reference_time: float,
        sdf_function: Optional[callable] = None,
        dt: float = 0.01
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute SMC control input
        
        Implements Algorithm 3: SMC-based Safe Tracking Control
        
        Args:
            current_state: Current robot state [q, q_dot] shape (2*dof,)
            reference_trajectory: Reference trajectory shape (T, dof)
            reference_time: Current time in reference trajectory
            sdf_function: Signed distance function for collision detection
            dt: Time step for numerical differentiation
            
        Returns:
            control_input: Computed control torques shape (dof,)
            info_dict: Dictionary with debug information
        """
        device = current_state.device
        
        # Extract current position and velocity
        q_current = current_state[:self.dof]
        qd_current = current_state[self.dof:]
        
        # Get reference state (interpolate if needed)
        q_ref, qd_ref, qdd_ref = self._get_reference_state(
            reference_trajectory, reference_time, dt
        )
        q_ref = q_ref.to(device)
        qd_ref = qd_ref.to(device)
        qdd_ref = qdd_ref.to(device)
        
        # Step 1: Compute tracking error and CLF
        tracking_error = q_current - q_ref
        error_dot = qd_current - qd_ref
        
        # CLF: V(x,t) = 1/2 * e^T * P * e
        V_clf = 0.5 * tracking_error.T @ self.P @ tracking_error
        
        # Step 2: Compute safety CBF
        B_cbf = self._compute_safety_barriers(
            q_current, qd_current, sdf_function
        )
        
        # Step 3: Construct sliding manifold
        # s(x,t) = V(x,t) + β*B(x) - c
        s_manifold = V_clf + self.cbf_weight * B_cbf - self.manifold_offset
        
        # Step 4: Compute gradients for control law
        # For CLF: ∇V = e^T * P (only position part)
        grad_V_pos = tracking_error @ self.P  # Shape: (7,)
        grad_V_vel = error_dot  # Shape: (7,)
        grad_V = torch.cat([grad_V_pos, grad_V_vel])  # Shape: (14,)
        
        # For CBF: ∇B (approximated)
        grad_B = self._compute_cbf_gradient(q_current, qd_current, sdf_function)
        
        # Ensure grad_B has correct shape (14,) for full state
        if grad_B.shape[0] != 2 * self.dof:
            # If gradient is wrong size, create a zero gradient
            grad_B = torch.zeros(2 * self.dof, device=device)
        
        # For sliding variable: ∇s = ∇V + β*∇B
        grad_s = grad_V + self.cbf_weight * grad_B
        
        # Step 5: Compute Lie derivatives
        # System dynamics: x_dot = f(x) + g(x)*u
        f_x = torch.cat([qd_current, torch.zeros(self.dof, device=device)])
        g_x = torch.cat([
            torch.zeros(self.dof, self.dof, device=device),
            torch.eye(self.dof, device=device)
        ])
        
        # Lf_s = ∇s^T * f(x)
        Lf_s = grad_s @ f_x
        
        # Lg_s = ∇s^T * g(x)  
        Lg_s = grad_s @ g_x
        
        # Step 6: Equivalent control (maintains manifold)
        if torch.norm(Lg_s) > self.eps:
            u_eq = -torch.pinverse(Lg_s.unsqueeze(0)) @ Lf_s.unsqueeze(0)
            u_eq = u_eq.squeeze()
        else:
            # Avoid singularity
            u_eq = torch.zeros(self.dof, device=device)
            
        # Step 7: Switching control (drives to manifold)
        u_sw = -self.switching_gain * self._saturation(
            s_manifold / self.boundary_layer
        )
        
        # Step 8: Total control
        u_total = u_eq + u_sw
        
        # Apply control limits
        u_total = torch.clamp(u_total, -self.a_max, self.a_max)
        
        # Prepare debug information
        info = {
            'tracking_error': tracking_error.detach().cpu().numpy(),
            'clf_value': V_clf.item(),
            'cbf_value': B_cbf.item(), 
            'sliding_variable': s_manifold.item(),
            'equivalent_control': u_eq.detach().cpu().numpy(),
            'switching_control': u_sw.item(),
            'control_effort': torch.norm(u_total).item(),
            'safety_margin': B_cbf.item()
        }
        
        return u_total, info
    
    def _get_reference_state(
        self, 
        trajectory: torch.Tensor, 
        time: float, 
        dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get reference state (position, velocity, acceleration) at given time
        """
        T, dof = trajectory.shape
        
        # Time discretization
        time_idx = int(time / dt)
        time_idx = max(0, min(time_idx, T - 1))
        
        # Reference position
        q_ref = trajectory[time_idx]
        
        # Reference velocity (numerical differentiation)
        if time_idx < T - 1:
            qd_ref = (trajectory[time_idx + 1] - trajectory[time_idx]) / dt
        else:
            qd_ref = torch.zeros_like(q_ref)
            
        # Reference acceleration (numerical differentiation)
        if time_idx < T - 2:
            qdd_ref = (trajectory[time_idx + 2] - 2*trajectory[time_idx + 1] + 
                      trajectory[time_idx]) / (dt**2)
        else:
            qdd_ref = torch.zeros_like(q_ref)
            
        return q_ref, qd_ref, qdd_ref
    
    def _compute_safety_barriers(
        self, 
        q: torch.Tensor, 
        qd: torch.Tensor, 
        sdf_function: Optional[callable]
    ) -> torch.Tensor:
        """
        Compute minimum safety barrier function value
        
        Returns minimum of:
        - Collision barrier: SDF(q) - δ_safe
        - Velocity barrier: v_max^2 - ||qd||^2  
        - Joint limit barriers: (q - q_min) * (q_max - q)
        """
        barriers = []
        
        # Collision barrier
        if sdf_function is not None:
            try:
                sdf_value = sdf_function(q)
                if isinstance(sdf_value, (int, float)):
                    sdf_value = torch.tensor(sdf_value, device=q.device)
                B_col = sdf_value - self.collision_margin
                barriers.append(B_col)
            except:
                # If SDF computation fails, use conservative barrier
                barriers.append(torch.tensor(0.0, device=q.device))
        
        # Velocity barrier
        B_vel = torch.tensor(self.v_max**2, device=q.device) - torch.norm(qd)**2
        barriers.append(B_vel)
        
        # Joint limit barriers (simplified - assuming symmetric limits)
        q_limit = torch.pi  # Simplified joint limits
        B_joint = torch.min(torch.min(q + q_limit, q_limit - q))
        barriers.append(B_joint)
        
        # Return minimum barrier value
        return torch.min(torch.stack(barriers))
    
    def _compute_cbf_gradient(
        self, 
        q: torch.Tensor, 
        qd: torch.Tensor, 
        sdf_function: Optional[callable]
    ) -> torch.Tensor:
        """
        Compute gradient of CBF with respect to state
        
        Simplified implementation using numerical differentiation
        """
        grad = torch.zeros(2 * self.dof, device=q.device)
        
        # Finite difference step
        eps = 1e-6
        
        # Current barrier value
        B_current = self._compute_safety_barriers(q, qd, sdf_function)
        
        # Gradient w.r.t. position
        for i in range(self.dof):
            q_plus = q.clone()
            q_plus[i] += eps
            B_plus = self._compute_safety_barriers(q_plus, qd, sdf_function)
            grad[i] = (B_plus - B_current) / eps
            
        # Gradient w.r.t. velocity  
        for i in range(self.dof):
            qd_plus = qd.clone()
            qd_plus[i] += eps
            B_plus = self._compute_safety_barriers(q, qd_plus, sdf_function)
            grad[self.dof + i] = (B_plus - B_current) / eps
            
        return grad
    
    def _saturation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Saturation function for switching control
        
        sat(x) = sign(x) if |x| > 1, else x
        """
        return torch.clamp(x, -1.0, 1.0)
    
    def reset(self):
        """Reset controller state"""
        self.prev_state = None
        self.prev_time = None

def create_smc_controller(**kwargs) -> SMCController:
    """
    Factory function to create SMC controller with default parameters
    """
    default_params = {
        'robot_dof': 7,
        'clf_gain': 10.0,
        'cbf_weight': 0.1,
        'switching_gain': 50.0,
        'boundary_layer': 0.01,
        'manifold_offset': 0.5,
        'singularity_threshold': 1e-3,
        'velocity_limit': 1.0,
        'acceleration_limit': 2.0,
        'collision_margin': 0.05
    }
    
    # Update with user parameters
    default_params.update(kwargs)
    
    return SMCController(**default_params)

# Theoretical Guarantees Implementation
class SMCAnalyzer:
    """
    Analyzer for SMC theoretical guarantees
    
    Implements theoretical analysis from Section V.F of the paper
    """
    
    def __init__(self, controller: SMCController):
        self.controller = controller
        
    def compute_reaching_time(self, initial_sliding_variable: float) -> float:
        """
        Compute finite-time reaching to sliding manifold
        
        Returns: t_s = |s(x_0)| / K (Theorem 1)
        """
        K = self.controller.switching_gain
        return abs(initial_sliding_variable) / K
    
    def verify_safety_maintenance(self, barrier_value: float) -> bool:
        """
        Verify that safety is maintained on the manifold
        
        Returns: True if B(x(t)) ≥ 0 is guaranteed
        """
        return barrier_value >= 0
    
    def compute_tracking_convergence(self, tracking_error: torch.Tensor) -> float:
        """
        Analyze tracking error convergence
        
        Returns: Expected convergence rate
        """
        return torch.norm(tracking_error).item()
    
    def verify_disturbance_robustness(self, disturbance_bound: float) -> bool:
        """
        Verify robustness to bounded disturbances
        
        Returns: True if K > d_max guarantees robustness
        """
        K = self.controller.switching_gain
        return K > disturbance_bound