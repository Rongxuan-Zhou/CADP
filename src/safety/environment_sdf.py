"""
Environment Signed Distance Field for CADP
Implementation of collision detection using SDF representation
Author: CADP Project Team
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional


class EnvironmentSDF:
    """
    Signed Distance Field for environment representation
    Critical for collision checking in CBF verification
    
    Provides efficient collision queries for robot trajectories
    """
    
    def __init__(self, workspace_bounds: List[Tuple[float, float]] = None):
        """
        Initialize SDF with workspace configuration
        
        Args:
            workspace_bounds: [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        """
        self.workspace_bounds = workspace_bounds or [(-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)]
        self.obstacles = []
        self.grid_resolution = 0.01  # 1cm resolution for precomputed grids
        self.precomputed_sdf = None
        self.grid_coords = None
        
        print(f"🌐 Environment SDF initialized:")
        print(f"   • Workspace: {self.workspace_bounds}")
        print(f"   • Grid resolution: {self.grid_resolution}m")
    
    def add_sphere_obstacle(self, center: List[float], radius: float):
        """Add spherical obstacle to environment"""
        obstacle = {
            'type': 'sphere',
            'center': torch.tensor(center, dtype=torch.float32),
            'radius': radius
        }
        self.obstacles.append(obstacle)
        print(f"   ➕ Added sphere obstacle: center={center}, radius={radius}")
        
    def add_box_obstacle(self, center: List[float], dimensions: List[float]):
        """Add box obstacle to environment"""
        obstacle = {
            'type': 'box',
            'center': torch.tensor(center, dtype=torch.float32),
            'dimensions': torch.tensor(dimensions, dtype=torch.float32)
        }
        self.obstacles.append(obstacle)
        print(f"   ➕ Added box obstacle: center={center}, dimensions={dimensions}")
    
    def add_static_obstacles_scenario(self, scenario: str = 'cluttered'):
        """
        Add predefined obstacle scenarios for testing
        Matches scenarios from CADP paper Section VI
        """
        if scenario == 'cluttered':
            # Scenario 1: Cluttered Table-Top Manipulation (5-10 obstacles)
            obstacles_config = [
                {'center': [0.2, 0.3, 0.4], 'radius': 0.05},
                {'center': [-0.1, 0.2, 0.3], 'radius': 0.04},
                {'center': [0.3, -0.2, 0.5], 'radius': 0.06},
                {'center': [-0.3, -0.1, 0.2], 'radius': 0.03},
                {'center': [0.1, -0.3, 0.4], 'radius': 0.05},
            ]
            
            for obs in obstacles_config:
                self.add_sphere_obstacle(obs['center'], obs['radius'])
                
        elif scenario == 'narrow_corridor':
            # Scenario 3: Narrow Passage Navigation (15cm gap)
            corridor_width = 0.15
            wall_thickness = 0.05
            
            # Left wall
            self.add_box_obstacle(
                center=[-corridor_width/2 - wall_thickness/2, 0.0, 0.5],
                dimensions=[wall_thickness, 0.6, 0.3]
            )
            
            # Right wall  
            self.add_box_obstacle(
                center=[corridor_width/2 + wall_thickness/2, 0.0, 0.5],
                dimensions=[wall_thickness, 0.6, 0.3]
            )
            
        elif scenario == 'dynamic_test':
            # Simple setup for dynamic obstacle testing
            self.add_sphere_obstacle([0.0, 0.0, 0.5], 0.08)
            
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
        print(f"✅ Loaded '{scenario}' obstacle scenario ({len(self.obstacles)} obstacles)")
    
    def compute_sdf(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance to nearest obstacle
        
        Args:
            positions: [N, 3] or [3] tensor of 3D positions
            
        Returns:
            sdf_values: [N] or scalar - signed distance values
                       Positive: free space, Negative: inside obstacle
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = positions.shape[0]
        min_distances = torch.full((batch_size,), float('inf'))
        
        if len(self.obstacles) == 0:
            # No obstacles - return large positive values (safe)
            result = torch.full((batch_size,), 10.0)
        else:
            # Compute distance to each obstacle
            for obstacle in self.obstacles:
                if obstacle['type'] == 'sphere':
                    distances = self._sdf_sphere(positions, obstacle)
                elif obstacle['type'] == 'box':
                    distances = self._sdf_box(positions, obstacle)
                else:
                    continue
                    
                min_distances = torch.minimum(min_distances, distances)
            
            result = min_distances
        
        # Apply workspace boundaries as additional constraints
        workspace_sdf = self._sdf_workspace(positions)
        result = torch.minimum(result, workspace_sdf)
        
        if squeeze_output:
            result = result.squeeze(0)
            
        return result
    
    def _sdf_sphere(self, positions: torch.Tensor, obstacle: Dict) -> torch.Tensor:
        """Signed distance to sphere obstacle"""
        center = obstacle['center']
        radius = obstacle['radius']
        
        # Distance to sphere surface
        distances_to_center = torch.norm(positions - center, dim=-1)
        return distances_to_center - radius
    
    def _sdf_box(self, positions: torch.Tensor, obstacle: Dict) -> torch.Tensor:
        """Signed distance to box obstacle"""
        center = obstacle['center'] 
        dimensions = obstacle['dimensions']
        
        # Transform to box local coordinates
        local_pos = torch.abs(positions - center)
        
        # Distance to box surface (simplified)
        half_dims = dimensions / 2.0
        
        # Distance in each dimension
        distances = local_pos - half_dims
        
        # Outside distance: max(0, max(distances))
        outside_dist = torch.clamp(distances, min=0).norm(dim=-1)
        
        # Inside distance: max(distances) (negative when inside)
        inside_dist = torch.max(distances, dim=-1)[0]
        
        # Combine: positive when outside, negative when inside
        return torch.where(
            torch.all(distances <= 0, dim=-1),
            inside_dist,  # Inside box
            outside_dist  # Outside box
        )
    
    def _sdf_workspace(self, positions: torch.Tensor) -> torch.Tensor:
        """Signed distance to workspace boundaries"""
        batch_size = positions.shape[0]
        workspace_margins = []
        
        for dim in range(3):
            lower_bound, upper_bound = self.workspace_bounds[dim]
            
            # Distance to bounds in this dimension
            lower_margin = positions[:, dim] - lower_bound
            upper_margin = upper_bound - positions[:, dim]
            
            # Minimum margin (closest boundary)
            dim_margin = torch.minimum(lower_margin, upper_margin)
            workspace_margins.append(dim_margin)
        
        # Overall workspace constraint: minimum across dimensions
        return torch.min(torch.stack(workspace_margins), dim=0)[0]
    
    def check_trajectory_collisions(self, trajectory: torch.Tensor) -> Dict:
        """
        Check entire trajectory for collisions
        
        Args:
            trajectory: [T, 3] end-effector positions over time
            
        Returns:
            collision_info: Statistics about trajectory safety
        """
        T = trajectory.shape[0]
        sdf_values = self.compute_sdf(trajectory)
        
        # Collision detection (SDF < 0 means inside obstacle)
        collisions = sdf_values < 0
        collision_timesteps = torch.where(collisions)[0]
        
        # Safety margin violations (SDF < safety_margin)
        safety_margin = 0.05  # 5cm margin
        margin_violations = sdf_values < safety_margin
        margin_timesteps = torch.where(margin_violations)[0]
        
        return {
            'collision_rate': collisions.float().mean().item(),
            'collision_timesteps': collision_timesteps.tolist(),
            'min_sdf_value': sdf_values.min().item(),
            'mean_sdf_value': sdf_values.mean().item(),
            'margin_violation_rate': margin_violations.float().mean().item(),
            'margin_violation_timesteps': margin_timesteps.tolist(),
            'sdf_values': sdf_values
        }
    
    def visualize_sdf_slice(self, z_height: float = 0.5, resolution: int = 200, save_path: Optional[str] = None):
        """
        Visualize 2D slice of SDF field for debugging and validation
        """
        x_min, x_max = self.workspace_bounds[0]
        y_min, y_max = self.workspace_bounds[1]
        
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create 3D positions for the slice
        positions = torch.tensor(np.stack([
            X.flatten(), 
            Y.flatten(), 
            np.full(X.size, z_height)
        ], axis=1), dtype=torch.float32)
        
        # Compute SDF values
        sdf_values = self.compute_sdf(positions).numpy().reshape(resolution, resolution)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # SDF contour plot
        levels = np.linspace(sdf_values.min(), sdf_values.max(), 20)
        contour = plt.contourf(X, Y, sdf_values, levels=levels, cmap='RdBu')
        plt.colorbar(contour, label='SDF Value (m)')
        
        # Zero-level contour (obstacle boundaries)
        zero_contour = plt.contour(X, Y, sdf_values, levels=[0], colors='black', linewidths=3)
        plt.clabel(zero_contour, inline=True, fontsize=10, fmt='Obstacle Boundary')
        
        # Safety margin contour
        safety_contour = plt.contour(X, Y, sdf_values, levels=[0.05], colors='red', linewidths=2, linestyles='--')
        plt.clabel(safety_contour, inline=True, fontsize=8, fmt='Safety Margin')
        
        # Mark obstacle centers
        for i, obs in enumerate(self.obstacles):
            if obs['type'] == 'sphere':
                center = obs['center'].numpy()
                if abs(center[2] - z_height) < 0.1:  # Near the slice height
                    plt.plot(center[0], center[1], 'ko', markersize=10, markeredgewidth=2, markerfacecolor='white')
                    plt.text(center[0], center[1], f'Obs{i+1}', ha='center', va='center', fontsize=8)
        
        plt.title(f'SDF Field Visualization at z={z_height}m\\n'
                 f'{len(self.obstacles)} obstacles, Resolution: {resolution}×{resolution}')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 SDF visualization saved to {save_path}")
        
        plt.show()
    
    def create_dynamic_obstacle(self, initial_center: List[float], velocity: List[float], radius: float = 0.05):
        """
        Create a moving obstacle for dynamic avoidance scenarios
        
        Returns a function that gives obstacle position at time t
        """
        def obstacle_at_time(t: float) -> Dict:
            center = [
                initial_center[0] + velocity[0] * t,
                initial_center[1] + velocity[1] * t,
                initial_center[2] + velocity[2] * t
            ]
            return {
                'type': 'sphere',
                'center': torch.tensor(center, dtype=torch.float32),
                'radius': radius
            }
        
        return obstacle_at_time
    
    def get_environment_info(self) -> Dict:
        """Return summary of current environment configuration"""
        obstacle_summary = {}
        for i, obs in enumerate(self.obstacles):
            obstacle_summary[f'obstacle_{i}'] = {
                'type': obs['type'],
                'center': obs['center'].tolist(),
                'radius': obs.get('radius', None),
                'dimensions': obs.get('dimensions', torch.tensor([])).tolist()
            }
        
        return {
            'workspace_bounds': self.workspace_bounds,
            'num_obstacles': len(self.obstacles),
            'obstacles': obstacle_summary,
            'grid_resolution': self.grid_resolution
        }


def create_test_environment(scenario: str = 'cluttered') -> EnvironmentSDF:
    """
    Factory function to create test environments matching CADP paper scenarios
    """
    # Standard workspace for Franka Panda (from paper Section V.C)
    workspace_bounds = [
        (-0.5, 0.5),   # x bounds (m)
        (-0.5, 0.5),   # y bounds (m) 
        (0.0, 1.0)     # z bounds (m)
    ]
    
    env_sdf = EnvironmentSDF(workspace_bounds)
    env_sdf.add_static_obstacles_scenario(scenario)
    
    return env_sdf


if __name__ == "__main__":
    # Test SDF implementation
    print("🧪 Testing Environment SDF...")
    
    # Create test environment
    env = create_test_environment('cluttered')
    
    # Test single point queries
    test_points = torch.tensor([
        [0.0, 0.0, 0.5],    # Should be in free space
        [0.2, 0.3, 0.4],    # Near obstacle center
        [0.2, 0.3, 0.41],   # Inside first obstacle
        [2.0, 0.0, 0.5]     # Outside workspace
    ])
    
    sdf_values = env.compute_sdf(test_points)
    
    print(f"\\n📍 SDF Test Results:")
    for i, (point, sdf_val) in enumerate(zip(test_points, sdf_values)):
        safety_status = "🟢 SAFE" if sdf_val > 0.05 else ("🟡 MARGIN" if sdf_val > 0 else "🔴 COLLISION")
        print(f"   Point {i+1} {point.tolist()}: SDF = {sdf_val:.4f} {safety_status}")
    
    # Test trajectory collision checking
    T = 50
    test_trajectory = torch.randn(T, 3) * 0.2  # Random trajectory in workspace
    collision_info = env.check_trajectory_collisions(test_trajectory)
    
    print(f"\\n🛤️  Trajectory Analysis:")
    print(f"   • Collision rate: {collision_info['collision_rate']:.2%}")
    print(f"   • Margin violations: {collision_info['margin_violation_rate']:.2%}")
    print(f"   • Min SDF value: {collision_info['min_sdf_value']:.4f}")
    print(f"   • Mean SDF value: {collision_info['mean_sdf_value']:.4f}")
    
    # Visualize environment (optional)
    print(f"\\n📊 Creating SDF visualization...")
    env.visualize_sdf_slice(z_height=0.4, resolution=150)
    
    print("🎉 Environment SDF test completed!")