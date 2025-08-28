"""
Key-Configuration Selection for Environment Representation
Implementation of Algorithm 1 from CADP paper

This module implements the key configuration selection algorithm that identifies
critical robot configurations near obstacles for efficient environment representation.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import random
from pathlib import Path

class KeyConfigurationSelector:
    """
    Key-Configuration Selection Algorithm
    
    Implements Algorithm 1 from the CADP paper for selecting critical
    configurations that efficiently represent environment constraints.
    """
    
    def __init__(
        self,
        robot_dof: int = 7,
        workspace_dim: int = 3,
        min_cspace_distance: float = 0.2,
        min_workspace_distance: float = 0.1,
        collision_proportion_bound: float = 0.3,
        num_environment_samples: int = 1000
    ):
        """
        Initialize Key Configuration Selector
        
        Args:
            robot_dof: Degrees of freedom of the robot
            workspace_dim: Dimension of workspace (typically 3D)
            min_cspace_distance: Minimum separation in configuration space (d_min_q)
            min_workspace_distance: Minimum separation in workspace (d_min_x)  
            collision_proportion_bound: Collision proportion bound (c)
            num_environment_samples: Number of samples for environment collision checking
        """
        self.dof = robot_dof
        self.workspace_dim = workspace_dim
        self.d_min_q = min_cspace_distance
        self.d_min_x = min_workspace_distance
        self.collision_bound = collision_proportion_bound
        self.M = num_environment_samples  # Number of environment samples for collision checking
        
        # Storage for key configurations
        self.key_configurations = []
        self.configuration_metadata = []
        
    def select_key_configurations(
        self,
        motion_dataset: List[Dict],
        num_key_configs: int,
        forward_kinematics_fn: Callable,
        collision_check_fn: Callable,
        max_attempts: int = 10000
    ) -> Tuple[List[torch.Tensor], List[Dict]]:
        """
        Select key configurations from motion planning dataset
        
        Implements Algorithm 1: Key-Configuration Selection
        
        Args:
            motion_dataset: List of motion plan instances
                Each instance: {
                    'trajectory': torch.Tensor (T, dof),
                    'start': torch.Tensor (dof,),
                    'goal': torch.Tensor (dof,), 
                    'task_goal': Any
                }
            num_key_configs: Number of key configurations to select (K)
            forward_kinematics_fn: Function to compute end-effector pose from joint config
            collision_check_fn: Function to check collision for given configuration
            max_attempts: Maximum attempts to find configurations
            
        Returns:
            key_configs: List of selected key configurations
            metadata: List of metadata for each configuration
        """
        print(f"Selecting {num_key_configs} key configurations...")
        
        # Initialize key configuration buffer
        key_configs = []
        metadata = []
        attempts = 0
        
        while len(key_configs) < num_key_configs and attempts < max_attempts:
            attempts += 1
            
            # Step 1: Sample a motion plan instance from dataset  
            motion_instance = random.choice(motion_dataset)
            trajectory = motion_instance['trajectory']
            
            # Step 2: Sample a configuration from the trajectory
            t_idx = random.randint(0, trajectory.shape[0] - 1)
            q_candidate = trajectory[t_idx].clone()
            
            # Step 3: Check minimum distance constraints
            
            # Compute minimum C-space distance
            d_q = self._min_cspace_distance(q_candidate, key_configs)
            
            # Compute minimum workspace distance
            d_x = self._min_workspace_distance(
                q_candidate, key_configs, forward_kinematics_fn
            )
            
            # Step 4: Check collision proportion
            p_c = self._collision_proportion(q_candidate, collision_check_fn)
            
            # Step 5: Accept configuration if it meets all criteria
            if (d_q >= self.d_min_q and 
                d_x >= self.d_min_x and 
                self.collision_bound <= p_c <= (1 - self.collision_bound)):
                
                key_configs.append(q_candidate)
                
                # Store metadata
                config_metadata = {
                    'config_index': len(key_configs) - 1,
                    'source_trajectory': motion_instance,
                    'trajectory_index': t_idx,
                    'cspace_distance': d_q,
                    'workspace_distance': d_x,
                    'collision_proportion': p_c,
                    'workspace_pose': forward_kinematics_fn(q_candidate)
                }
                metadata.append(config_metadata)
                
                if len(key_configs) % 10 == 0:
                    print(f"Selected {len(key_configs)}/{num_key_configs} configurations")
        
        if len(key_configs) < num_key_configs:
            print(f"Warning: Only found {len(key_configs)}/{num_key_configs} configurations "
                  f"after {max_attempts} attempts")
        else:
            print(f"Successfully selected {num_key_configs} key configurations")
        
        # Store for future use
        self.key_configurations = key_configs
        self.configuration_metadata = metadata
        
        return key_configs, metadata
    
    def _min_cspace_distance(
        self, 
        q_candidate: torch.Tensor, 
        existing_configs: List[torch.Tensor]
    ) -> float:
        """
        Compute minimum distance in configuration space
        
        Args:
            q_candidate: Candidate configuration
            existing_configs: List of existing key configurations
            
        Returns:
            Minimum distance to any existing configuration
        """
        if not existing_configs:
            return float('inf')  # No existing configurations
        
        distances = []
        for q_existing in existing_configs:
            # Euclidean distance in configuration space
            dist = torch.norm(q_candidate - q_existing).item()
            distances.append(dist)
        
        return min(distances)
    
    def _min_workspace_distance(
        self, 
        q_candidate: torch.Tensor,
        existing_configs: List[torch.Tensor],
        forward_kinematics_fn: Callable
    ) -> float:
        """
        Compute minimum distance in workspace
        
        Args:
            q_candidate: Candidate configuration  
            existing_configs: List of existing key configurations
            forward_kinematics_fn: Function to compute forward kinematics
            
        Returns:
            Minimum distance in workspace
        """
        if not existing_configs:
            return float('inf')
        
        # Compute workspace position for candidate
        try:
            pose_candidate = forward_kinematics_fn(q_candidate)
            # Extract position (first 3 elements typically)
            pos_candidate = pose_candidate[:3] if len(pose_candidate) > 3 else pose_candidate
        except Exception as e:
            print(f"Warning: Forward kinematics failed for candidate: {e}")
            return 0.0  # Conservative fallback
        
        distances = []
        for q_existing in existing_configs:
            try:
                pose_existing = forward_kinematics_fn(q_existing)
                pos_existing = pose_existing[:3] if len(pose_existing) > 3 else pose_existing
                
                # Euclidean distance in workspace
                dist = torch.norm(pos_candidate - pos_existing).item()
                distances.append(dist)
            except Exception as e:
                print(f"Warning: Forward kinematics failed for existing config: {e}")
                distances.append(0.0)  # Conservative fallback
        
        return min(distances) if distances else 0.0
    
    def _collision_proportion(
        self, 
        q: torch.Tensor, 
        collision_check_fn: Callable
    ) -> float:
        """
        Compute collision proportion for configuration with environment variations
        
        Args:
            q: Configuration to check
            collision_check_fn: Function to check collision
            
        Returns:
            Proportion of environment samples where configuration is in collision
        """
        collision_count = 0
        
        # Generate M different environment samples/variations
        for m in range(self.M):
            # Create slight environment variations by perturbing collision checker
            # This is a simplified implementation - in practice, would vary obstacle positions
            try:
                # Add small random perturbation to simulate environment variation
                q_perturbed = q + torch.randn_like(q) * 0.01
                
                is_collision = collision_check_fn(q_perturbed)
                if is_collision:
                    collision_count += 1
            except Exception as e:
                # If collision check fails, assume collision for safety
                collision_count += 1
        
        return collision_count / self.M
    
    def get_key_configuration_encoding(
        self, 
        query_config: torch.Tensor,
        encoding_dim: int = 128
    ) -> torch.Tensor:
        """
        Generate key configuration encoding for a query configuration
        
        Args:
            query_config: Configuration to encode
            encoding_dim: Dimension of output encoding
            
        Returns:
            Encoded representation based on key configurations
        """
        if not self.key_configurations:
            # Return zero encoding if no key configurations selected
            return torch.zeros(encoding_dim)
        
        # Compute distances to all key configurations
        distances = []
        for key_config in self.key_configurations:
            dist = torch.norm(query_config - key_config)
            distances.append(dist)
        
        distances = torch.tensor(distances)
        
        # Use RBF-like encoding: exp(-γ * d²)
        gamma = 0.5
        similarities = torch.exp(-gamma * distances**2)
        
        # Normalize to sum to 1
        similarities = similarities / (torch.sum(similarities) + 1e-8)
        
        # Pad or truncate to desired encoding dimension
        if len(similarities) >= encoding_dim:
            encoding = similarities[:encoding_dim]
        else:
            # Pad with zeros
            encoding = torch.zeros(encoding_dim)
            encoding[:len(similarities)] = similarities
        
        return encoding
    
    def save_key_configurations(self, filepath: str):
        """Save key configurations to file"""
        save_data = {
            'key_configurations': [config.cpu().numpy() for config in self.key_configurations],
            'metadata': self.configuration_metadata,
            'parameters': {
                'dof': self.dof,
                'd_min_q': self.d_min_q,
                'd_min_x': self.d_min_x,
                'collision_bound': self.collision_bound,
                'num_samples': self.M
            }
        }
        
        torch.save(save_data, filepath)
        print(f"Key configurations saved to {filepath}")
    
    def load_key_configurations(self, filepath: str):
        """Load key configurations from file"""
        save_data = torch.load(filepath)
        
        self.key_configurations = [
            torch.tensor(config) for config in save_data['key_configurations']
        ]
        self.configuration_metadata = save_data['metadata']
        
        # Update parameters if saved
        if 'parameters' in save_data:
            params = save_data['parameters']
            self.dof = params.get('dof', self.dof)
            self.d_min_q = params.get('d_min_q', self.d_min_q)
            self.d_min_x = params.get('d_min_x', self.d_min_x)
            self.collision_bound = params.get('collision_bound', self.collision_bound)
            self.M = params.get('num_samples', self.M)
        
        print(f"Loaded {len(self.key_configurations)} key configurations from {filepath}")

def create_key_configuration_selector(**kwargs) -> KeyConfigurationSelector:
    """
    Factory function to create key configuration selector
    """
    default_params = {
        'robot_dof': 7,
        'workspace_dim': 3,
        'min_cspace_distance': 0.2,
        'min_workspace_distance': 0.1,
        'collision_proportion_bound': 0.3,
        'num_environment_samples': 100  # Reduced for efficiency
    }
    
    default_params.update(kwargs)
    return KeyConfigurationSelector(**default_params)

# Simplified implementations for testing
def dummy_forward_kinematics(q: torch.Tensor) -> torch.Tensor:
    """
    Dummy forward kinematics for testing
    Returns simplified end-effector position
    """
    # Simple approximation: end-effector position as weighted sum of joint angles
    weights = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02])
    if len(q) != len(weights):
        weights = weights[:len(q)]
    
    # Approximate end-effector position
    x = torch.sum(weights * torch.cos(q))
    y = torch.sum(weights * torch.sin(q))
    z = torch.sum(weights * q) * 0.1  # Height approximation
    
    return torch.tensor([x, y, z])

def dummy_collision_checker(q: torch.Tensor) -> bool:
    """
    Dummy collision checker for testing
    Returns collision based on joint configuration
    """
    # Simple heuristic: collision if any joint exceeds certain limits
    joint_limits = torch.pi * 0.8  # 80% of full range
    
    # Check if any joint is near limits (simplified collision detection)
    near_limits = torch.any(torch.abs(q) > joint_limits)
    
    # Add some randomness to simulate environment variation
    random_collision = torch.rand(1).item() < 0.1  # 10% random collision rate
    
    return near_limits.item() or random_collision