"""
CADP (Collision-Aware Diffusion Policy) Model
Author: CADP Project Team

Enhanced diffusion policy with physics-informed loss functions for safe robotic manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List
from .diffusion_model import RoboMimicDiffusionPolicy


class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss components for CADP"""
    
    def __init__(self, 
                 collision_weight: float = 0.1,
                 smoothness_weight: float = 0.05,
                 safety_margin: float = 0.05):
        super().__init__()
        self.collision_weight = collision_weight
        self.smoothness_weight = smoothness_weight
        self.safety_margin = safety_margin
        
    def collision_loss(self, actions: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Collision avoidance loss based on predicted robot configuration
        
        Args:
            actions: [batch_size, horizon, action_dim] - predicted actions
            observations: [batch_size, obs_dim] - current observations
            
        Returns:
            collision_loss: scalar tensor
        """
        batch_size, horizon, action_dim = actions.shape
        
        # Extract robot position from observations (assume first 3 dimensions are xyz)
        if observations.shape[-1] >= 3:
            robot_pos = observations[:, :3]  # [batch_size, 3]
        else:
            # Fallback: use zero position
            robot_pos = torch.zeros(batch_size, 3, device=actions.device)
            
        # Simulate forward kinematics approximately
        # For 7-DOF robot, actions represent joint velocities or positions
        predicted_positions = []
        
        for t in range(horizon):
            # Simple forward prediction: current_pos + action_delta
            if action_dim >= 3:
                # Use first 3 actions as position deltas
                pos_delta = actions[:, t, :3] * 0.01  # Scale factor
                next_pos = robot_pos + pos_delta
            else:
                next_pos = robot_pos
                
            predicted_positions.append(next_pos)
            robot_pos = next_pos
            
        predicted_positions = torch.stack(predicted_positions, dim=1)  # [batch_size, horizon, 3]
        
        # Collision detection with workspace boundaries
        workspace_bounds = torch.tensor([
            [-0.5, 0.5],   # x bounds
            [-0.5, 0.5],   # y bounds  
            [0.0, 1.0]     # z bounds
        ], device=actions.device)
        
        collision_penalty = 0.0
        
        for dim in range(3):
            # Lower bound violations
            lower_violations = F.relu(workspace_bounds[dim, 0] + self.safety_margin - predicted_positions[:, :, dim])
            # Upper bound violations  
            upper_violations = F.relu(predicted_positions[:, :, dim] - (workspace_bounds[dim, 1] - self.safety_margin))
            
            collision_penalty += (lower_violations + upper_violations).mean()
            
        # Self-collision penalty (prevent joint limits)
        if action_dim >= 7:
            joint_angles = actions.cumsum(dim=1)  # Approximate joint positions
            joint_limits = torch.tensor([-2.8, 2.8], device=actions.device)  # Typical joint limits
            
            joint_violations = (
                F.relu(-joint_limits[0] - joint_angles) + 
                F.relu(joint_angles - joint_limits[1])
            ).mean()
            
            collision_penalty += joint_violations
            
        return collision_penalty
    
    def smoothness_loss(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Action smoothness loss to ensure smooth robot motion
        
        Args:
            actions: [batch_size, horizon, action_dim]
            
        Returns:
            smoothness_loss: scalar tensor  
        """
        if actions.shape[1] < 2:
            return torch.tensor(0.0, device=actions.device)
            
        # First-order smoothness (velocity)
        action_diff = actions[:, 1:] - actions[:, :-1]
        velocity_penalty = (action_diff ** 2).mean()
        
        # Second-order smoothness (acceleration) 
        if actions.shape[1] >= 3:
            action_diff2 = action_diff[:, 1:] - action_diff[:, :-1]
            acceleration_penalty = (action_diff2 ** 2).mean()
        else:
            acceleration_penalty = 0.0
            
        return velocity_penalty + 0.5 * acceleration_penalty
    
    def forward(self, 
                actions: torch.Tensor, 
                observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all physics-informed loss components
        
        Returns:
            Dictionary containing individual loss components
        """
        collision_loss = self.collision_loss(actions, observations)
        smoothness_loss = self.smoothness_loss(actions)
        
        return {
            'collision_loss': collision_loss,
            'smoothness_loss': smoothness_loss,
            'total_physics_loss': (
                self.collision_weight * collision_loss + 
                self.smoothness_weight * smoothness_loss
            )
        }


class CADPModel(RoboMimicDiffusionPolicy):
    """
    CADP: Collision-Aware Diffusion Policy Model
    
    Extends the RoboMimicDiffusionPolicy with physics-informed loss functions
    for safe robotic manipulation.
    """
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 horizon: int,
                 # Base diffusion model parameters (match RoboMimicDiffusionPolicy)
                 hidden_dims: Optional[List[int]] = None,
                 time_embed_dim: int = 64,
                 cond_dim: int = 128,
                 num_diffusion_steps: int = 50,
                 # CADP-specific parameters
                 collision_weight: float = 0.1,
                 smoothness_weight: float = 0.05,
                 safety_margin: float = 0.05,
                 enable_safety: bool = True):
        
        # Initialize base diffusion model
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dims=hidden_dims,
            time_embed_dim=time_embed_dim,
            cond_dim=cond_dim,
            num_diffusion_steps=num_diffusion_steps
        )
        
        # Physics-informed loss module
        self.physics_loss = PhysicsInformedLoss(
            collision_weight=collision_weight,
            smoothness_weight=smoothness_weight,
            safety_margin=safety_margin
        )
        
        self.enable_safety = enable_safety
        self.collision_weight = collision_weight
        self.smoothness_weight = smoothness_weight
    
    def compute_diffusion_loss(self,
                              actions: torch.Tensor,
                              observations: torch.Tensor) -> torch.Tensor:
        """
        Compute standard diffusion loss (MSE between predicted and actual noise)
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        # Sample random timesteps
        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        
        # Add noise to actions
        noise = torch.randn_like(actions)
        
        # Simple noise scheduling (linear)
        alpha = 1.0 - timesteps.float() / self.num_diffusion_steps
        alpha = alpha.view(-1, 1, 1)  # [batch_size, 1, 1]
        
        noisy_actions = torch.sqrt(alpha) * actions + torch.sqrt(1 - alpha) * noise
        
        # Predict noise
        predicted_noise = self.forward(noisy_actions, timesteps, observations)
        
        # MSE loss between predicted and actual noise
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        
        return diffusion_loss
    
    def compute_loss(self,
                    actions: torch.Tensor,
                    observations: torch.Tensor,
                    **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute CADP total loss = diffusion loss + physics losses
        
        Args:
            actions: Ground truth actions [batch_size, horizon, action_dim]  
            observations: Current observations [batch_size, obs_dim]
            
        Returns:
            Dictionary containing all loss components
        """
        # Compute base diffusion loss
        diffusion_loss = self.compute_diffusion_loss(actions, observations)
        
        losses = {
            'diffusion_loss': diffusion_loss,
            'total_loss': diffusion_loss  # Default to diffusion only
        }
        
        if self.enable_safety:
            # Compute physics-informed losses
            physics_losses = self.physics_loss(actions, observations)
            
            # Combine all losses
            total_loss = (
                diffusion_loss + 
                physics_losses['total_physics_loss']
            )
            
            losses.update({
                'collision_loss': physics_losses['collision_loss'],
                'smoothness_loss': physics_losses['smoothness_loss'],
                'physics_loss': physics_losses['total_physics_loss'],
                'total_loss': total_loss
            })
            
        return losses
    
    def get_safety_metrics(self,
                          actions: torch.Tensor,
                          observations: torch.Tensor) -> Dict[str, float]:
        """
        Compute safety metrics for evaluation
        
        Returns:
            Dictionary with collision rate, smoothness score, etc.
        """
        with torch.no_grad():
            physics_losses = self.physics_loss(actions, observations)
            
            # Convert losses to interpretable metrics
            collision_rate = (physics_losses['collision_loss'] > 0.01).float().mean().item()
            smoothness_score = 1.0 / (1.0 + physics_losses['smoothness_loss'].item())
            
            return {
                'collision_rate': collision_rate,
                'smoothness_score': smoothness_score,
                'collision_loss_raw': physics_losses['collision_loss'].item(),
                'smoothness_loss_raw': physics_losses['smoothness_loss'].item()
            }
    
    def sample_actions(self, 
                      observations: torch.Tensor, 
                      num_samples: int = 1) -> torch.Tensor:
        """
        Sample actions using CADP diffusion model
        
        Args:
            observations: [batch_size, obs_dim] current observations
            num_samples: number of action samples to generate
            
        Returns:
            sampled_actions: [batch_size, horizon, action_dim]
        """
        self.eval()
        batch_size = observations.shape[0]
        device = observations.device
        
        with torch.no_grad():
            # Initialize with pure noise
            actions = torch.randn(batch_size, self.horizon, self.action_dim, device=device)
            
            # Reverse diffusion process
            for t in reversed(range(self.num_diffusion_steps)):
                timesteps = torch.full((batch_size,), t, device=device)
                
                # Predict noise at timestep t
                predicted_noise = self.forward(actions, timesteps, observations)
                
                # Denoise (simplified DDPM sampling)
                alpha = 1.0 - t / self.num_diffusion_steps
                alpha_tensor = torch.tensor(alpha, device=device)
                if t > 0:
                    noise = torch.randn_like(actions)
                else:
                    noise = 0
                
                # Denoising step
                actions = (actions - (1 - alpha) * predicted_noise) / torch.sqrt(alpha_tensor)
                if t > 0:
                    actions = actions + torch.sqrt(torch.tensor(1 - alpha, device=device)) * noise
                    
        return actions


def create_cadp_model(obs_dim: int,
                     action_dim: int, 
                     horizon: int,
                     device: torch.device,
                     **kwargs) -> CADPModel:
    """
    Factory function to create CADP model with default parameters
    """
    model = CADPModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=horizon,
        **kwargs
    )
    return model.to(device)


if __name__ == "__main__":
    # Test CADP model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_cadp_model(
        obs_dim=39,
        action_dim=7,
        horizon=16,
        device=device,
        collision_weight=0.1,
        smoothness_weight=0.05
    )
    
    print(f"CADP Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Physics-informed losses enabled: {model.enable_safety}")