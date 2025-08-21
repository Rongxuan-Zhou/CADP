"""
Diffusion Policy Model for RoboMimic
Author: CADP Project Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalResidualBlock1D(nn.Module):
    """1D Residual block with condition injection"""
    
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.activation = nn.SiLU()
        
        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Add condition
        cond_proj = self.cond_proj(cond)
        out = out + cond_proj.unsqueeze(-1)
        
        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        
        return self.activation(out + residual)


class RoboMimicDiffusionPolicy(nn.Module):
    """
    Diffusion Policy for RoboMimic Low-Dimensional Data
    Optimized for RTX 4070 with reduced model size
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 horizon: int,
                 hidden_dims: List[int] = None,
                 time_embed_dim: int = 64,
                 cond_dim: int = 128,
                 num_diffusion_steps: int = 50):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 128]  # Reduced from [128, 256, 256]
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Observation encoder with dropout
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, cond_dim // 2),
            nn.LayerNorm(cond_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(cond_dim // 2, cond_dim),
            nn.LayerNorm(cond_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Combine time and observation embeddings
        self.cond_combine = nn.Linear(time_embed_dim + cond_dim, cond_dim)
        
        # U-Net encoder
        self.encoder_blocks = nn.ModuleList()
        in_dim = action_dim
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(
                ConditionalResidualBlock1D(in_dim, hidden_dim, cond_dim)
            )
            in_dim = hidden_dim
        
        # U-Net decoder
        self.decoder_blocks = nn.ModuleList()
        for i, hidden_dim in enumerate(reversed(hidden_dims)):
            if i == 0:
                # First decoder block
                self.decoder_blocks.append(
                    ConditionalResidualBlock1D(hidden_dim, hidden_dim, cond_dim)
                )
            else:
                # Skip connections from encoder
                skip_dim = hidden_dims[len(hidden_dims) - i]
                self.decoder_blocks.append(
                    ConditionalResidualBlock1D(hidden_dim + skip_dim, hidden_dim, cond_dim)
                )
        
        # Final output projection
        self.final_conv = nn.Conv1d(hidden_dims[0], action_dim, 1)
        
        print(f"\nModel initialized:")
        print(f"  - Obs dim: {obs_dim}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Horizon: {horizon}")
        print(f"  - Hidden dims: {hidden_dims}")
        print(f"  - Diffusion steps: {num_diffusion_steps}")
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  - Total parameters: {total_params/1e6:.2f}M")
    
    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        noisy_actions: [B, horizon, action_dim]
        timesteps: [B]
        observations: [B, obs_dim]
        """
        batch_size = noisy_actions.shape[0]
        
        # Embed time and observations
        time_embed = self.time_embed(timesteps)
        obs_embed = self.obs_encoder(observations)
        
        # Combine conditions
        cond = self.cond_combine(torch.cat([time_embed, obs_embed], dim=-1))
        
        # Convert to [B, action_dim, horizon] for 1D conv
        x = noisy_actions.transpose(1, 2)
        
        # Encoder with skip connections
        encoder_features = []
        for block in self.encoder_blocks:
            x = block(x, cond)
            encoder_features.append(x)
        
        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            if i > 0:
                # Add skip connection
                skip_feat = encoder_features[-(i+1)]
                x = torch.cat([x, skip_feat], dim=1)
            x = block(x, cond)
        
        # Final prediction
        x = self.final_conv(x)
        
        # Convert back to [B, horizon, action_dim]
        return x.transpose(1, 2)


class DDPMScheduler:
    """DDPM noise scheduler"""
    
    def __init__(self, num_train_timesteps: int = 50, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For sampling
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples according to timesteps"""
        device = original_samples.device
        alphas_cumprod = self.alphas_cumprod.to(device)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples


def create_model(obs_dim: int, action_dim: int, horizon: int, device: str = 'cuda', 
                 hidden_dims: List[int] = None, time_embed_dim: int = 64, 
                 cond_dim: int = 128, **kwargs) -> RoboMimicDiffusionPolicy:
    """Create and initialize the diffusion model with configurable parameters"""
    
    # Handle extended configuration
    if hidden_dims is None:
        hidden_dims = [64, 128, 128]  # Default configuration
    
    model = RoboMimicDiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dims=hidden_dims,
        time_embed_dim=time_embed_dim,
        cond_dim=cond_dim,
        **kwargs
    ).to(device)
    
    return model