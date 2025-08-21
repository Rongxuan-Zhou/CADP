"""
Enhanced Diffusion Policy Model with Maximum Capacity
Author: CADP Project Team

Extended version supporting larger architectures and advanced features
for pushing performance to device limits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Enhanced sinusoidal position embeddings"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention for sequence modeling"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out


class EnhancedResidualBlock1D(nn.Module):
    """Enhanced residual block with attention and advanced features"""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 cond_dim: int, 
                 kernel_size: int = 3,
                 use_attention: bool = False,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Normalization
        if use_layer_norm:
            self.norm1 = nn.GroupNorm(1, out_channels)  # Layer norm equivalent
            self.norm2 = nn.GroupNorm(1, out_channels)
        else:
            self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        
        # Self-attention
        self.use_attention = use_attention
        if use_attention and out_channels >= num_heads:
            self.attention = MultiHeadSelfAttention(out_channels, num_heads, dropout)
            self.attn_norm = nn.GroupNorm(1, out_channels)
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        
        # First convolution
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Add condition
        cond_proj = self.cond_proj(cond)
        out = out + cond_proj.unsqueeze(-1)
        
        # Second convolution
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Self-attention
        if self.use_attention and hasattr(self, 'attention'):
            # Convert to sequence format for attention
            B, C, T = out.shape
            attn_out = out.transpose(1, 2)  # (B, T, C)
            attn_out = self.attention(attn_out)
            attn_out = attn_out.transpose(1, 2)  # (B, C, T)
            
            out = self.attn_norm(out + attn_out)
        
        return self.activation(out + residual)


class EnhancedDiffusionPolicy(nn.Module):
    """
    Enhanced Diffusion Policy with maximum capacity and advanced features
    """
    
    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 horizon: int,
                 hidden_dims: List[int] = None,
                 time_embed_dim: int = 256,
                 cond_dim: int = 512,
                 num_diffusion_steps: int = 100,
                 use_attention: bool = True,
                 num_heads: int = 8,
                 dropout: float = 0.15,
                 use_layer_norm: bool = True,
                 use_residual_connections: bool = True,
                 num_resnet_blocks: int = 2,
                 **kwargs):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]  # Larger default
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.use_attention = use_attention
        
        # Enhanced time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(time_embed_dim * 4, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim)
        )
        
        # Enhanced observation encoder
        obs_hidden = cond_dim // 2
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, obs_hidden),
            nn.LayerNorm(obs_hidden) if use_layer_norm else nn.BatchNorm1d(obs_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(obs_hidden, obs_hidden),
            nn.LayerNorm(obs_hidden) if use_layer_norm else nn.BatchNorm1d(obs_hidden), 
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(obs_hidden, cond_dim),
            nn.LayerNorm(cond_dim) if use_layer_norm else nn.BatchNorm1d(cond_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Combine time and observation embeddings
        self.cond_combine = nn.Sequential(
            nn.Linear(time_embed_dim + cond_dim, cond_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # Enhanced U-Net encoder with multiple blocks per level
        self.encoder_blocks = nn.ModuleList()
        in_dim = action_dim
        for hidden_dim in hidden_dims:
            level_blocks = nn.ModuleList()
            for i in range(num_resnet_blocks):
                level_blocks.append(
                    EnhancedResidualBlock1D(
                        in_dim if i == 0 else hidden_dim, 
                        hidden_dim, 
                        cond_dim,
                        use_attention=use_attention and hidden_dim >= num_heads,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_layer_norm=use_layer_norm
                    )
                )
            self.encoder_blocks.append(level_blocks)
            in_dim = hidden_dim
        
        # Enhanced U-Net decoder with proper skip connections
        self.decoder_blocks = nn.ModuleList()
        decoder_dims = list(reversed(hidden_dims))
        
        for i, hidden_dim in enumerate(decoder_dims):
            level_blocks = nn.ModuleList()
            for j in range(num_resnet_blocks):
                if i == 0 and j == 0:
                    # First decoder block - no skip connection
                    input_dim = hidden_dim
                elif j == 0 and i > 0:
                    # First block of level with skip connection
                    skip_dim = hidden_dims[len(hidden_dims) - i]  # Corresponding encoder dimension
                    input_dim = hidden_dim + skip_dim
                else:
                    # Subsequent blocks in same level
                    input_dim = hidden_dim
                    
                level_blocks.append(
                    EnhancedResidualBlock1D(
                        input_dim,
                        hidden_dim, 
                        cond_dim,
                        use_attention=use_attention and hidden_dim >= num_heads,
                        num_heads=num_heads,
                        dropout=dropout,
                        use_layer_norm=use_layer_norm
                    )
                )
            self.decoder_blocks.append(level_blocks)
        
        # Final output projection with residual
        self.final_conv = nn.Sequential(
            nn.Conv1d(hidden_dims[0], hidden_dims[0] // 2, 1),
            nn.GroupNorm(1, hidden_dims[0] // 2) if use_layer_norm else nn.GroupNorm(min(8, hidden_dims[0] // 2), hidden_dims[0] // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(hidden_dims[0] // 2, action_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"\nEnhanced Model initialized:")
        print(f"  - Obs dim: {obs_dim}")
        print(f"  - Action dim: {action_dim}")
        print(f"  - Horizon: {horizon}")
        print(f"  - Hidden dims: {hidden_dims}")
        print(f"  - Time embed dim: {time_embed_dim}")
        print(f"  - Condition dim: {cond_dim}")
        print(f"  - Use attention: {use_attention}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Diffusion steps: {num_diffusion_steps}")
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  - Total parameters: {total_params/1e6:.2f}M")
        print(f"  - Trainable parameters: {trainable_params/1e6:.2f}M")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass
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
        for level_blocks in self.encoder_blocks:
            for block in level_blocks:
                x = block(x, cond)
            encoder_features.append(x)
        
        # Decoder with skip connections  
        for i, level_blocks in enumerate(self.decoder_blocks):
            for j, block in enumerate(level_blocks):
                if i > 0 and j == 0:
                    # Add skip connection from corresponding encoder level
                    skip_feat = encoder_features[-(i+1)]
                    x = torch.cat([x, skip_feat], dim=1)
                x = block(x, cond)
        
        # Final prediction
        x = self.final_conv(x)
        
        # Convert back to [B, horizon, action_dim]
        return x.transpose(1, 2)


def create_enhanced_model(obs_dim: int, action_dim: int, horizon: int, device: str = 'cuda', **kwargs) -> EnhancedDiffusionPolicy:
    """Create enhanced diffusion model with maximum capacity"""
    
    model = EnhancedDiffusionPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        horizon=horizon,
        **kwargs
    ).to(device)
    
    return model