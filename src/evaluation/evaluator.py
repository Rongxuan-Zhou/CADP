"""
Evaluation utilities for RoboMimic Diffusion Policy
Author: CADP Project Team
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional

from ..models.diffusion_model import DDPMScheduler


def ddmp_sample(model: torch.nn.Module, 
               scheduler: DDPMScheduler, 
               observations: torch.Tensor, 
               num_inference_steps: int = 25) -> torch.Tensor:
    """
    DDPM sampling for action generation
    Reduced inference steps for faster evaluation
    """
    device = next(model.parameters()).device
    batch_size = observations.shape[0]
    
    # Start from random noise
    shape = (batch_size, model.horizon, model.action_dim)
    actions = torch.randn(shape, device=device)
    
    # Sampling timesteps
    timesteps = torch.linspace(
        scheduler.num_train_timesteps - 1, 0, 
        num_inference_steps, dtype=torch.long, device=device
    )
    
    model.eval()
    with torch.no_grad():
        for t in tqdm(timesteps, desc='Sampling'):
            # Predict noise
            timestep_batch = t.repeat(batch_size)
            
            with autocast():
                pred_noise = model(actions, timestep_batch, observations)
            
            # DDPM sampling step
            alpha_t = scheduler.alphas[t]
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod_prev[t]
            
            # Predict original sample
            pred_original = (actions - (1 - alpha_prod_t) ** 0.5 * pred_noise) / (alpha_prod_t ** 0.5)
            
            # Compute previous sample mean
            pred_prev_sample = (
                (alpha_prod_t_prev ** 0.5) * pred_original + 
                ((1 - alpha_prod_t_prev) ** 0.5) * pred_noise
            )
            
            # Add noise for non-final steps
            if t > 0:
                noise = torch.randn_like(actions)
                beta_t = scheduler.betas[t]
                variance = beta_t * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
                pred_prev_sample += (variance ** 0.5) * noise
            
            actions = pred_prev_sample
    
    return actions


def evaluate_model(model: torch.nn.Module, 
                  val_loader, 
                  dataset, 
                  num_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Comprehensive model evaluation"""
    device = next(model.parameters()).device
    scheduler = DDPMScheduler(num_train_timesteps=model.num_diffusion_steps)
    
    all_predictions = []
    all_targets = []
    all_observations = []
    
    print(f"\nEvaluating model on {num_samples} samples...")
    
    model.eval()
    samples_collected = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if samples_collected >= num_samples:
                break
                
            obs = batch['observation'].to(device)
            actions = batch['action'].to(device)
            
            # Generate predictions
            predicted_actions = ddmp_sample(model, scheduler, obs, num_inference_steps=15)
            
            all_predictions.append(predicted_actions.cpu())
            all_targets.append(actions.cpu())
            all_observations.append(obs.cpu())
            
            samples_collected += obs.shape[0]
    
    # Concatenate results
    predictions = torch.cat(all_predictions, dim=0)[:num_samples]
    targets = torch.cat(all_targets, dim=0)[:num_samples]
    observations = torch.cat(all_observations, dim=0)[:num_samples]
    
    # Denormalize if needed
    if hasattr(dataset, 'normalize') and dataset.normalize:
        predictions = predictions * torch.tensor(dataset.action_std) + torch.tensor(dataset.action_mean)
        targets = targets * torch.tensor(dataset.action_std) + torch.tensor(dataset.action_mean)
    
    # Calculate metrics
    mse = F.mse_loss(predictions, targets)
    mae = F.l1_loss(predictions, targets)
    
    # Per-dimension metrics
    dim_mse = F.mse_loss(predictions, targets, reduction='none').mean(dim=[0, 1])
    dim_mae = F.l1_loss(predictions, targets, reduction='none').mean(dim=[0, 1])
    
    print(f"\nEvaluation Results:")
    print(f"  - Overall MSE: {mse:.6f}")
    print(f"  - Overall MAE: {mae:.6f}")
    print(f"\nPer-dimension MSE: {dim_mse.numpy()}")
    print(f"Per-dimension MAE: {dim_mae.numpy()}")
    
    return predictions, targets, observations, {
        'mse': mse.item(), 'mae': mae.item(),
        'dim_mse': dim_mse.numpy(), 'dim_mae': dim_mae.numpy()
    }


def visualize_results(predictions: torch.Tensor, 
                     targets: torch.Tensor, 
                     observations: torch.Tensor, 
                     metrics: Dict[str, Any], 
                     num_plots: int = 6,
                     save_path: Optional[str] = None):
    """Visualize evaluation results"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplots
    num_rows = 3
    num_cols = 4
    
    # Plot individual trajectories
    for i in range(min(num_plots, len(predictions))):
        ax = fig.add_subplot(num_rows, num_cols, i+1)
        
        # Plot first 3 action dimensions
        for j in range(min(3, predictions.shape[-1])):
            ax.plot(targets[i, :, j].numpy(), 
                   label=f'True Dim {j}', 
                   linestyle='--', alpha=0.8, linewidth=2)
            ax.plot(predictions[i, :, j].numpy(), 
                   label=f'Pred Dim {j}', 
                   alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Action Value')
        ax.set_title(f'Sample {i+1}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    # Error analysis plots
    error = predictions - targets
    
    # Per-dimension error distribution
    ax_error_dist = fig.add_subplot(num_rows, num_cols, num_plots + 1)
    for j in range(min(3, error.shape[-1])):
        errors_j = error[:, :, j].flatten().numpy()
        ax_error_dist.hist(errors_j, alpha=0.6, bins=30, label=f'Dim {j}')
    ax_error_dist.set_xlabel('Prediction Error')
    ax_error_dist.set_ylabel('Frequency')
    ax_error_dist.set_title('Error Distribution')
    ax_error_dist.legend()
    ax_error_dist.grid(True, alpha=0.3)
    
    # Error over time
    ax_error_time = fig.add_subplot(num_rows, num_cols, num_plots + 2)
    time_errors = torch.abs(error).mean(dim=[0, 2]).numpy()  # Mean over samples and dimensions
    ax_error_time.plot(time_errors, 'o-', alpha=0.8)
    ax_error_time.set_xlabel('Time Step')
    ax_error_time.set_ylabel('Mean Absolute Error')
    ax_error_time.set_title('Error vs Time')
    ax_error_time.grid(True, alpha=0.3)
    
    # Metrics summary
    ax_metrics = fig.add_subplot(num_rows, num_cols, num_plots + 3)
    metric_names = ['MSE', 'MAE']
    metric_values = [metrics['mse'], metrics['mae']]
    bars = ax_metrics.bar(metric_names, metric_values, alpha=0.7)
    ax_metrics.set_ylabel('Metric Value')
    ax_metrics.set_title('Overall Metrics')
    ax_metrics.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                       f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Evaluation results saved: {save_path}")
    plt.close()  # Automatically close the figure


def save_model_for_inference(model: torch.nn.Module, 
                            dataset, 
                            filepath: str = 'robomimic_baseline_model.pt'):
    """Save model with all necessary information for inference"""
    
    model_info = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'obs_dim': model.obs_dim,
            'action_dim': model.action_dim,
            'horizon': model.horizon,
            'num_diffusion_steps': model.num_diffusion_steps,
        },
        'dataset_info': {
            'obs_keys': dataset.obs_keys,
            'action_mean': dataset.action_mean if hasattr(dataset, 'action_mean') else None,
            'action_std': dataset.action_std if hasattr(dataset, 'action_std') else None,
            'obs_mean': dataset.obs_mean if hasattr(dataset, 'obs_mean') else None,
            'obs_std': dataset.obs_std if hasattr(dataset, 'obs_std') else None,
            'normalize': dataset.normalize if hasattr(dataset, 'normalize') else False,
        }
    }
    
    torch.save(model_info, filepath)
    print(f"✓ Model saved for inference: {filepath}")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Summary:")
    print(f"  - Parameters: {total_params/1e6:.2f}M")
    print(f"  - Observation dim: {model.obs_dim}")
    print(f"  - Action dim: {model.action_dim}")
    print(f"  - Horizon: {model.horizon}")
    print(f"  - Diffusion steps: {model.num_diffusion_steps}")


def load_model_for_inference(filepath: str, device: str = 'cuda'):
    """Load model for inference"""
    from ..models.diffusion_model import RoboMimicDiffusionPolicy
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create model
    model_config = checkpoint['model_config']
    model = RoboMimicDiffusionPolicy(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    dataset_info = checkpoint['dataset_info']
    
    print(f"✓ Model loaded from {filepath}")
    
    return model, dataset_info


def run_evaluation(model: torch.nn.Module, 
                  val_loader, 
                  dataset, 
                  save_dir: str = 'results') -> Dict[str, Any]:
    """Complete evaluation pipeline"""
    
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Run evaluation
    predictions, targets, observations, metrics = evaluate_model(
        model, val_loader, dataset, num_samples=30
    )
    
    # Visualize results
    plot_path = os.path.join(save_dir, 'evaluation_results.png')
    visualize_results(predictions, targets, observations, metrics, save_path=plot_path)
    
    # Save model for inference
    model_path = os.path.join(save_dir, 'robomimic_baseline_model.pt')
    save_model_for_inference(model, dataset, model_path)
    
    return metrics