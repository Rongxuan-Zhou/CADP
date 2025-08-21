"""
Training Manager for RoboMimic Diffusion Policy
Author: CADP Project Team
"""

import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output
from typing import Dict, Any, Optional, List

from ..models.diffusion_model import DDPMScheduler


class RoboMimicTrainer:
    """Training manager for RoboMimic diffusion policy"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=model.num_diffusion_steps)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['lr'] * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop_early = False
        
        print(f"\nTrainer initialized:")
        print(f"  - Learning rate: {config['lr']}")
        print(f"  - Weight decay: {config['weight_decay']}")
        print(f"  - Gradient accumulation: {config['gradient_accumulation']}")
        print(f"  - Early stopping patience: {self.early_stopping_patience}")
        print(f"  - Mixed precision: Enabled")
    
    def train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        accumulated_loss = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            obs = batch['observation'].to(self.device, non_blocking=True)
            actions = batch['action'].to(self.device, non_blocking=True)
            
            batch_size = actions.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.scheduler.num_train_timesteps, 
                (batch_size,), device=self.device
            )
            
            # Sample noise
            noise = torch.randn_like(actions)
            
            # Add noise to actions
            noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
            
            # Predict noise
            with autocast():
                pred_noise = self.model(noisy_actions, timesteps, obs)
                loss = F.mse_loss(pred_noise, noise)
                loss = loss / self.config['gradient_accumulation']
            
            # Backward pass
            self.scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
            # Update weights
            if (batch_idx + 1) % self.config['gradient_accumulation'] == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Record loss
                actual_loss = accumulated_loss * self.config['gradient_accumulation']
                epoch_losses.append(actual_loss)
                accumulated_loss = 0
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{actual_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB'
                })
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """Validate model"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                obs = batch['observation'].to(self.device, non_blocking=True)
                actions = batch['action'].to(self.device, non_blocking=True)
                
                batch_size = actions.shape[0]
                timesteps = torch.randint(
                    0, self.scheduler.num_train_timesteps, 
                    (batch_size,), device=self.device
                )
                noise = torch.randn_like(actions)
                noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
                
                with autocast():
                    pred_noise = self.model(noisy_actions, timesteps, obs)
                    loss = F.mse_loss(pred_noise, noise)
                
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        self.val_losses.append(avg_val_loss)
        
        # Early stopping logic
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered! No improvement for {self.early_stopping_patience} validations")
                self.should_stop_early = True
        
        return avg_val_loss
    
    def plot_progress(self, save_path: Optional[str] = None, show_plot: bool = False):
        """Plot training progress"""
        if show_plot:
            clear_output(wait=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        if self.val_losses:
            val_epochs = np.arange(len(self.val_losses)) * len(self.train_losses) // len(self.val_losses)
            axes[0].plot(val_epochs, self.val_losses, 'o-', label='Val Loss', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Learning rate
        axes[1].plot(self.learning_rates, alpha=0.8, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # GPU memory usage
        current_memory = torch.cuda.memory_allocated() / 1024**3
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        axes[2].bar(['Current', 'Peak'], [current_memory, max_memory], alpha=0.7)
        axes[2].set_ylabel('Memory (GB)')
        axes[2].set_title('GPU Memory Usage')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()  # Automatically close the figure without showing
    
    def save_checkpoint(self, filepath: str, epoch: int, val_loss: Optional[float] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config,
            'val_loss': val_loss
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        
        print(f"✓ Checkpoint loaded from: {filepath}")
        return checkpoint


def train_model(model: torch.nn.Module,
                train_loader,
                val_loader,
                config: Dict[str, Any],
                save_dir: str = 'checkpoints') -> RoboMimicTrainer:
    """Complete training loop"""
    
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    trainer = RoboMimicTrainer(model, config)
    
    best_val_loss = float('inf')
    
    # Create checkpoints directory
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validation
        if (epoch + 1) % config['validate_every'] == 0:
            val_loss = trainer.validate(val_loader)
            print(f"Validation Loss: {val_loss:.6f}")
            print(f"Best Val Loss: {trainer.best_val_loss:.6f} (Patience: {trainer.patience_counter}/{trainer.early_stopping_patience})")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(
                    os.path.join(save_dir, 'best_model.pt'), 
                    epoch, 
                    val_loss
                )
            
            # Check early stopping
            if trainer.should_stop_early:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Update learning rate
        trainer.lr_scheduler.step()
        
        # Periodic saves
        if (epoch + 1) % config['save_every'] == 0:
            trainer.save_checkpoint(
                os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'),
                epoch
            )
        
        # Plot progress (save only, don't show during training)
        if (epoch + 1) % 2 == 0:  # Plot every 2 epochs
            plot_path = os.path.join(save_dir, f'training_progress_epoch_{epoch+1}.png')
            trainer.plot_progress(save_path=plot_path, show_plot=False)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print("="*60)
    
    # Show final training progress plot
    print("📊 Displaying final training results...")
    final_plot_path = os.path.join(save_dir, 'final_training_results.png')
    trainer.plot_progress(save_path=final_plot_path, show_plot=True)
    
    # Final save
    trainer.save_checkpoint(os.path.join(save_dir, 'final_model.pt'), epoch)
    
    return trainer