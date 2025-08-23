#!/usr/bin/env python3
"""
CADP Training Script
Author: CADP Project Team

Train the Collision-Aware Diffusion Policy on the successful vanilla baseline.
"""

import os
import sys
import argparse
import torch
import warnings
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.multi_task_dataset import create_multi_task_dataloaders
from src.models.cadp_model import create_cadp_model
from src.training.trainer import train_model
from src.evaluation.evaluator import run_evaluation


class CADPTrainer:
    """CADP training manager"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load successful baseline configuration
        self.baseline_config = {
            'stage_2': {
                'tasks': ['lift', 'can', 'square'],
                'data_type': 'ph',
                'model_params': 2.67e6,
                'baseline_success_rate': 0.70
            },
            'stage_4': {
                'tasks': ['lift', 'can', 'square'], 
                'data_type': 'mh',
                'model_params': 2.67e6,
                'baseline_success_rate': 0.70
            }
        }
        
        # Task configurations
        self.task_configs = {
            'ph': {
                'lift': 'data/robomimic_lowdim/ph/lift_ph_lowdim.hdf5',
                'can': 'data/robomimic_lowdim/ph/can_ph_lowdim.hdf5',
                'square': 'data/robomimic_lowdim/ph/square_ph_lowdim.hdf5',
                'toolhang': 'data/robomimic_lowdim/ph/toolhang_ph_lowdim.hdf5'
            },
            'mh': {
                'lift': 'data/robomimic_lowdim/mh/lift_mh_lowdim.hdf5',
                'can': 'data/robomimic_lowdim/mh/can_mh_lowdim.hdf5',
                'square': 'data/robomimic_lowdim/mh/square_mh_lowdim.hdf5'
            }
        }
        
        print(f"🚀 CADP Training Initialized")
        print(f"   Device: {self.device}")
        print(f"   Base model: Vanilla Diffusion Policy (73.8% success rate)")
        
    def load_baseline_checkpoint(self, stage: str) -> dict:
        """Load successful baseline checkpoint"""
        if stage == 'stage_2':
            checkpoint_path = 'checkpoints_stage_2_ph/best_model.pt'
        elif stage == 'stage_4':
            checkpoint_path = 'checkpoints_stage_4_mh/best_model.pt'
        else:
            raise ValueError(f"Unknown stage: {stage}")
            
        if os.path.exists(checkpoint_path):
            print(f"📁 Loading baseline checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return checkpoint
        else:
            print(f"⚠️  Baseline checkpoint not found: {checkpoint_path}")
            return None
    
    def train_cadp_stage(self, stage: str):
        """Train CADP on a specific stage"""
        
        print("\\n" + "="*80)
        print(f"🎯 CADP Training: {stage.upper()}")
        print("="*80)
        
        config = self.baseline_config[stage]
        
        # Prepare task configurations
        selected_tasks = {}
        data_type_configs = self.task_configs[config['data_type']]
        
        for task in config['tasks']:
            if task in data_type_configs:
                selected_tasks[task] = data_type_configs[task]
        
        print(f"   Tasks: {config['tasks']}")
        print(f"   Data type: {config['data_type'].upper()}")
        print(f"   Baseline success rate: {config['baseline_success_rate']:.1%}")
        
        # Create dataloaders
        train_loader, val_loader, dataset = create_multi_task_dataloaders(
            task_configs=selected_tasks,
            horizon=20,  # Slightly increased for better planning
            batch_size=16,
            max_demos=self.args.max_demos,
            train_ratio=0.85,
            curriculum_learning=False  # Disable for focused CADP training
        )
        
        # Get observation dimension
        max_obs_dim = max(dataset.datasets[task].obs_dim for task in dataset.datasets.keys())
        
        # Create CADP model
        cadp_model = create_cadp_model(
            obs_dim=max_obs_dim,
            action_dim=7,
            horizon=20,
            device=self.device,
            hidden_dims=[128, 256, 256],
            time_embed_dim=128,
            cond_dim=256,
            num_diffusion_steps=100,
            collision_weight=self.args.collision_weight,
            smoothness_weight=self.args.smoothness_weight,
            safety_margin=0.05,
            enable_safety=True
        )
        
        # Load baseline weights if available
        baseline_checkpoint = self.load_baseline_checkpoint(stage)
        if baseline_checkpoint and 'model_state_dict' in baseline_checkpoint:
            try:
                # Load baseline weights (CADP inherits from RoboMimicDiffusionPolicy)
                base_state_dict = baseline_checkpoint['model_state_dict']
                cadp_model.load_state_dict(base_state_dict, strict=False)
                print("✅ Loaded baseline weights into CADP model")
            except Exception as e:
                print(f"⚠️  Could not load baseline weights: {e}")
        
        total_params = sum(p.numel() for p in cadp_model.parameters())
        print(f"\\n🧠 CADP Model Configuration:")
        print(f"   • Observation dimension: {max_obs_dim}")
        print(f"   • Action dimension: 7") 
        print(f"   • Horizon: 20")
        print(f"   • Total parameters: {total_params/1e6:.2f}M")
        print(f"   • Collision weight: {self.args.collision_weight}")
        print(f"   • Smoothness weight: {self.args.smoothness_weight}")
        
        # Training configuration
        training_config = {
            'num_epochs': self.args.epochs,
            'lr': 1e-4,  # Conservative for safety fine-tuning
            'lr_schedule': 'cosine',
            'lr_min': 1e-6,
            'lr_warmup_epochs': 10,
            'weight_decay': 3e-4,
            'gradient_accumulation': 2,
            'clip_grad_norm': 0.5,  # More aggressive clipping for stability
            'ema_decay': 0.9995,
            'use_ema': True,
            'validate_every': 3,
            'save_every': 20,
            'early_stopping_patience': 25,
            'use_mixed_precision': True,
        }
        
        # Save directory
        save_dir = f"checkpoints_cadp_{stage}_{config['data_type']}"
        results_dir = f"results_cadp_{stage}_{config['data_type']}"
        
        print(f"\\n⚙️  CADP Training Configuration:")
        print(f"   • Epochs: {training_config['num_epochs']}")
        print(f"   • Learning rate: {training_config['lr']}")
        print(f"   • Early stopping patience: {training_config['early_stopping_patience']}")
        print(f"   • Physics-informed loss: Enabled")
        
        # Start training
        print(f"\\n🚀 Starting CADP training...")
        
        # Custom trainer function that handles CADP loss
        trainer = train_cadp_model(
            model=cadp_model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            save_dir=save_dir
        )
        
        # Evaluation
        print(f"\\n📊 CADP evaluation...")
        metrics = run_evaluation(
            model=cadp_model,
            val_loader=val_loader,
            dataset=dataset,
            save_dir=results_dir
        )
        
        # Safety metrics analysis
        print(f"\\n🛡️  Safety Metrics Analysis...")
        safety_metrics = evaluate_safety_metrics(cadp_model, val_loader)
        
        best_val_loss = min(trainer.val_losses)
        print(f"\\n" + "="*80)
        print(f"📊 CADP {stage.upper()} RESULTS")
        print("="*80)
        print(f"   • Best validation loss: {best_val_loss:.6f}")
        print(f"   • Collision rate: {safety_metrics['collision_rate']:.1%}")
        print(f"   • Smoothness score: {safety_metrics['smoothness_score']:.3f}")
        print(f"   • Baseline comparison: {config['baseline_success_rate']:.1%} → Target: 75%+")
        
        return {
            'stage': stage,
            'best_val_loss': best_val_loss,
            'safety_metrics': safety_metrics,
            'baseline_success_rate': config['baseline_success_rate']
        }


def train_cadp_model(model, train_loader, val_loader, config, save_dir):
    """
    Custom training function for CADP with physics-informed loss
    """
    import time
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = AdamW(model.parameters(), 
                      lr=config['lr'], 
                      weight_decay=config['weight_decay'])
    
    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=config['num_epochs'], 
                                  eta_min=config['lr_min'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    print(f"Starting CADP training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss_sum = 0.0
        physics_loss_sum = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch in train_loader:
            observations = batch['observation'].to(next(model.parameters()).device)
            actions = batch['action'].to(next(model.parameters()).device)
            
            optimizer.zero_grad()
            
            # Compute CADP losses
            loss_dict = model.compute_loss(actions, observations)
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            if config.get('clip_grad_norm'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_grad_norm'])
            
            optimizer.step()
            
            train_loss_sum += loss_dict['diffusion_loss'].item()
            if 'physics_loss' in loss_dict:
                physics_loss_sum += loss_dict['physics_loss'].item()
            num_batches += 1
        
        avg_train_loss = train_loss_sum / num_batches
        avg_physics_loss = physics_loss_sum / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        if epoch % config['validate_every'] == 0:
            model.eval()
            val_loss_sum = 0.0
            val_physics_loss_sum = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    observations = batch['observation'].to(next(model.parameters()).device)
                    actions = batch['action'].to(next(model.parameters()).device)
                    
                    loss_dict = model.compute_loss(actions, observations)
                    val_loss_sum += loss_dict['diffusion_loss'].item()
                    if 'physics_loss' in loss_dict:
                        val_physics_loss_sum += loss_dict['physics_loss'].item()
                    val_batches += 1
            
            avg_val_loss = val_loss_sum / val_batches
            avg_val_physics_loss = val_physics_loss_sum / val_batches
            val_losses.append(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'physics_loss': avg_physics_loss,
                    'config': config
                }, f"{save_dir}/best_model.pt")
                
                print(f"✓ Best model saved at epoch {epoch}")
            else:
                patience_counter += 1
                
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch:3d}/{config['num_epochs']:3d} | "
                  f"Train Loss: {avg_train_loss:.6f} | "
                  f"Val Loss: {avg_val_loss:.6f} | "
                  f"Physics: {avg_physics_loss:.6f} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
        scheduler.step()
    
    # Return trainer-like object
    class TrainerResult:
        def __init__(self, train_losses, val_losses):
            self.train_losses = train_losses
            self.val_losses = val_losses
    
    return TrainerResult(train_losses, val_losses)


def evaluate_safety_metrics(model, val_loader):
    """Evaluate CADP safety metrics"""
    model.eval()
    
    collision_rates = []
    smoothness_scores = []
    
    with torch.no_grad():
        for batch in val_loader:
            observations = batch['observation'].to(next(model.parameters()).device)
            actions = batch['action'].to(next(model.parameters()).device)
            safety_metrics = model.get_safety_metrics(actions, observations)
            collision_rates.append(safety_metrics['collision_rate'])
            smoothness_scores.append(safety_metrics['smoothness_score'])
    
    return {
        'collision_rate': np.mean(collision_rates),
        'smoothness_score': np.mean(smoothness_scores)
    }


def main():
    parser = argparse.ArgumentParser(description='CADP Training')
    parser.add_argument('--stage', choices=['stage_2', 'stage_4', 'both'], 
                       default='both', help='Which stage to train CADP on')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--max_demos', type=int, default=None, help='Max demos per task')
    parser.add_argument('--collision_weight', type=float, default=0.1, 
                       help='Weight for collision loss')
    parser.add_argument('--smoothness_weight', type=float, default=0.05,
                       help='Weight for smoothness loss')
    
    args = parser.parse_args()
    
    warnings.filterwarnings('ignore')
    
    trainer = CADPTrainer(args)
    
    results = []
    
    if args.stage in ['stage_2', 'both']:
        result = trainer.train_cadp_stage('stage_2')
        results.append(result)
        
    if args.stage in ['stage_4', 'both']:  
        result = trainer.train_cadp_stage('stage_4')
        results.append(result)
    
    # Final summary
    print("\\n" + "="*100)
    print("🏆 CADP TRAINING FINAL SUMMARY")
    print("="*100)
    
    for result in results:
        print(f"\\n📊 {result['stage'].upper()}:")
        print(f"   • Validation loss: {result['best_val_loss']:.6f}")
        print(f"   • Collision rate: {result['safety_metrics']['collision_rate']:.1%}")
        print(f"   • Smoothness score: {result['safety_metrics']['smoothness_score']:.3f}")
        print(f"   • Baseline: {result['baseline_success_rate']:.1%} → CADP Target: 75%+")
    
    print("\\n🎉 CADP training completed! Ready for safety testing.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())