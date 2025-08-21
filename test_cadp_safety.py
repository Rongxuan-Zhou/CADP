#!/usr/bin/env python3
"""
CADP Safety Testing Script
Author: CADP Project Team

Comprehensive safety evaluation of trained CADP models.
"""

import os
import sys
import argparse
import torch
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.multi_task_dataset import create_multi_task_dataloaders
from src.models.cadp_model import create_cadp_model
from src.evaluation.safety_evaluator import CADPSafetyEvaluator, create_safety_test_report


def load_cadp_model(checkpoint_path: str, device: torch.device):
    """Load trained CADP model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CADP checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model configuration from checkpoint
    config = checkpoint.get('config', {})
    
    # Create CADP model with correct architecture (from checkpoint analysis)
    model = create_cadp_model(
        obs_dim=23,  # Multi-task observation dimension
        action_dim=7,
        horizon=20,
        device=device,
        hidden_dims=[128, 256, 256],
        time_embed_dim=128,
        cond_dim=256,
        num_diffusion_steps=100,
        collision_weight=0.1,
        smoothness_weight=0.05,
        enable_safety=True
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded CADP model from: {checkpoint_path}")
    print(f"   • Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   • Validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"   • Physics loss: {checkpoint.get('physics_loss', 'N/A'):.6f}")
    
    return model


def run_safety_benchmark(model, device):
    """Run comprehensive safety benchmark tests"""
    
    print("\\n🛡️  CADP Safety Benchmark Testing")
    print("=" * 60)
    
    # Task configurations for testing
    task_configs = {
        'lift': 'data/robomimic_lowdim/ph/lift_ph_lowdim.hdf5',
        'can': 'data/robomimic_lowdim/ph/can_ph_lowdim.hdf5',
        'square': 'data/robomimic_lowdim/ph/square_ph_lowdim.hdf5'
    }
    
    # Create test dataloader
    print("📊 Preparing test data...")
    _, val_loader, dataset = create_multi_task_dataloaders(
        task_configs=task_configs,
        horizon=20,
        batch_size=8,  # Smaller batch for detailed analysis
        max_demos=50,  # Limited for faster testing
        train_ratio=0.8,
        curriculum_learning=False
    )
    
    # Initialize safety evaluator
    evaluator = CADPSafetyEvaluator(model, device)
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(val_loader)
    
    return results


def analyze_physics_loss_impact(model, val_loader, device):
    """Analyze the impact of physics-informed losses"""
    
    print("\\n🔬 Physics Loss Impact Analysis")
    print("-" * 40)
    
    model.eval()
    collision_losses = []
    smoothness_losses = []
    diffusion_losses = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 20:  # Limit analysis
                break
                
            observations = batch['observation'].to(device)
            actions = batch['action'].to(device)
            
            # Compute all loss components
            loss_dict = model.compute_loss(actions, observations)
            
            collision_losses.append(loss_dict['collision_loss'].item())
            smoothness_losses.append(loss_dict['smoothness_loss'].item())
            diffusion_losses.append(loss_dict['diffusion_loss'].item())
    
    # Statistical analysis
    print(f"📊 Loss Component Statistics:")
    print(f"   • Collision Loss: {np.mean(collision_losses):.4f} ± {np.std(collision_losses):.4f}")
    print(f"   • Smoothness Loss: {np.mean(smoothness_losses):.4f} ± {np.std(smoothness_losses):.4f}")
    print(f"   • Diffusion Loss: {np.mean(diffusion_losses):.4f} ± {np.std(diffusion_losses):.4f}")
    
    # Physics loss effectiveness
    physics_loss_total = np.mean(collision_losses) + np.mean(smoothness_losses)
    diffusion_loss_avg = np.mean(diffusion_losses)
    
    physics_ratio = physics_loss_total / diffusion_loss_avg * 100
    print(f"   • Physics/Diffusion Ratio: {physics_ratio:.1f}%")
    
    if physics_ratio > 5:
        print("   🎯 Physics constraints are actively influencing training")
    else:
        print("   ⚠️ Physics constraints may need stronger weighting")
    
    return {
        'collision_loss_mean': np.mean(collision_losses),
        'smoothness_loss_mean': np.mean(smoothness_losses),
        'diffusion_loss_mean': np.mean(diffusion_losses),
        'physics_ratio': physics_ratio
    }


def benchmark_inference_speed(model, device, num_samples: int = 100):
    """Benchmark CADP inference speed"""
    
    print("\\n⚡ Inference Speed Benchmark")
    print("-" * 30)
    
    model.eval()
    
    # Create dummy test input
    batch_size = 16
    obs_dim = 23
    observations = torch.randn(batch_size, obs_dim, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model.sample_actions(observations, num_samples=1)
    
    # Benchmark
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            start_time = time.time()
            predicted_actions = model.sample_actions(observations, num_samples=1)
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000 / batch_size)  # ms per sample
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    max_time = np.max(times)
    
    print(f"📊 Inference Performance:")
    print(f"   • Average: {avg_time:.2f}ms per trajectory")
    print(f"   • Std Dev: {std_time:.2f}ms")
    print(f"   • Maximum: {max_time:.2f}ms")
    print(f"   • Target: <50ms")
    
    if avg_time < 50:
        print("   ✅ Meets real-time performance requirement")
    else:
        print("   ⚠️ Exceeds real-time performance target")
    
    return {
        'avg_inference_time_ms': avg_time,
        'max_inference_time_ms': max_time,
        'std_inference_time_ms': std_time
    }


def compare_with_baseline(cadp_results: dict, baseline_success_rate: float = 0.70):
    """Compare CADP safety results with baseline"""
    
    print("\\n📈 CADP vs Baseline Comparison")
    print("=" * 40)
    
    cadp_success = cadp_results['static_obstacles']['avg_success_rate']
    cadp_collision = cadp_results['static_obstacles']['avg_collision_rate']
    cadp_inference = cadp_results['static_obstacles']['avg_inference_time_ms']
    
    print(f"📊 Performance Comparison:")
    print(f"   • Success Rate:")
    print(f"     - Baseline: {baseline_success_rate:.1%}")
    print(f"     - CADP: {cadp_success:.1%}")
    print(f"     - Change: {(cadp_success - baseline_success_rate) * 100:+.1f}%")
    
    print(f"   • Safety Metrics:")
    print(f"     - Collision Rate: {cadp_collision:.1%}")
    print(f"     - Inference Time: {cadp_inference:.1f}ms")
    
    # Overall assessment
    success_maintained = cadp_success >= baseline_success_rate * 0.95
    safety_achieved = cadp_collision < 0.05
    speed_maintained = cadp_inference < 50
    
    print(f"\\n🎯 CADP Assessment:")
    print(f"   • Success Rate Maintained: {'✅' if success_maintained else '❌'}")
    print(f"   • Safety Target Achieved: {'✅' if safety_achieved else '❌'}")
    print(f"   • Real-time Performance: {'✅' if speed_maintained else '❌'}")
    
    if success_maintained and safety_achieved and speed_maintained:
        print("\\n🏆 CADP successfully enhances safety while maintaining performance!")
    else:
        print("\\n🔧 CADP needs further optimization to meet all targets.")
    
    return {
        'success_maintained': success_maintained,
        'safety_achieved': safety_achieved,
        'speed_maintained': speed_maintained
    }


def main():
    parser = argparse.ArgumentParser(description='CADP Safety Testing')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to CADP checkpoint file')
    parser.add_argument('--output_dir', type=str, default='cadp_safety_results',
                       help='Output directory for results')
    parser.add_argument('--baseline_success', type=float, default=0.70,
                       help='Baseline success rate for comparison')
    
    args = parser.parse_args()
    
    warnings.filterwarnings('ignore')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 CADP Safety Testing Suite")
    print(f"   Device: {device}")
    print(f"   Checkpoint: {args.checkpoint}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CADP model
    model = load_cadp_model(args.checkpoint, device)
    
    # Create test data
    task_configs = {
        'lift': 'data/robomimic_lowdim/ph/lift_ph_lowdim.hdf5',
        'can': 'data/robomimic_lowdim/ph/can_ph_lowdim.hdf5',
        'square': 'data/robomimic_lowdim/ph/square_ph_lowdim.hdf5'
    }
    
    _, val_loader, _ = create_multi_task_dataloaders(
        task_configs=task_configs,
        horizon=20,
        batch_size=8,
        max_demos=50,
        train_ratio=0.8,
        curriculum_learning=False
    )
    
    # Run comprehensive safety evaluation
    print("\\n" + "="*60)
    safety_results = run_safety_benchmark(model, device)
    
    # Analyze physics loss impact
    physics_analysis = analyze_physics_loss_impact(model, val_loader, device)
    
    # Benchmark inference speed
    speed_results = benchmark_inference_speed(model, device)
    
    # Compare with baseline
    comparison_results = compare_with_baseline(safety_results, args.baseline_success)
    
    # Generate comprehensive report
    all_results = {
        'static_obstacles': safety_results['static_obstacles'],
        'dynamic_obstacles': safety_results['dynamic_obstacles'], 
        'narrow_corridor': safety_results['narrow_corridor']
    }
    
    report = create_safety_test_report(all_results, 
                                     os.path.join(args.output_dir, 'cadp_safety_report.md'))
    
    # Save detailed results
    import json
    detailed_results = {
        'safety_evaluation': safety_results,
        'physics_analysis': physics_analysis,
        'inference_benchmark': speed_results,
        'baseline_comparison': comparison_results,
        'summary': {
            'collision_rate': safety_results['static_obstacles']['avg_collision_rate'],
            'success_rate': safety_results['static_obstacles']['avg_success_rate'],
            'inference_time_ms': safety_results['static_obstacles']['avg_inference_time_ms'],
            'meets_safety_target': safety_results['static_obstacles']['avg_collision_rate'] < 0.05,
            'meets_speed_target': safety_results['static_obstacles']['avg_inference_time_ms'] < 50,
            'maintains_performance': safety_results['static_obstacles']['avg_success_rate'] >= args.baseline_success * 0.95
        }
    }
    
    with open(os.path.join(args.output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\\n📄 Results saved to: {args.output_dir}")
    print("\\n🎉 CADP safety testing completed!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())