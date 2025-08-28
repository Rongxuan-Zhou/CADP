#!/usr/bin/env python3
"""
Simple CBF Optimization Test
Based on ALGORITHM_COMPARISON_ANALYSIS.md Phase 1 recommendations
"""

import os
import sys
import torch
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.safety.cbf_verifier import create_franka_cbf_verifier
from src.safety.cbf_verifier_batch_optimized import create_batch_optimized_cbf_verifier

def run_optimization_test():
    print("🚀 CBF Optimization Test")
    print("=" * 50)
    
    # Create verifiers
    baseline_cbf = create_franka_cbf_verifier()
    optimized_cbf = create_batch_optimized_cbf_verifier(batch_size=16, use_gpu=True)
    
    print(f"✅ Baseline CBF created")
    print(f"✅ Optimized CBF created (GPU: {torch.cuda.is_available()})")
    
    # Test configurations
    trajectory_lengths = [10, 20, 30, 50]
    num_trials = 5
    
    results = {}
    
    for T in trajectory_lengths:
        print(f"\n📏 Testing T={T}")
        print("-" * 25)
        
        # Create test trajectories
        test_trajectories = []
        for i in range(10):
            traj = torch.randn(T, 7) * 0.3
            # Add violations
            traj[min(5, T-1), 0] = 3.2  # Joint limit violation
            if T >= 20:
                traj[T//2, 1] = -2.1
            test_trajectories.append(traj)
        
        # Test baseline
        baseline_times = []
        for trial in range(num_trials):
            start = time.time()
            result = baseline_cbf.verify_trajectory(test_trajectories[0])
            end = time.time()
            baseline_times.append((end - start) * 1000)
        
        baseline_avg = np.mean(baseline_times)
        
        # Test optimized (batch)
        optimized_times = []
        for trial in range(num_trials):
            start = time.time()
            batch_results = optimized_cbf.batch_verify_trajectories(test_trajectories)
            end = time.time()
            total_time = (end - start) * 1000
            avg_per_trajectory = total_time / len(test_trajectories)
            optimized_times.append(avg_per_trajectory)
        
        optimized_avg = np.mean(optimized_times)
        speedup = baseline_avg / optimized_avg if optimized_avg > 0 else 0
        
        results[T] = {
            'baseline_ms': baseline_avg,
            'optimized_ms': optimized_avg,
            'speedup': speedup,
            'target_met': optimized_avg < 50
        }
        
        status = "✅" if optimized_avg < 50 else "❌"
        print(f"   Baseline: {baseline_avg:.1f}ms")
        print(f"   Optimized: {optimized_avg:.1f}ms")
        print(f"   Speedup: {speedup:.1f}x")
        print(f"   Target (<50ms): {status}")
    
    # Summary
    print(f"\n📊 Summary Results:")
    print(f"   {'T':<4} {'Baseline':<10} {'Optimized':<10} {'Speedup':<8} {'Target'}")
    print(f"   {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*6}")
    
    total_speedup = 0
    targets_met = 0
    
    for T, data in results.items():
        status = "✅" if data['target_met'] else "❌"
        print(f"   {T:<4} {data['baseline_ms']:<10.1f} {data['optimized_ms']:<10.1f} {data['speedup']:<8.1f}x {status}")
        total_speedup += data['speedup']
        if data['target_met']:
            targets_met += 1
    
    avg_speedup = total_speedup / len(results)
    success_rate = (targets_met / len(results)) * 100
    
    print(f"\n🎯 Overall Assessment:")
    print(f"   • Average speedup: {avg_speedup:.1f}x")
    print(f"   • Targets met: {targets_met}/{len(results)} ({success_rate:.0f}%)")
    
    if avg_speedup >= 5 and success_rate >= 50:
        print(f"   • Status: 🎉 Optimization successful!")
    elif avg_speedup >= 2:
        print(f"   • Status: 📈 Partial improvement")
    else:
        print(f"   • Status: ⚠️ Limited improvement")
    
    return results

if __name__ == "__main__":
    try:
        results = run_optimization_test()
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()