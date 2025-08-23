#!/usr/bin/env python3
"""
Final CBF Performance Test - Comparison between original and optimized
Author: CADP Project Team
"""

import os
import sys
import torch
import time
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.safety.cbf_verifier import create_franka_cbf_verifier
from src.safety.cbf_verifier_optimized import create_optimized_cbf_verifier


def run_performance_test():
    print("🚀 Final CBF Performance Comparison Test")
    print("=" * 60)
    
    # Initialize both versions
    cbf_original = create_franka_cbf_verifier()
    cbf_optimized = create_optimized_cbf_verifier()
    
    # Test configurations
    trajectory_lengths = [10, 20, 30, 50]
    num_trials = 5
    
    print(f"📊 Test Configuration:")
    print(f"   • Trajectory lengths: {trajectory_lengths}")
    print(f"   • Trials per length: {num_trials}")
    print(f"   • Target: <50ms verification time")
    
    results = {}
    
    for T in trajectory_lengths:
        print(f"\\n📏 Testing T={T}")
        print("-" * 30)
        
        original_times = []
        optimized_times = []
        
        for trial in range(num_trials):
            # Create test trajectory with violations
            trajectory = torch.randn(T, 7) * 0.3
            # Add some violations
            if T > 5:
                trajectory[T//4, 0] = 3.2  # Joint limit violation
            if T > 10:
                trajectory[T//2] = trajectory[T//2-1] + torch.randn(7) * 1.0  # Velocity violation
            
            # Test original version
            start_time = time.time()
            try:
                original_result = cbf_original.verify_trajectory(trajectory, dt=0.1)
                original_time = (time.time() - start_time) * 1000
                original_times.append(original_time)
            except Exception as e:
                print(f"⚠️  Original failed: {e}")
                original_times.append(float('inf'))
            
            # Test optimized version
            start_time = time.time()
            try:
                optimized_result = cbf_optimized.verify_trajectory_optimized(trajectory, dt=0.1)
                optimized_time = (time.time() - start_time) * 1000
                optimized_times.append(optimized_time)
            except Exception as e:
                print(f"⚠️  Optimized failed: {e}")
                optimized_times.append(float('inf'))
        
        # Calculate statistics
        orig_mean = np.mean([t for t in original_times if t != float('inf')])
        orig_std = np.std([t for t in original_times if t != float('inf')])
        opt_mean = np.mean([t for t in optimized_times if t != float('inf')])
        opt_std = np.std([t for t in optimized_times if t != float('inf')])
        
        speedup = orig_mean / opt_mean if opt_mean > 0 else float('inf')
        target_met = opt_mean < 50.0
        
        results[T] = {
            'original': {'mean': orig_mean, 'std': orig_std},
            'optimized': {'mean': opt_mean, 'std': opt_std},
            'speedup': speedup,
            'target_met': target_met
        }
        
        print(f"   Original:  {orig_mean:6.1f}ms ± {orig_std:5.1f}ms")
        print(f"   Optimized: {opt_mean:6.1f}ms ± {opt_std:5.1f}ms")
        print(f"   Speedup:   {speedup:6.1f}x")
        print(f"   Target:    {'✅ MET' if target_met else '❌ MISSED'}")
    
    # Summary report
    print("\\n" + "=" * 60)
    print("📊 FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    print(f"\\n{'Length':<8} {'Original':<12} {'Optimized':<12} {'Speedup':<10} {'Target':<8}")
    print("-" * 60)
    
    targets_met = 0
    total_speedup = 0
    
    for T in trajectory_lengths:
        result = results[T]
        orig_time = result['original']['mean']
        opt_time = result['optimized']['mean']
        speedup = result['speedup']
        target_status = "✅" if result['target_met'] else "❌"
        
        if result['target_met']:
            targets_met += 1
        total_speedup += speedup
        
        print(f"{T:<8} {orig_time:<12.1f} {opt_time:<12.1f} {speedup:<10.1f} {target_status:<8}")
    
    print("-" * 60)
    
    avg_speedup = total_speedup / len(trajectory_lengths)
    success_rate = targets_met / len(trajectory_lengths) * 100
    
    print(f"\\n🎯 Overall Results:")
    print(f"   • Average speedup: {avg_speedup:.1f}x")
    print(f"   • Targets met: {targets_met}/{len(trajectory_lengths)} ({success_rate:.0f}%)")
    print(f"   • Max trajectory meeting target: T={max([T for T in trajectory_lengths if results[T]['target_met']], default=0)}")
    
    # Performance analysis
    if success_rate == 100:
        print(f"\\n🎉 OPTIMIZATION SUCCESS!")
        print(f"   ✅ All trajectory lengths meet <50ms target")
        print(f"   ✅ Ready for real-time CADP integration")
    elif success_rate >= 75:
        print(f"\\n🟡 PARTIAL SUCCESS")
        print(f"   ⚠️  Most trajectories meet target")
        print(f"   💡 Consider further optimization for longer trajectories")
    else:
        print(f"\\n❌ OPTIMIZATION INCOMPLETE")
        print(f"   ⚠️  Significant performance gaps remain")
        print(f"   💡 Major algorithmic changes needed")
    
    return results


if __name__ == "__main__":
    try:
        results = run_performance_test()
        print("\\n✅ Performance test completed successfully!")
    except Exception as e:
        print(f"\\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()