#!/usr/bin/env python3
"""
Final CBF Performance Comparison - All three implementations
Original vs Optimized vs Advanced
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
from src.safety.cbf_verifier_advanced import create_advanced_cbf_verifier


def run_final_comparison():
    print("🚀 Final CBF Performance Comparison")
    print("Original vs Optimized vs Advanced")
    print("=" * 60)
    
    # Initialize all three implementations
    print("Initializing CBF verifiers...")
    cbf_original = create_franka_cbf_verifier()
    cbf_optimized = create_optimized_cbf_verifier()
    cbf_advanced = create_advanced_cbf_verifier()
    
    # Test configurations
    trajectory_lengths = [10, 20, 30, 50, 100]
    num_trials = 3
    
    print(f"\\n📊 Test Configuration:")
    print(f"   • Trajectory lengths: {trajectory_lengths}")
    print(f"   • Trials per length: {num_trials}")
    print(f"   • Target: <50ms verification time")
    
    results = {
        'original': {},
        'optimized': {},
        'advanced': {}
    }
    
    for T in trajectory_lengths:
        print(f"\\n📏 Testing T={T}")
        print("-" * 40)
        
        original_times = []
        optimized_times = []
        advanced_times = []
        
        for trial in range(num_trials):
            # Create consistent test trajectory
            torch.manual_seed(42 + trial)  # Consistent random seed
            trajectory = torch.randn(T, 7) * 0.3
            
            # Add controlled violations
            if T > 5:
                trajectory[T//4, 0] = 3.2  # Joint limit violation
            if T > 10:
                trajectory[T//2] = trajectory[T//2-1] + torch.randn(7) * 1.0  # Velocity violation
            if T > 20:
                trajectory[3*T//4, 2] = -3.1  # Another joint violation
            
            # Test original implementation (skip for very long trajectories to save time)
            if T <= 50:
                try:
                    start_time = time.time()
                    original_result = cbf_original.verify_trajectory(trajectory, dt=0.1)
                    original_time = (time.time() - start_time) * 1000
                    original_times.append(original_time)
                except Exception as e:
                    print(f"Original failed at T={T}: {e}")
                    original_times.append(float('inf'))
            else:
                original_times.append(float('inf'))  # Skip very long trajectories
            
            # Test optimized implementation
            try:
                start_time = time.time()
                optimized_result = cbf_optimized.verify_trajectory_optimized(trajectory, dt=0.1)
                optimized_time = (time.time() - start_time) * 1000
                optimized_times.append(optimized_time)
            except Exception as e:
                print(f"Optimized failed at T={T}: {e}")
                optimized_times.append(float('inf'))
            
            # Test advanced implementation
            try:
                start_time = time.time()
                advanced_result = cbf_advanced.verify_trajectory_advanced(trajectory, dt=0.1)
                advanced_time = (time.time() - start_time) * 1000
                advanced_times.append(advanced_time)
            except Exception as e:
                print(f"Advanced failed at T={T}: {e}")
                advanced_times.append(float('inf'))
        
        # Calculate statistics
        def calc_stats(times):
            valid_times = [t for t in times if t != float('inf')]
            if not valid_times:
                return {'mean': float('inf'), 'std': 0, 'min': float('inf'), 'max': float('inf')}
            return {
                'mean': np.mean(valid_times),
                'std': np.std(valid_times),
                'min': np.min(valid_times),
                'max': np.max(valid_times)
            }
        
        orig_stats = calc_stats(original_times)
        opt_stats = calc_stats(optimized_times)
        adv_stats = calc_stats(advanced_times)
        
        results['original'][T] = orig_stats
        results['optimized'][T] = opt_stats
        results['advanced'][T] = adv_stats
        
        # Print results for this length
        print(f"   Original:  {orig_stats['mean']:7.1f}ms ± {orig_stats['std']:5.1f}ms")
        print(f"   Optimized: {opt_stats['mean']:7.1f}ms ± {opt_stats['std']:5.1f}ms")
        print(f"   Advanced:  {adv_stats['mean']:7.1f}ms ± {adv_stats['std']:5.1f}ms")
        
        # Speedup calculations
        if orig_stats['mean'] != float('inf') and opt_stats['mean'] > 0:
            opt_speedup = orig_stats['mean'] / opt_stats['mean']
            print(f"   Opt Speedup: {opt_speedup:.1f}x")
        
        if orig_stats['mean'] != float('inf') and adv_stats['mean'] > 0:
            adv_speedup = orig_stats['mean'] / adv_stats['mean']
            print(f"   Adv Speedup: {adv_speedup:.1f}x")
        
        # Target achievement
        opt_target = "✅" if opt_stats['mean'] < 50 else "❌"
        adv_target = "✅" if adv_stats['mean'] < 50 else "❌"
        print(f"   Target <50ms: Opt {opt_target}, Adv {adv_target}")
    
    # Generate comprehensive report
    generate_final_report(results, trajectory_lengths)
    
    return results


def generate_final_report(results, trajectory_lengths):
    """Generate comprehensive comparison report"""
    print("\\n" + "=" * 80)
    print("📊 COMPREHENSIVE FINAL COMPARISON REPORT")
    print("=" * 80)
    
    # Summary table
    print("\\n📋 Performance Summary Table:")
    print("-" * 80)
    print(f"{'Length':<8} {'Original':<12} {'Optimized':<12} {'Advanced':<12} {'Opt Target':<11} {'Adv Target':<11}")
    print("-" * 80)
    
    opt_targets_met = 0
    adv_targets_met = 0
    total_lengths = len(trajectory_lengths)
    
    for T in trajectory_lengths:
        orig_time = results['original'][T]['mean']
        opt_time = results['optimized'][T]['mean']
        adv_time = results['advanced'][T]['mean']
        
        orig_str = f"{orig_time:.1f}" if orig_time != float('inf') else "SKIP"
        opt_str = f"{opt_time:.1f}" if opt_time != float('inf') else "FAIL"
        adv_str = f"{adv_time:.1f}" if adv_time != float('inf') else "FAIL"
        
        opt_target = "✅" if opt_time < 50 else "❌"
        adv_target = "✅" if adv_time < 50 else "❌"
        
        if opt_time < 50:
            opt_targets_met += 1
        if adv_time < 50:
            adv_targets_met += 1
        
        print(f"{T:<8} {orig_str:<12} {opt_str:<12} {adv_str:<12} {opt_target:<11} {adv_target:<11}")
    
    print("-" * 80)
    
    # Overall statistics
    print(f"\\n🎯 Overall Achievement:")
    print(f"   • Optimized targets met: {opt_targets_met}/{total_lengths} ({opt_targets_met/total_lengths*100:.0f}%)")
    print(f"   • Advanced targets met: {adv_targets_met}/{total_lengths} ({adv_targets_met/total_lengths*100:.0f}%)")
    
    # Find maximum trajectory length meeting target
    max_opt_target = max([T for T in trajectory_lengths if results['optimized'][T]['mean'] < 50], default=0)
    max_adv_target = max([T for T in trajectory_lengths if results['advanced'][T]['mean'] < 50], default=0)
    
    print(f"   • Max trajectory meeting target:")
    print(f"     - Optimized: T={max_opt_target}")
    print(f"     - Advanced:  T={max_adv_target}")
    
    # Performance improvement analysis
    print(f"\\n📈 Performance Improvement Analysis:")
    
    total_opt_speedup = 0
    total_adv_speedup = 0
    valid_comparisons = 0
    
    for T in trajectory_lengths:
        orig_time = results['original'][T]['mean']
        opt_time = results['optimized'][T]['mean']
        adv_time = results['advanced'][T]['mean']
        
        if orig_time != float('inf') and opt_time > 0 and adv_time > 0:
            opt_speedup = orig_time / opt_time
            adv_speedup = orig_time / adv_time
            total_opt_speedup += opt_speedup
            total_adv_speedup += adv_speedup
            valid_comparisons += 1
    
    if valid_comparisons > 0:
        avg_opt_speedup = total_opt_speedup / valid_comparisons
        avg_adv_speedup = total_adv_speedup / valid_comparisons
        
        print(f"   • Average speedup over original:")
        print(f"     - Optimized: {avg_opt_speedup:.1f}x")
        print(f"     - Advanced:  {avg_adv_speedup:.1f}x")
    
    # Scalability analysis
    print(f"\\n📏 Scalability Analysis:")
    
    # Check if advanced version handles long trajectories better
    long_trajectory_performance = []
    for T in [50, 100]:
        if T in results['advanced']:
            adv_time = results['advanced'][T]['mean']
            if adv_time != float('inf'):
                long_trajectory_performance.append((T, adv_time))
    
    if len(long_trajectory_performance) >= 2:
        # Simple linear regression
        lengths = [x[0] for x in long_trajectory_performance]
        times = [x[1] for x in long_trajectory_performance]
        
        if len(lengths) >= 2:
            slope = (times[1] - times[0]) / (lengths[1] - lengths[0])
            intercept = times[0] - slope * lengths[0]
            
            print(f"   • Advanced CBF scaling: ~{slope:.2f}ms per waypoint")
            print(f"   • Base overhead: ~{intercept:.1f}ms")
            
            # Predict performance for T=200
            pred_200 = slope * 200 + intercept
            print(f"   • Predicted T=200: {pred_200:.1f}ms ({'✅' if pred_200 < 50 else '❌'} target)")
    
    # Technology readiness assessment
    print(f"\\n🚀 Technology Readiness Assessment:")
    
    if adv_targets_met == total_lengths:
        print("   ✅ PRODUCTION READY: All trajectory lengths meet real-time requirements")
        print("   💡 Recommended: Deploy advanced CBF for all trajectory lengths")
    elif adv_targets_met >= total_lengths * 0.8:
        print("   🟡 MOSTLY READY: Most trajectory lengths meet requirements")
        print("   💡 Recommended: Deploy with trajectory length limits or segmentation")
    else:
        print("   ⚠️  OPTIMIZATION NEEDED: Further improvements required")
        print("   💡 Recommended: Continue optimization efforts or use hierarchical approach")
    
    # Final recommendations
    print(f"\\n💡 Final Recommendations:")
    
    if max_adv_target >= 50:
        print("   1. ✅ Deploy advanced CBF for production use")
        print("   2. 🎯 Configure trajectory segmentation for T>50 if needed")
        print("   3. 🔧 Monitor performance in real-world scenarios")
    else:
        print("   1. 🔧 Implement GPU acceleration for batch operations")
        print("   2. 📊 Consider neural network approximation for common cases")
        print("   3. 🎯 Use hierarchical verification with adaptive resolution")
    
    print("   4. 🧪 Validate with actual CADP model integration")
    print("   5. 📈 Benchmark on target hardware platform")


if __name__ == "__main__":
    try:
        results = run_final_comparison()
        print("\\n🎉 Final CBF comparison completed successfully!")
    except Exception as e:
        print(f"\\n❌ Final comparison failed: {e}")
        import traceback
        traceback.print_exc()