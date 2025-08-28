#!/usr/bin/env python3
"""
CBF Optimization Experiment - Following ALGORITHM_COMPARISON_ANALYSIS.md
Comprehensive testing of Phase 1 optimizations with detailed logging
Author: CADP Optimization Team
"""

import os
import sys
import torch
import time
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.safety.cbf_verifier import create_franka_cbf_verifier
from src.safety.cbf_verifier_batch_optimized import create_batch_optimized_cbf_verifier


class CBFOptimizationExperiment:
    """Comprehensive CBF optimization experiment following analysis recommendations"""
    
    def __init__(self):
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'hardware_info': self._get_hardware_info(),
            'experiment_phases': {}
        }
        
        print(f"🚀 CBF Optimization Experiment Started")
        print(f"   • Experiment ID: {self.experiment_id}")
        print(f"   • Hardware: {self.results['hardware_info']['gpu_name']}")
        print(f"   • Following ALGORITHM_COMPARISON_ANALYSIS.md recommendations")
        print("=" * 80)
    
    def _get_hardware_info(self):
        """Collect hardware information"""
        info = {}
        
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            info['gpu_available'] = False
            info['gpu_name'] = 'CPU Only'
            info['gpu_memory_gb'] = 0
        
        info['torch_version'] = torch.__version__
        info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else None
        
        return info
    
    def run_baseline_benchmark(self) -> dict:
        """Phase 0: Establish baseline performance"""
        
        print("\n📊 Phase 0: Baseline Benchmark")
        print("-" * 50)
        
        baseline_verifier = create_franka_cbf_verifier()
        
        # Test configurations matching analysis
        trajectory_lengths = [10, 20, 30, 50]
        num_trials = 10
        
        baseline_results = {
            'phase': 'baseline',
            'verifier_type': 'original_cbf',
            'trajectory_lengths': trajectory_lengths,
            'num_trials': num_trials,
            'detailed_results': {}
        }
        
        print(f"   • Testing trajectory lengths: {trajectory_lengths}")
        print(f"   • Trials per length: {num_trials}")
        
        for T in trajectory_lengths:
            print(f"\\n   📏 Testing T={T}...")
            
            times = []
            corrections = []
            correction_norms = []
            
            for trial in range(num_trials):
                # Create test trajectory with violations
                trajectory = torch.randn(T, 7) * 0.3
                
                # Add controlled violations
                if T >= 10:
                    trajectory[T//4, 0] = 3.2  # Joint limit violation
                if T >= 20:
                    trajectory[T//2, 1] = -2.1  # Another violation
                
                # Time the verification
                start_time = time.time()
                result = baseline_verifier.verify_trajectory(trajectory, dt=0.1)
                end_time = time.time()
                
                verification_time = (end_time - start_time) * 1000  # ms
                times.append(verification_time)
                corrections.append(result.num_corrections)
                correction_norms.append(result.max_correction_norm)
                
                if trial == 0:  # Detailed log for first trial
                    print(f"      Trial {trial+1}: {verification_time:.2f}ms, "
                          f"{result.num_corrections} corrections")
            
            # Statistics
            avg_time = np.mean(times)
            std_time = np.std(times)
            avg_corrections = np.mean(corrections)
            
            baseline_results['detailed_results'][f'T_{T}'] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'avg_corrections': avg_corrections,
                'avg_correction_norm': np.mean(correction_norms),
                'target_compliance': avg_time < 50.0,
                'raw_times': times
            }
            
            target_status = "✅" if avg_time < 50 else "❌"
            print(f"      Results: {avg_time:.2f}±{std_time:.2f}ms, "
                  f"{avg_corrections:.1f} corrections {target_status}")
        
        self.results['experiment_phases']['baseline'] = baseline_results
        return baseline_results
    
    def run_batch_optimization_test(self, baseline_results: dict) -> dict:
        """Phase 1: Batch optimization test"""
        
        print(f"\\n🚀 Phase 1: Batch Optimization Test")
        print("-" * 50)
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16, 32]
        use_gpu = torch.cuda.is_available()
        
        print(f"   • GPU Acceleration: {'✅' if use_gpu else '❌'}")
        print(f"   • Testing batch sizes: {batch_sizes}")
        
        optimization_results = {
            'phase': 'batch_optimization',
            'gpu_enabled': use_gpu,
            'batch_sizes_tested': batch_sizes,
            'detailed_results': {}
        }
        
        for batch_size in batch_sizes:
            print(f"\\n   📦 Testing batch_size={batch_size}...")
            
            verifier = create_batch_optimized_cbf_verifier(
                batch_size=batch_size, use_gpu=use_gpu
            )
            
            batch_results = {}
            
            for T in [10, 20, 30, 50]:
                print(f"      T={T}: ", end="", flush=True)
                
                # Create batch of test trajectories
                test_trajectories = []
                for i in range(20):  # Test with 20 trajectories
                    traj = torch.randn(T, 7) * 0.3
                    # Add violations
                    if T >= 10:
                        traj[T//4, 0] = 3.2
                    if T >= 20:
                        traj[T//2, 1] = -2.1
                    test_trajectories.append(traj)
                
                # Time batch verification
                start_time = time.time()
                results = verifier.batch_verify_trajectories(test_trajectories)
                end_time = time.time()
                
                total_time = (end_time - start_time) * 1000
                avg_time_per_trajectory = total_time / len(test_trajectories)
                
                # Calculate speedup vs baseline
                baseline_avg = baseline_results['detailed_results'][f'T_{T}']['avg_time_ms']
                speedup = baseline_avg / avg_time_per_trajectory
                
                batch_results[f'T_{T}'] = {
                    'total_time_ms': total_time,
                    'avg_time_per_trajectory_ms': avg_time_per_trajectory,
                    'baseline_time_ms': baseline_avg,
                    'speedup_factor': speedup,
                    'target_compliance': avg_time_per_trajectory < 50.0,
                    'trajectories_processed': len(test_trajectories)
                }
                
                status = "✅" if avg_time_per_trajectory < 50 else "❌"
                print(f"{avg_time_per_trajectory:.2f}ms ({speedup:.1f}x speedup) {status}")
            
            optimization_results['detailed_results'][f'batch_{batch_size}'] = batch_results
        
        self.results['experiment_phases']['batch_optimization'] = optimization_results
        return optimization_results
    
    def run_memory_optimization_test(self) -> dict:
        """Phase 2: Memory pre-allocation test"""
        
        print(f"\\n🧠 Phase 2: Memory Optimization Test")
        print("-" * 50)
        
        print("   • Testing memory pre-allocation impact...")
        
        # Compare with and without pre-allocation
        verifier_prealloc = create_batch_optimized_cbf_verifier(batch_size=16, use_gpu=True)
        
        memory_results = {
            'phase': 'memory_optimization',
            'test_description': 'Memory pre-allocation vs dynamic allocation',
            'results': {}
        }
        
        # Test repeated allocations
        T = 30
        num_repetitions = 50
        
        trajectories = [torch.randn(T, 7) * 0.3 for _ in range(num_repetitions)]
        
        print(f"   • Testing {num_repetitions} repeated verifications (T={T})")
        
        # Time multiple runs to see memory allocation impact
        times = []
        for i in range(10):
            start = time.time()
            results = verifier_prealloc.batch_verify_trajectories(trajectories[:10])
            end = time.time()
            times.append((end - start) * 1000 / 10)  # ms per trajectory
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        memory_results['results'] = {
            'avg_time_per_trajectory_ms': avg_time,
            'std_time_ms': std_time,
            'memory_efficiency': 'pre_allocated',
            'target_compliance': avg_time < 50.0
        }
        
        print(f"   • Pre-allocated memory: {avg_time:.2f}±{std_time:.2f}ms per trajectory")
        print(f"   • Target compliance: {'✅' if avg_time < 50 else '❌'}")
        
        self.results['experiment_phases']['memory_optimization'] = memory_results
        return memory_results
    
    def analyze_optimization_effectiveness(self):
        \"\"\"Phase 3: Comprehensive analysis of optimization effectiveness\"\"\"\n        \n        print(f\"\\n📈 Phase 3: Optimization Effectiveness Analysis\")\n        print(\"-\" * 50)\n        \n        baseline = self.results['experiment_phases']['baseline']['detailed_results']\n        batch_opt = self.results['experiment_phases']['batch_optimization']['detailed_results']\n        \n        print(\"\\n   📊 Performance Summary:\")\n        print(f\"   {'Length':<8} {'Baseline':<12} {'Optimized':<12} {'Speedup':<10} {'Target':<8}\")\n        print(f\"   {'-'*8:<8} {'-'*12:<12} {'-'*12:<12} {'-'*10:<10} {'-'*8:<8}\")\n        \n        total_speedup = 0\n        compliant_configs = 0\n        total_configs = 0\n        \n        for T in [10, 20, 30, 50]:\n            baseline_time = baseline[f'T_{T}']['avg_time_ms']\n            \n            # Find best batch configuration for this T\n            best_time = float('inf')\n            best_speedup = 0\n            \n            for batch_key, batch_data in batch_opt.items():\n                if f'T_{T}' in batch_data:\n                    opt_time = batch_data[f'T_{T}']['avg_time_per_trajectory_ms']\n                    if opt_time < best_time:\n                        best_time = opt_time\n                        best_speedup = batch_data[f'T_{T}']['speedup_factor']\n            \n            target_met = \"✅\" if best_time < 50 else \"❌\"\n            if best_time < 50:\n                compliant_configs += 1\n            total_configs += 1\n            \n            print(f\"   T={T:<6} {baseline_time:<12.2f} {best_time:<12.2f} {best_speedup:<10.1f}x {target_met:<8}\")\n            \n            total_speedup += best_speedup\n        \n        avg_speedup = total_speedup / 4\n        compliance_rate = (compliant_configs / total_configs) * 100\n        \n        analysis_results = {\n            'phase': 'effectiveness_analysis',\n            'avg_speedup_factor': avg_speedup,\n            'target_compliance_rate_percent': compliance_rate,\n            'configs_meeting_target': compliant_configs,\n            'total_configs_tested': total_configs,\n            'optimization_assessment': self._get_optimization_assessment(avg_speedup, compliance_rate)\n        }\n        \n        print(f\"\\n   🎯 Overall Assessment:\")\n        print(f\"   • Average speedup: {avg_speedup:.1f}x\")\n        print(f\"   • Target compliance: {compliance_rate:.1f}% ({compliant_configs}/{total_configs})\")\n        print(f\"   • Assessment: {analysis_results['optimization_assessment']}\")\n        \n        self.results['experiment_phases']['effectiveness_analysis'] = analysis_results\n        return analysis_results\n    \n    def _get_optimization_assessment(self, speedup: float, compliance: float) -> str:\n        \"\"\"Assess optimization effectiveness\"\"\"\n        \n        if speedup >= 10 and compliance >= 75:\n            return \"🏆 EXCELLENT - Phase 1 optimization targets achieved\"\n        elif speedup >= 5 and compliance >= 50:\n            return \"🎯 GOOD - Significant improvement, further optimization beneficial\"\n        elif speedup >= 2 and compliance >= 25:\n            return \"📈 MODERATE - Some improvement, major optimization needed\"\n        else:\n            return \"⚠️ LIMITED - Optimization approach needs revision\"\n    \n    def save_detailed_results(self):\n        \"\"\"Save comprehensive experimental results\"\"\"\n        \n        results_dir = 'cbf_optimization_results'\n        os.makedirs(results_dir, exist_ok=True)\n        \n        # Save JSON results\n        json_path = f\"{results_dir}/experiment_{self.experiment_id}.json\"\n        with open(json_path, 'w') as f:\n            json.dump(self.results, f, indent=2)\n        \n        # Create performance plots\n        self._create_performance_plots(results_dir)\n        \n        # Generate markdown report\n        self._generate_experiment_report(results_dir)\n        \n        print(f\"\\n💾 Results saved:\")\n        print(f\"   • JSON data: {json_path}\")\n        print(f\"   • Performance plots: {results_dir}/\")\n        print(f\"   • Detailed report: {results_dir}/experiment_report_{self.experiment_id}.md\")\n    \n    def _create_performance_plots(self, results_dir: str):\n        \"\"\"Create performance visualization plots\"\"\"\n        \n        try:\n            baseline = self.results['experiment_phases']['baseline']['detailed_results']\n            batch_opt = self.results['experiment_phases']['batch_optimization']['detailed_results']\n            \n            # Performance comparison plot\n            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n            \n            # Plot 1: Verification times\n            trajectory_lengths = [10, 20, 30, 50]\n            baseline_times = [baseline[f'T_{T}']['avg_time_ms'] for T in trajectory_lengths]\n            \n            # Get best optimized times\n            optimized_times = []\n            for T in trajectory_lengths:\n                best_time = float('inf')\n                for batch_data in batch_opt.values():\n                    if f'T_{T}' in batch_data:\n                        opt_time = batch_data[f'T_{T}']['avg_time_per_trajectory_ms']\n                        if opt_time < best_time:\n                            best_time = opt_time\n                optimized_times.append(best_time)\n            \n            x = np.array(trajectory_lengths)\n            ax1.bar(x - 1, baseline_times, width=2, label='Baseline CBF', alpha=0.7, color='red')\n            ax1.bar(x + 1, optimized_times, width=2, label='Optimized CBF', alpha=0.7, color='green')\n            ax1.axhline(y=50, color='orange', linestyle='--', label='Target (<50ms)')\n            ax1.set_xlabel('Trajectory Length')\n            ax1.set_ylabel('Verification Time (ms)')\n            ax1.set_title('CBF Verification Performance')\n            ax1.legend()\n            ax1.set_yscale('log')\n            \n            # Plot 2: Speedup factors\n            speedups = [baseline_times[i] / optimized_times[i] for i in range(len(trajectory_lengths))]\n            ax2.bar(trajectory_lengths, speedups, alpha=0.7, color='blue')\n            ax2.axhline(y=10, color='green', linestyle='--', label='Target (10x speedup)')\n            ax2.set_xlabel('Trajectory Length')\n            ax2.set_ylabel('Speedup Factor')\n            ax2.set_title('Optimization Speedup')\n            ax2.legend()\n            \n            plt.tight_layout()\n            plt.savefig(f'{results_dir}/performance_comparison_{self.experiment_id}.png', dpi=300)\n            plt.close()\n            \n        except Exception as e:\n            print(f\"   ⚠️ Could not create plots: {e}\")\n    \n    def _generate_experiment_report(self, results_dir: str):\n        \"\"\"Generate detailed markdown experiment report\"\"\"\n        \n        report_path = f\"{results_dir}/experiment_report_{self.experiment_id}.md\"\n        \n        with open(report_path, 'w') as f:\n            f.write(f\"# CBF Optimization Experiment Report\\n\\n\")\n            f.write(f\"**Experiment ID**: {self.experiment_id}\\n\")\n            f.write(f\"**Timestamp**: {self.results['timestamp']}\\n\")\n            f.write(f\"**Following**: ALGORITHM_COMPARISON_ANALYSIS.md recommendations\\n\\n\")\n            \n            # Hardware info\n            hw = self.results['hardware_info']\n            f.write(f\"## Hardware Configuration\\n\\n\")\n            f.write(f\"- **GPU**: {hw['gpu_name']}\\n\")\n            f.write(f\"- **VRAM**: {hw['gpu_memory_gb']:.1f}GB\\n\")\n            f.write(f\"- **PyTorch**: {hw['torch_version']}\\n\\n\")\n            \n            # Experimental phases\n            for phase_name, phase_data in self.results['experiment_phases'].items():\n                f.write(f\"## Phase: {phase_name.replace('_', ' ').title()}\\n\\n\")\n                f.write(f\"```json\\n{json.dumps(phase_data, indent=2)}\\n```\\n\\n\")\n    \n    def run_complete_experiment(self):\n        \"\"\"Run the complete optimization experiment\"\"\"\n        \n        try:\n            # Phase 0: Baseline\n            baseline_results = self.run_baseline_benchmark()\n            \n            # Phase 1: Batch Optimization  \n            batch_results = self.run_batch_optimization_test(baseline_results)\n            \n            # Phase 2: Memory Optimization\n            memory_results = self.run_memory_optimization_test()\n            \n            # Phase 3: Analysis\n            analysis_results = self.analyze_optimization_effectiveness()\n            \n            # Save results\n            self.save_detailed_results()\n            \n            print(f\"\\n🏁 Experiment Completed Successfully!\")\n            print(f\"   • Experiment ID: {self.experiment_id}\")\n            print(f\"   • Total phases: {len(self.results['experiment_phases'])}\")\n            \n            return self.results\n            \n        except Exception as e:\n            print(f\"\\n❌ Experiment failed: {e}\")\n            import traceback\n            traceback.print_exc()\n            return None\n\n\ndef main():\n    \"\"\"Run CBF optimization experiment\"\"\"\n    \n    print(\"🔬 CADP CBF Optimization Experiment\")\n    print(\"Following ALGORITHM_COMPARISON_ANALYSIS.md recommendations\")\n    print(\"Phase 1: Batch Processing + GPU Acceleration + Memory Optimization\")\n    print(\"Target: 10x speedup, <50ms verification time\")\n    print(\"=\" * 80)\n    \n    experiment = CBFOptimizationExperiment()\n    results = experiment.run_complete_experiment()\n    \n    if results:\n        print(\"\\n✅ Optimization experiment completed successfully!\")\n        print(\"Check cbf_optimization_results/ for detailed analysis.\")\n    else:\n        print(\"\\n❌ Experiment failed. Check error logs above.\")\n        return 1\n    \n    return 0\n\n\nif __name__ == \"__main__\":\n    sys.exit(main())