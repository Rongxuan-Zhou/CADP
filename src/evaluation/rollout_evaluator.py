#!/usr/bin/env python3
"""
RoboMimic Environment Rollout Evaluator
Author: CADP Project Team

This module implements rollout evaluation in RoboMimic environments
to calculate task success rates for diffusion policy models.
"""

import os
import sys
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add robomimic to path
try:
    import robomimic
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.env_utils as EnvUtils
    from robomimic.envs.env_base import EnvBase
    print("✅ RoboMimic imports successful")
except ImportError as e:
    print(f"⚠️ RoboMimic not available: {e}")
    print("Install with: pip install robomimic")


class RolloutEvaluator:
    """Evaluates diffusion policy models through environment rollout"""
    
    def __init__(self, model, dataset, device='cuda', verbose=True):
        """
        Args:
            model: Trained diffusion policy model
            dataset: RoboMimicLowDimDataset instance
            device: Device for inference
            verbose: Print detailed results
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.verbose = verbose
        
        # Initialize environment
        self.env = None
        self.obs_keys = None
        
    def setup_environment(self, env_meta: Dict) -> bool:
        """Setup RoboMimic environment from dataset metadata"""
        try:
            # Extract environment info from dataset
            env_name = env_meta.get("env_name", "Lift")
            env_type = env_meta.get("type", "single_arm_opposed_thumb")
            
            if self.verbose:
                print(f"Setting up environment: {env_name} ({env_type})")
            
            # Create environment
            env_config = {
                "env_name": env_name,
                "type": env_type,
                "render": False,  # No rendering for batch evaluation
                "render_offscreen": False,
                "use_image_obs": False,  # We're using low-dim observations
                "reward_shaping": False,
            }
            
            self.env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=False,
                use_image_obs=False
            )
            
            # Get observation keys
            self.obs_keys = list(env_meta.get("env_kwargs", {}).get("observation_keys", 
                                                                  ["robot0_eef_pos", "robot0_eef_quat", 
                                                                   "robot0_gripper_qpos", "object"]))
            
            if self.verbose:
                print(f"Environment setup successful")
                print(f"Observation keys: {self.obs_keys}")
                
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def extract_observation(self, obs_dict: Dict) -> np.ndarray:
        """Extract and concatenate observation features"""
        obs_list = []
        
        for key in self.obs_keys:
            if key in obs_dict:
                obs_data = obs_dict[key]
                if isinstance(obs_data, np.ndarray):
                    obs_list.append(obs_data.flatten())
                else:
                    obs_list.append(np.array([obs_data]).flatten())
        
        if obs_list:
            return np.concatenate(obs_list).astype(np.float32)
        else:
            # Fallback: use all available observations
            all_obs = []
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray) and value.dtype in [np.float32, np.float64]:
                    all_obs.append(value.flatten())
            return np.concatenate(all_obs).astype(np.float32) if all_obs else np.zeros(19)
    
    def predict_actions(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action sequence using diffusion policy"""
        self.model.eval()
        
        with torch.no_grad():
            # Normalize observation
            obs_normalized = self.dataset.normalize_obs(obs)
            obs_tensor = torch.FloatTensor(obs_normalized).unsqueeze(0).to(self.device)
            
            # Generate action sequence
            if deterministic:
                # Use DDIM sampling for faster deterministic inference
                actions = self.model.conditional_sample(
                    cond=obs_tensor, 
                    batch_size=1,
                    deterministic=True
                )
            else:
                # Full DDPM sampling
                actions = self.model.conditional_sample(
                    cond=obs_tensor, 
                    batch_size=1,
                    deterministic=False
                )
            
            # Denormalize actions
            actions = actions.cpu().numpy()[0]  # Remove batch dimension
            actions = self.dataset.denormalize_actions(actions)
            
        return actions
    
    def run_rollout(self, max_steps: int = 200, deterministic: bool = True) -> Dict:
        """Run a single rollout episode"""
        if self.env is None:
            raise RuntimeError("Environment not setup. Call setup_environment first.")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        step = 0
        success = False
        trajectory_length = 0
        
        # Track trajectory
        observations = []
        actions = []
        rewards = []
        
        action_queue = []  # Queue for action sequence
        
        while not done and step < max_steps:
            # Extract observation vector
            obs_vector = self.extract_observation(obs)
            observations.append(obs_vector)
            
            # Generate new action sequence if queue is empty
            if len(action_queue) == 0:
                try:
                    predicted_actions = self.predict_actions(obs_vector, deterministic=deterministic)
                    action_queue = list(predicted_actions)  # Convert to list for queue
                except Exception as e:
                    print(f"Action prediction failed: {e}")
                    break
            
            # Execute next action from queue
            if action_queue:
                action = action_queue.pop(0)
                actions.append(action)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                rewards.append(reward)
                step += 1
                trajectory_length = step
                
                # Check for success
                if info.get("is_success", False):
                    success = True
                    break
        
        return {
            'success': success,
            'trajectory_length': trajectory_length,
            'total_reward': sum(rewards),
            'observations': observations,
            'actions': actions,
            'rewards': rewards
        }
    
    def evaluate_success_rate(self, num_episodes: int = 50, max_steps: int = 200, 
                            deterministic: bool = True) -> Dict:
        """Evaluate model success rate over multiple episodes"""
        if self.env is None:
            print("❌ Environment not setup")
            return {'success_rate': 0.0, 'error': 'Environment not available'}
        
        print(f"🎯 Running rollout evaluation ({num_episodes} episodes)")
        
        results = []
        successes = 0
        
        for episode in tqdm(range(num_episodes), desc="Rollout Episodes"):
            try:
                result = self.run_rollout(max_steps=max_steps, deterministic=deterministic)
                results.append(result)
                
                if result['success']:
                    successes += 1
                
                if self.verbose and episode < 5:  # Show first few episodes
                    status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                    print(f"Episode {episode+1}: {status} (steps: {result['trajectory_length']}, reward: {result['total_reward']:.3f})")
                    
            except Exception as e:
                print(f"Episode {episode+1} failed: {e}")
                results.append({
                    'success': False,
                    'trajectory_length': 0,
                    'total_reward': 0.0,
                    'error': str(e)
                })
        
        # Calculate statistics
        success_rate = successes / num_episodes
        avg_trajectory_length = np.mean([r['trajectory_length'] for r in results])
        avg_reward = np.mean([r['total_reward'] for r in results])
        
        evaluation_results = {
            'success_rate': success_rate,
            'success_count': successes,
            'total_episodes': num_episodes,
            'avg_trajectory_length': avg_trajectory_length,
            'avg_reward': avg_reward,
            'results': results
        }
        
        if self.verbose:
            print(f"\n📊 ROLLOUT EVALUATION RESULTS:")
            print(f"  • Success Rate: {success_rate:.1%} ({successes}/{num_episodes})")
            print(f"  • Average Trajectory Length: {avg_trajectory_length:.1f} steps")
            print(f"  • Average Reward: {avg_reward:.3f}")
        
        return evaluation_results


def evaluate_model_rollout(model, dataset, hdf5_path: str, num_episodes: int = 50) -> Dict:
    """
    Convenience function to evaluate a trained model with rollout
    
    Args:
        model: Trained diffusion policy model
        dataset: RoboMimicLowDimDataset instance  
        hdf5_path: Path to original dataset file for environment metadata
        num_episodes: Number of rollout episodes
        
    Returns:
        Dictionary with evaluation results
    """
    
    # Load environment metadata from dataset
    try:
        with h5py.File(hdf5_path, 'r') as f:
            env_meta = {}
            if 'data' in f:
                # Extract metadata
                if 'env_args' in f['data'].attrs:
                    env_meta = f['data'].attrs['env_args']
                else:
                    # Fallback metadata for standard RoboMimic datasets
                    env_meta = {
                        "env_name": "Lift",
                        "type": "single_arm_opposed_thumb",
                        "env_kwargs": {
                            "observation_keys": ["robot0_eef_pos", "robot0_eef_quat", 
                                               "robot0_gripper_qpos", "object"]
                        }
                    }
            
    except Exception as e:
        print(f"⚠️ Could not load environment metadata: {e}")
        print("Using default Lift environment configuration")
        env_meta = {
            "env_name": "Lift", 
            "type": "single_arm_opposed_thumb",
            "env_kwargs": {
                "observation_keys": ["robot0_eef_pos", "robot0_eef_quat", 
                                   "robot0_gripper_qpos", "object"]
            }
        }
    
    # Create evaluator
    evaluator = RolloutEvaluator(model, dataset)
    
    # Setup environment
    if not evaluator.setup_environment(env_meta):
        return {
            'success_rate': 0.0,
            'error': 'Failed to setup environment. RoboMimic may not be properly installed.',
            'fallback_used': True
        }
    
    # Run evaluation
    results = evaluator.evaluate_success_rate(num_episodes=num_episodes)
    
    return results


def create_rollout_analysis(results: Dict, save_path: str = "rollout_analysis.png"):
    """Create analysis plots for rollout results"""
    
    if 'error' in results:
        print(f"Cannot create analysis plots: {results['error']}")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Success Rate
    success_rate = results['success_rate'] 
    axes[0, 0].bar(['Success Rate'], [success_rate * 100], color='green' if success_rate >= 0.6 else 'orange')
    axes[0, 0].axhline(y=60, color='red', linestyle='--', label='Target (60%)')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title(f'Task Success Rate: {success_rate:.1%}')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 100)
    
    # Plot 2: Trajectory Length Distribution
    if 'results' in results:
        traj_lengths = [r['trajectory_length'] for r in results['results'] if r['trajectory_length'] > 0]
        if traj_lengths:
            axes[0, 1].hist(traj_lengths, bins=20, alpha=0.7, color='blue')
            axes[0, 1].axvline(x=np.mean(traj_lengths), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(traj_lengths):.1f}')
            axes[0, 1].set_xlabel('Trajectory Length (steps)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Trajectory Length Distribution')
            axes[0, 1].legend()
    
    # Plot 3: Reward Distribution
    if 'results' in results:
        rewards = [r['total_reward'] for r in results['results']]
        if rewards:
            axes[1, 0].hist(rewards, bins=20, alpha=0.7, color='purple')
            axes[1, 0].axvline(x=np.mean(rewards), color='red', linestyle='--',
                              label=f'Mean: {np.mean(rewards):.2f}')
            axes[1, 0].set_xlabel('Total Reward')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Reward Distribution')
            axes[1, 0].legend()
    
    # Plot 4: Comparison with Baselines
    baseline_data = {
        'Current Model': success_rate * 100,
        'Target (60%)': 60,
        'DPPO Baseline\n(~100%)': 100,
        'Original DP\n(~90%)': 90
    }
    
    colors = ['orange' if success_rate < 0.6 else 'green', 'red', 'darkgreen', 'blue']
    bars = axes[1, 1].bar(range(len(baseline_data)), list(baseline_data.values()), color=colors)
    axes[1, 1].set_xticks(range(len(baseline_data)))
    axes[1, 1].set_xticklabels(list(baseline_data.keys()), rotation=15, ha='right')
    axes[1, 1].set_ylabel('Success Rate (%)')
    axes[1, 1].set_title('Comparison with Baselines')
    axes[1, 1].set_ylim(0, 105)
    
    # Add value labels on bars
    for bar, value in zip(bars, baseline_data.values()):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Rollout analysis saved: {save_path}")
    plt.close()  # Automatically close the figure


if __name__ == '__main__':
    print("RoboMimic Rollout Evaluator")
    print("This module provides rollout evaluation functionality")
    print("Import and use evaluate_model_rollout() function")