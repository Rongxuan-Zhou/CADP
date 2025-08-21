"""
Multi-Task RoboMimic Dataset
Author: CADP Project Team

Support for training across multiple robotic manipulation tasks
with curriculum learning and task balancing strategies.
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from .robomimic_dataset import RoboMimicLowDimDataset


class MultiTaskRoboMimicDataset(Dataset):
    """Multi-task dataset combining multiple RoboMimic tasks"""
    
    def __init__(self, 
                 task_configs: Dict[str, str],
                 horizon: int = 16,
                 max_demos: Optional[int] = None,
                 normalize: bool = True,
                 task_weights: Optional[Dict[str, float]] = None,
                 curriculum_learning: bool = True):
        """
        Args:
            task_configs: Dict mapping task names to HDF5 file paths
            horizon: Action sequence length
            max_demos: Max demonstrations per task (None = all)
            normalize: Whether to normalize observations and actions
            task_weights: Sampling weights for each task
            curriculum_learning: Whether to use curriculum learning
        """
        self.task_configs = task_configs
        self.horizon = horizon
        self.max_demos = max_demos
        self.normalize = normalize
        self.curriculum_learning = curriculum_learning
        
        # Task difficulty ranking (easy to hard)
        self.task_difficulty = {
            'lift': 1,      # Easiest
            'can': 2, 
            'square': 3,
            'toolhang': 4   # Hardest
        }
        
        # Default task weights based on difficulty
        if task_weights is None:
            self.task_weights = {
                'lift': 1.0,
                'can': 0.8,
                'square': 0.8, 
                'toolhang': 0.6
            }
        else:
            self.task_weights = task_weights
            
        # Load individual task datasets
        self.datasets = {}
        self.task_samples = {}
        self.total_samples = 0
        
        print(f"📚 Loading multi-task dataset...")
        print(f"   Tasks: {list(task_configs.keys())}")
        
        for task_name, hdf5_path in task_configs.items():
            if os.path.exists(hdf5_path):
                dataset = RoboMimicLowDimDataset(
                    hdf5_path=hdf5_path,
                    horizon=horizon,
                    max_demos=max_demos,
                    normalize=normalize,
                    add_noise=True,
                    noise_scale=0.01
                )
                
                self.datasets[task_name] = dataset
                self.task_samples[task_name] = len(dataset)
                self.total_samples += len(dataset)
                
                print(f"   • {task_name}: {len(dataset)} samples ({dataset.num_demos} demos)")
            else:
                print(f"   ❌ Task {task_name} file not found: {hdf5_path}")
        
        # Create sample indices with task labels
        self.sample_indices = []
        for task_name, dataset in self.datasets.items():
            for i in range(len(dataset)):
                self.sample_indices.append((task_name, i))
                
        print(f"   Total samples: {self.total_samples}")
        
        # Compute global normalization statistics
        if normalize:
            self._compute_global_normalization()
    
    def _compute_global_normalization(self):
        """Compute normalization statistics across all tasks"""
        print("🔧 Computing global normalization statistics...")
        
        # For multi-task scenarios with different observation dimensions,
        # we use task-specific normalization instead of global normalization
        
        all_actions = []
        
        for task_name, dataset in self.datasets.items():
            # Collect actions from each task (actions are consistent across tasks)
            task_actions = []
            
            for i in range(min(100, len(dataset))):  # Sample for efficiency
                sample = dataset[i]
                task_actions.append(sample['action'].reshape(-1))  # Flatten
            
            if task_actions:
                all_actions.extend(task_actions)
        
        if all_actions:
            # Global action normalization only (actions are consistent)
            action_array = np.array(all_actions)  # Shape: [N_samples * horizon, action_dim]
            action_reshaped = action_array.reshape(-1, 7)  # Flatten to [N_total, 7]
            self.global_action_mean = action_reshaped.mean(axis=0)  # Shape: [7]
            self.global_action_std = action_reshaped.std(axis=0)    # Shape: [7] 
            self.global_action_std[self.global_action_std < 1e-6] = 1.0
            
            # Apply global action normalization to all task datasets
            # Keep task-specific observation normalization
            for dataset in self.datasets.values():
                dataset.action_mean = self.global_action_mean
                dataset.action_std = self.global_action_std
                # Note: Keep each dataset's own obs_mean and obs_std
            
            # Set action normalization attributes for this multi-task dataset
            self.action_mean = self.global_action_mean
            self.action_std = self.global_action_std
                
            print(f"   ✅ Global action normalization computed (task-specific obs normalization kept)")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Get sample with task information"""
        task_name, task_idx = self.sample_indices[idx]
        sample = self.datasets[task_name][task_idx]
        
        # Add task information
        sample['task_name'] = task_name
        sample['task_id'] = list(self.task_configs.keys()).index(task_name)
        sample['task_weight'] = self.task_weights.get(task_name, 1.0)
        
        return sample
    
    def get_task_sample(self, task_name: str, idx: int):
        """Get sample from specific task"""
        if task_name not in self.datasets:
            raise ValueError(f"Task {task_name} not found")
        return self.datasets[task_name][idx]
    
    def get_task_info(self) -> Dict[str, Any]:
        """Get information about all tasks"""
        info = {}
        for task_name, dataset in self.datasets.items():
            info[task_name] = {
                'num_demos': dataset.num_demos,
                'num_samples': len(dataset),
                'obs_dim': dataset.obs_dim,
                'action_dim': dataset.action_dim,
                'horizon': dataset.horizon,
                'difficulty': self.task_difficulty.get(task_name, 3),
                'weight': self.task_weights.get(task_name, 1.0)
            }
        return info


class CurriculumSampler:
    """Curriculum learning sampler for multi-task training"""
    
    def __init__(self, 
                 dataset: MultiTaskRoboMimicDataset,
                 curriculum_schedule: str = 'linear',
                 total_epochs: int = 100):
        self.dataset = dataset
        self.curriculum_schedule = curriculum_schedule
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Sort tasks by difficulty
        self.task_order = sorted(
            dataset.task_configs.keys(),
            key=lambda t: dataset.task_difficulty.get(t, 3)
        )
        
    def update_epoch(self, epoch: int):
        """Update current training epoch for curriculum"""
        self.current_epoch = epoch
        
    def get_task_probabilities(self) -> Dict[str, float]:
        """Get sampling probabilities for each task based on curriculum"""
        if not self.dataset.curriculum_learning:
            # Equal probability for all tasks
            num_tasks = len(self.dataset.datasets)
            return {task: 1.0/num_tasks for task in self.dataset.datasets.keys()}
        
        # Curriculum-based probabilities
        progress = self.current_epoch / self.total_epochs
        
        if self.curriculum_schedule == 'linear':
            # Linear introduction of tasks
            num_active_tasks = max(1, int(progress * len(self.task_order)) + 1)
            active_tasks = self.task_order[:num_active_tasks]
            
            probs = {}
            for task in self.dataset.datasets.keys():
                if task in active_tasks:
                    probs[task] = self.dataset.task_weights.get(task, 1.0)
                else:
                    probs[task] = 0.0
                    
            # Normalize probabilities
            total_weight = sum(probs.values())
            if total_weight > 0:
                probs = {k: v/total_weight for k, v in probs.items()}
            
            return probs
        
        else:  # Default: all tasks with original weights
            return self.dataset.task_weights


def multi_task_collate_fn(batch):
    """Custom collate function for multi-task batches with different observation dimensions"""
    
    # Find the maximum observation dimension in this batch
    max_obs_dim = max(sample['observation'].shape[-1] for sample in batch)
    
    # Prepare tensors
    observations = []
    actions = []
    task_names = []
    task_ids = []
    task_weights = []
    
    for sample in batch:
        obs = sample['observation']  # Shape: [obs_dim]
        
        # Pad observations to max dimension
        if obs.shape[-1] < max_obs_dim:
            padding_size = max_obs_dim - obs.shape[-1]
            obs_padded = torch.cat([obs, torch.zeros(padding_size, dtype=obs.dtype)], dim=-1)
        else:
            obs_padded = obs
            
        observations.append(obs_padded)
        actions.append(sample['action'])
        task_names.append(sample['task_name'])
        task_ids.append(sample['task_id'])
        task_weights.append(sample['task_weight'])
    
    # Stack tensors
    observations = torch.stack(observations, dim=0)  # [batch_size, max_obs_dim]
    actions = torch.stack(actions, dim=0)  # [batch_size, horizon, action_dim]
    task_ids = torch.tensor(task_ids, dtype=torch.long)
    task_weights = torch.tensor(task_weights, dtype=torch.float32)
    
    return {
        'observation': observations,
        'action': actions,
        'task_name': task_names,
        'task_id': task_ids,
        'task_weight': task_weights,
        'max_obs_dim': max_obs_dim
    }


def create_multi_task_dataloaders(
    task_configs: Dict[str, str],
    horizon: int = 16,
    batch_size: int = 16,
    max_demos: Optional[int] = None,
    train_ratio: float = 0.8,
    task_weights: Optional[Dict[str, float]] = None,
    curriculum_learning: bool = True
) -> Tuple[DataLoader, DataLoader, MultiTaskRoboMimicDataset]:
    """Create multi-task train and validation dataloaders"""
    
    # Create multi-task dataset
    dataset = MultiTaskRoboMimicDataset(
        task_configs=task_configs,
        horizon=horizon,
        max_demos=max_demos,
        normalize=True,
        task_weights=task_weights,
        curriculum_learning=curriculum_learning
    )
    
    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=multi_task_collate_fn
    )
    
    print(f"\n📊 Multi-task dataloaders created:")
    print(f"   • Train samples: {train_size} ({len(train_loader)} batches)")
    print(f"   • Validation samples: {val_size} ({len(val_loader)} batches)")
    print(f"   • Batch size: {batch_size} (train), {batch_size*2} (val)")
    
    return train_loader, val_loader, dataset


def analyze_task_distribution(dataset: MultiTaskRoboMimicDataset):
    """Analyze and display task distribution"""
    info = dataset.get_task_info()
    
    print(f"\n📈 Task Distribution Analysis:")
    print("-" * 60)
    
    total_samples = sum(info[task]['num_samples'] for task in info)
    
    for task_name, task_info in info.items():
        percentage = task_info['num_samples'] / total_samples * 100
        difficulty = task_info['difficulty']
        weight = task_info['weight']
        
        print(f"  • {task_name.upper()}:")
        print(f"    - Samples: {task_info['num_samples']} ({percentage:.1f}%)")
        print(f"    - Demos: {task_info['num_demos']}")
        print(f"    - Obs dim: {task_info['obs_dim']}")
        print(f"    - Difficulty: {difficulty}/4")
        print(f"    - Weight: {weight:.1f}")
        print()
    
    print(f"Total samples: {total_samples}")
    print("-" * 60)