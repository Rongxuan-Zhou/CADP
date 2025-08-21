"""
RoboMimic Low-Dimensional Dataset
Author: CADP Project Team
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any


class RoboMimicDataExplorer:
    """Explore and understand RoboMimic dataset structure"""
    
    def __init__(self, data_dir: str = 'data/robomimic_lowdim/robomimic/datasets'):
        self.data_dir = Path(data_dir)
        
    def find_hdf5_files(self) -> List[Path]:
        """Find all HDF5 files in the dataset"""
        hdf5_files = list(self.data_dir.glob('**/*.hdf5'))
        print(f"Found {len(hdf5_files)} HDF5 files:")
        for f in hdf5_files:
            file_size = f.stat().st_size / 1024**2  # MB
            rel_path = f.relative_to(self.data_dir)
            print(f"  - {rel_path}: {file_size:.1f} MB")
        return hdf5_files
    
    def explore_hdf5_structure(self, hdf5_path: Path) -> List[str]:
        """Explore HDF5 file structure in detail"""
        print(f"\n{'='*60}")
        print(f"Exploring: {hdf5_path.name}")
        print(f"{'='*60}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Top-level structure
            print("Top-level keys:", list(f.keys()))
            
            if 'data' in f:
                data_group = f['data']
                demo_keys = list(data_group.keys())
                print(f"\nNumber of demonstrations: {len(demo_keys)}")
                
                if len(demo_keys) > 0:
                    # Analyze first demonstration
                    first_demo = data_group[demo_keys[0]]
                    print(f"\nDemo '{demo_keys[0]}' structure:")
                    
                    # Actions
                    if 'actions' in first_demo:
                        actions = first_demo['actions'][:]
                        print(f"  - actions: shape={actions.shape}, dtype={actions.dtype}")
                        print(f"    range: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
                        print(f"    mean: {np.mean(actions, axis=0)[:5]}...")  # First 5 dims
                    
                    # Observations
                    if 'obs' in first_demo:
                        obs_group = first_demo['obs']
                        print(f"  - observations:")
                        for obs_key in sorted(obs_group.keys()):
                            obs_data = obs_group[obs_key]
                            print(f"    - {obs_key}: shape={obs_data.shape}, dtype={obs_data.dtype}")
                            if len(obs_data.shape) == 2 and obs_data.shape[-1] < 20:
                                sample = obs_data[0]
                                print(f"      sample: {sample[:5]}...")  # First 5 values
                    
                    # Rewards/dones
                    for key in ['rewards', 'dones']:
                        if key in first_demo:
                            data = first_demo[key][:]
                            print(f"  - {key}: shape={data.shape}")
                            if key == 'rewards':
                                success_rate = np.mean(data > 0)
                                print(f"    success indicators: {success_rate:.2%}")
                            elif key == 'dones':
                                print(f"    episode ends: {np.sum(data)} timesteps")
                
                # Calculate dataset statistics
                total_timesteps = 0
                episode_lengths = []
                
                for demo_key in demo_keys[:10]:  # Check first 10 demos
                    demo = data_group[demo_key]
                    if 'actions' in demo:
                        length = len(demo['actions'])
                        episode_lengths.append(length)
                        total_timesteps += length
                
                if episode_lengths:
                    print(f"\nDataset statistics (first 10 demos):")
                    print(f"  - Total timesteps: {total_timesteps}")
                    print(f"  - Episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
                    print(f"  - Min/Max length: {np.min(episode_lengths)}/{np.max(episode_lengths)}")
            
            # Environment info
            if 'env_args' in f.attrs:
                env_args = f.attrs['env_args']
                print(f"\nEnvironment config: {env_args}")
                
        return demo_keys


class RoboMimicLowDimDataset(Dataset):
    """
    RoboMimic Low-Dimensional Dataset for Diffusion Policy
    Optimized for RTX 4070: reduced sequence length and memory usage
    """
    
    def __init__(self, 
                 hdf5_path: Path, 
                 horizon: int = 16,
                 obs_keys: List[str] = None,
                 action_dim: int = 7,
                 normalize: bool = True,
                 max_demos: Optional[int] = None,
                 add_noise: bool = False,
                 noise_scale: float = 0.01,
                 sampling_strategy: str = 'overlapping'):
        
        if obs_keys is None:
            obs_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']
        
        self.hdf5_path = hdf5_path
        self.horizon = horizon
        self.obs_keys = obs_keys
        self.action_dim = action_dim
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.sampling_strategy = sampling_strategy
        
        # Load data structure without loading actual data
        with h5py.File(hdf5_path, 'r') as f:
            all_demo_keys = list(f['data'].keys())
            if max_demos:
                self.demo_keys = all_demo_keys[:max_demos]
            else:
                self.demo_keys = all_demo_keys
            
            print(f"Dataset initialized with {len(self.demo_keys)} demonstrations")
            
            # Calculate valid starting indices with improved sampling
            self.indices = []
            for demo_key in self.demo_keys:
                demo_length = len(f['data'][demo_key]['actions'])
                
                if sampling_strategy == 'uniform':
                    # Uniform sampling across episode
                    step_size = max(1, (demo_length - horizon) // 8)  # ~8 samples per episode
                    for start_idx in range(0, demo_length - horizon + 1, step_size):
                        self.indices.append((demo_key, start_idx))
                elif sampling_strategy == 'dense':
                    # Dense sampling (every timestep)
                    for start_idx in range(0, demo_length - horizon + 1):
                        self.indices.append((demo_key, start_idx))
                else:  # 'overlapping' (default)
                    # Overlapping windows
                    step_size = max(1, horizon // 4)
                    for start_idx in range(0, demo_length - horizon + 1, step_size):
                        self.indices.append((demo_key, start_idx))
            
            self.num_demos = len(self.demo_keys)
            
            print(f"Generated {len(self.indices)} training samples")
            print(f"Prediction horizon: {horizon} steps")
            
            # Determine observation dimensions
            self.obs_dim = self._calculate_obs_dim(f)
            print(f"Observation dimension: {self.obs_dim}")
            
            # Compute normalization statistics
            if normalize:
                self._compute_normalization_stats(f)
    
    def _calculate_obs_dim(self, h5_file) -> int:
        """Calculate total observation dimension"""
        total_dim = 0
        sample_demo = h5_file['data'][self.demo_keys[0]]
        
        for obs_key in self.obs_keys:
            if obs_key in sample_demo['obs']:
                obs_shape = sample_demo['obs'][obs_key].shape
                if len(obs_shape) > 1:
                    dim = np.prod(obs_shape[1:])  # Flatten all dims except time
                else:
                    dim = 1
                total_dim += dim
                print(f"  - {obs_key}: {obs_shape} -> {dim} dims")
        
        # If no specified keys found, use robot proprioception
        if total_dim == 0 and 'robot0_proprio' in sample_demo['obs']:
            proprio_shape = sample_demo['obs']['robot0_proprio'].shape
            total_dim = np.prod(proprio_shape[1:]) if len(proprio_shape) > 1 else 1
            print(f"  - Using robot0_proprio: {proprio_shape} -> {total_dim} dims")
        
        return total_dim
    
    def _compute_normalization_stats(self, h5_file):
        """Compute normalization statistics"""
        print("Computing normalization statistics...")
        
        all_actions = []
        all_obs = []
        
        # Sample subset of demonstrations for efficiency
        sample_size = min(50, len(self.demo_keys))
        sample_demos = np.random.choice(self.demo_keys, sample_size, replace=False)
        
        for demo_key in sample_demos:
            demo = h5_file['data'][demo_key]
            
            # Actions
            actions = demo['actions'][:]
            if actions.shape[-1] > self.action_dim:
                actions = actions[:, :self.action_dim]
            all_actions.append(actions)
            
            # Observations
            obs_data = self._extract_observation(demo['obs'])
            if obs_data is not None:
                all_obs.append(obs_data)
        
        # Compute statistics
        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            self.action_mean = np.mean(all_actions, axis=0).astype(np.float32)
            self.action_std = (np.std(all_actions, axis=0) + 1e-6).astype(np.float32)
        else:
            self.action_mean = np.zeros(self.action_dim, dtype=np.float32)
            self.action_std = np.ones(self.action_dim, dtype=np.float32)
        
        if all_obs:
            all_obs = np.concatenate(all_obs, axis=0)
            self.obs_mean = np.mean(all_obs, axis=0).astype(np.float32)
            self.obs_std = (np.std(all_obs, axis=0) + 1e-6).astype(np.float32)
        else:
            self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
            self.obs_std = np.ones(self.obs_dim, dtype=np.float32)
        
        print(f"✓ Action stats: mean={self.action_mean[:3]}, std={self.action_std[:3]}")
        print(f"✓ Observation stats computed for {self.obs_dim} dimensions")
    
    def _extract_observation(self, obs_group, timestep: Optional[int] = None):
        """Extract and concatenate observation data"""
        obs_list = []
        
        for obs_key in self.obs_keys:
            if obs_key in obs_group:
                if timestep is not None:
                    obs_data = obs_group[obs_key][timestep]
                    obs_list.append(np.array(obs_data, dtype=np.float32).flatten())
                else:
                    obs_data = obs_group[obs_key][:]
                    # For normalization computation, reshape to (timesteps * features,)
                    obs_list.append(np.array(obs_data, dtype=np.float32))
        
        # Fallback to proprioception if specified keys not found
        if not obs_list and 'robot0_proprio' in obs_group:
            if timestep is not None:
                obs_data = obs_group['robot0_proprio'][timestep]
                obs_list = [np.array(obs_data, dtype=np.float32).flatten()]
            else:
                obs_data = obs_group['robot0_proprio'][:]
                obs_list = [np.array(obs_data, dtype=np.float32)]
        
        if timestep is not None:
            # For single timestep, concatenate flattened features
            return np.concatenate(obs_list) if obs_list else None
        else:
            # For full sequences, concatenate along feature dimension
            if obs_list:
                return np.concatenate(obs_list, axis=-1) if len(obs_list[0].shape) > 1 else np.concatenate(obs_list)
            return None
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get training sample"""
        demo_key, start_idx = self.indices[idx]
        
        with h5py.File(self.hdf5_path, 'r') as f:
            demo = f['data'][demo_key]
            
            # Extract action sequence
            actions = demo['actions'][start_idx:start_idx + self.horizon]
            actions = np.array(actions, dtype=np.float32)
            
            # Truncate to desired action dimension
            if actions.shape[-1] > self.action_dim:
                actions = actions[:, :self.action_dim]
            
            # Extract initial observation
            observation = self._extract_observation(demo['obs'], timestep=start_idx)
            
            # Handle case where observation extraction fails
            if observation is None:
                observation = np.zeros(self.obs_dim, dtype=np.float32)
            
            # Apply data augmentation if enabled
            if self.add_noise:
                # Add Gaussian noise to observations for robustness
                obs_noise = np.random.normal(0, self.noise_scale, observation.shape).astype(np.float32)
                observation = observation + obs_noise
                
                # Add small noise to actions
                action_noise = np.random.normal(0, self.noise_scale * 0.5, actions.shape).astype(np.float32)
                actions = actions + action_noise
            
            # Normalize
            if self.normalize:
                actions = (actions - self.action_mean) / self.action_std
                observation = (observation - self.obs_mean) / self.obs_std
            
            # Data augmentation: add small noise occasionally
            if np.random.random() < 0.1:
                actions += np.random.randn(*actions.shape) * 0.01
        
        return {
            'observation': torch.FloatTensor(observation),
            'action': torch.FloatTensor(actions),
            'demo_key': demo_key,
            'start_idx': start_idx
        }


def create_dataloaders(dataset: RoboMimicLowDimDataset, 
                      batch_size: int = 2, 
                      train_ratio: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Split dataset
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  - Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  - Batch size: {batch_size} (train), {batch_size*2} (val)")
    
    return train_loader, val_loader


def find_dataset_file(data_dir: str = 'data/robomimic_lowdim/robomimic/datasets') -> Optional[Path]:
    """Find the appropriate dataset file for training"""
    explorer = RoboMimicDataExplorer(data_dir)
    hdf5_files = explorer.find_hdf5_files()
    
    # Look for lift task files
    lift_files = [f for f in hdf5_files if 'lift' in str(f)]
    print(f"Found {len(lift_files)} lift task files")
    
    # Choose the professional human (ph) low_dim file for training
    target_file = None
    for f in lift_files:
        if 'ph' in str(f) and 'low_dim.hdf5' in str(f):
            target_file = f
            break
    
    if target_file:
        print(f"Using for training: {target_file}")
        return target_file
    else:
        print("Warning: Could not find lift/ph/low_dim.hdf5 file")
        return hdf5_files[0] if hdf5_files else None