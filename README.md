# Constraint-Aware Diffusion Policy for Safe Robotic Manipulation

**Bridging Learning-based Generation with Guaranteed-Safe Execution**

This repository implements the Constraint-Aware Diffusion Policy (CADP) for safe robotic manipulation, along with a comprehensive baseline using the RoboMimic low-dimensional dataset.

## 🚀 Quick Start

### 1. System Requirements

- **GPU**: RTX 4070 or similar (8GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space
- **OS**: Linux (Ubuntu 18.04+ recommended)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/CADP.git
cd CADP

# Install dependencies
pip install torch torchvision numpy matplotlib tqdm h5py pathlib
```

### 3. Dataset Setup

The project uses the RoboMimic low-dimensional dataset:

```bash
# Extract the dataset (assuming robomimic_lowdim.zip is in data/)
cd data
unzip robomimic_lowdim.zip
cd ..
```

Expected directory structure:
```
data/
└── robomimic_lowdim/
    └── robomimic/
        └── datasets/
            ├── lift/
            │   └── ph/
            │       ├── low_dim.hdf5
            │       └── low_dim_abs.hdf5
            ├── can/
            ├── square/
            └── ...
```

### 4. Quick Test

Verify your setup before training:

```bash
python test_robomimic_setup.py
```

This will test:
- ✅ Data availability and structure
- ✅ GPU availability and memory
- ✅ Model creation and forward pass

### 5. Train Baseline Model

Train the RoboMimic diffusion policy baseline:

```bash
# Basic training (30 epochs, ~2-3 hours on RTX 4070)
python train_robomimic_baseline.py

# Custom configuration
python train_robomimic_baseline.py \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --max_demos 200
```

Training progress will be saved in:
- `checkpoints/`: Model checkpoints
- `results/`: Final model and evaluation plots

## 📁 Project Structure

```
CADP/
├── src/                           # Core implementation
│   ├── data/
│   │   └── robomimic_dataset.py  # Dataset loading and processing
│   ├── models/
│   │   └── diffusion_model.py    # Diffusion policy model
│   ├── training/
│   │   └── trainer.py            # Training manager
│   └── evaluation/
│       └── evaluator.py          # Evaluation and metrics
├── data/                          # Datasets
│   └── robomimic_lowdim/         # RoboMimic data
├── checkpoints/                   # Training checkpoints
├── results/                       # Final results and plots
├── experiments/                   # Experiment configurations
├── train_robomimic_baseline.py   # Main training script
├── test_robomimic_setup.py       # Setup verification
└── README.md                     # This file
```

## 🎯 Features

### RoboMimic Baseline Implementation

- **Optimized for RTX 4070**: Reduced model size and memory usage
- **Efficient Data Loading**: HDF5-based dataset with smart caching
- **Mixed Precision Training**: Faster training with automatic mixed precision
- **Comprehensive Evaluation**: Detailed metrics and visualization
- **Modular Design**: Easy to extend for CADP implementation

### Key Optimizations

1. **Memory Efficiency**:
   - Reduced horizon: 16 steps (vs. 20)
   - Smaller model: 64/128/128 hidden dims
   - Gradient accumulation: Effective batch size 16
   - Mixed precision training

2. **Training Speed**:
   - Optimized data loading with multiprocessing
   - Cached normalization statistics
   - Progressive evaluation schedule

3. **Monitoring**:
   - Real-time loss tracking
   - GPU memory monitoring
   - Learning rate scheduling
   - Comprehensive logging

## 📊 Expected Performance

### Training Metrics (RTX 4070)
- **Training Time**: ~2-3 hours (30 epochs)
- **GPU Memory**: 3-4GB peak usage
- **Model Size**: ~2.5M parameters
- **Final MSE**: <0.01 (typical)

### Evaluation Results
- **MSE**: Mean squared error on validation set
- **MAE**: Mean absolute error per action dimension
- **Trajectory Plots**: Visual comparison of predicted vs. true actions
- **Error Analysis**: Temporal and dimensional error breakdown

## 🔧 Usage Examples

### Basic Training
```python
from src.data.robomimic_dataset import find_dataset_file, RoboMimicLowDimDataset
from src.models.diffusion_model import create_model
from src.training.trainer import train_model

# Load data
target_file = find_dataset_file()
dataset = RoboMimicLowDimDataset(target_file, horizon=16, max_demos=100)

# Create model
model = create_model(dataset.obs_dim, dataset.action_dim, dataset.horizon)

# Train
config = {'num_epochs': 30, 'lr': 3e-4, 'weight_decay': 1e-4}
trainer = train_model(model, train_loader, val_loader, config)
```

### Loading Trained Model
```python
from src.evaluation.evaluator import load_model_for_inference

# Load model
model, dataset_info = load_model_for_inference('results/robomimic_baseline_model.pt')

# Use for inference
with torch.no_grad():
    predictions = model(noisy_actions, timesteps, observations)
```

### Custom Dataset Configuration
```python
# Custom observation keys
dataset = RoboMimicLowDimDataset(
    hdf5_path=target_file,
    obs_keys=['robot0_eef_pos', 'robot0_eef_quat', 'object'],
    horizon=20,
    action_dim=7,
    normalize=True
)
```

## 🎨 Visualization

The training script automatically generates:

1. **Training Curves**: Loss vs. epochs with validation tracking
2. **Learning Rate Schedule**: Cosine annealing visualization  
3. **Memory Usage**: GPU memory monitoring
4. **Evaluation Plots**: 
   - Trajectory comparisons (predicted vs. true)
   - Error distributions
   - Temporal error analysis
   - Per-dimension metrics

## 📈 Next Steps: CADP Integration

This baseline provides the foundation for implementing Constraint-Aware Diffusion Policy:

1. **Load Baseline**: Use `load_model_for_inference()` to load the trained baseline
2. **Add Constraints**: Implement safety constraints on top of the baseline model
3. **Constraint Integration**: Modify the sampling process to respect constraints
4. **Comparative Evaluation**: Compare CADP vs. baseline performance and safety

### Planned CADP Features
- [ ] Constraint definition interface
- [ ] Safe sampling algorithms
- [ ] Constraint violation detection
- [ ] Safety-performance trade-off analysis
- [ ] Real robot deployment tools

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size or horizon
   python train_robomimic_baseline.py --batch_size 1 --horizon 12
   ```

2. **Dataset Not Found**:
   ```bash
   # Check data directory structure
   ls -la data/robomimic_lowdim/robomimic/datasets/lift/ph/
   ```

3. **Import Errors**:
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

4. **HDF5 Errors**:
   ```bash
   # Install/update h5py
   pip install --upgrade h5py
   ```

### Performance Tips

1. **Faster Training**:
   - Use smaller `max_demos` for quick testing
   - Increase `batch_size` if you have more VRAM
   - Adjust `gradient_accumulation` for memory/speed trade-off

2. **Better Results**:
   - Increase `num_epochs` for better convergence
   - Try different learning rates (1e-4 to 1e-3)
   - Experiment with model architecture sizes

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{cadp2024,
  title={Constraint-Aware Diffusion Policy for Safe Robotic Manipulation},
  author={[Your Name]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## 🔗 Related Work

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [RoboMimic](https://github.com/ARISE-Initiative/robomimic)
- [DDPM](https://github.com/hojonathanho/diffusion)

---

**Status**: ✅ Baseline Implementation Complete | 🚧 CADP Implementation In Progress

Last Updated: December 2024