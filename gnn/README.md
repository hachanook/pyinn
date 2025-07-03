# Graph Neural Network Trainer for Finite Element Analysis

This repository contains a complete implementation of a Message Passing Neural Network (MPNN) for finite element analysis using PyTorch Geometric.

## Overview

The GNN trainer is designed to predict displacement fields (u, v, w) and von Mises stress on 3D tetrahedral meshes. The architecture consists of:

1. **Encoder**: Transforms node features (coordinates) and edge features (vectors + lengths) into latent representations
2. **Message Passing**: Multiple message passing operations in the latent space
3. **Decoder**: Transforms latent representations back to physical quantities

## Architecture

### Input Features
- **Node features**: 3D coordinates (x, y, z)
- **Edge features**: 3D normalized vector + scalar length (4 values total)

### Output Features
- **Displacements**: u, v, w components
- **Von Mises stress**: Scalar stress value

### Model Components
- **MLP**: Multi-layer perceptron with batch normalization
- **MessagePassingLayer**: Custom message passing with MLP-based message and update functions
- **MPNN**: Complete model with encoder, message passing, and decoder

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Install PyTorch Geometric (if not already installed):
```bash
pip install torch-geometric
```

3. For multi-GPU environments, the code automatically uses GPU 0. If you encounter device mismatch errors, run the device test:
```bash
python test_device.py
```

## Usage

### Basic Training

Run the complete training pipeline:

```bash
cd gnn
python main.py
```

This will:
1. Create synthetic FE datasets
2. Initialize the MPNN model
3. Train the model for 100 epochs
4. Evaluate performance
5. Generate visualizations

**Note**: All output files will be saved within the `gnn/` directory.

### Custom Training

```python
from dataset import create_fe_dataset
from model import create_mpnn_model
from trainer import create_trainer

# Configuration
config = {
    'node_input_dim': 3,
    'edge_input_dim': 4,
    'hidden_dim': 64,
    'latent_dim': 32,
    'num_message_passing': 3,
    'output_dim': 4,
    'batch_size': 16,
    'learning_rate': 0.001,
    'num_epochs': 100
}

# Create datasets
train_dataset, val_dataset = create_fe_dataset(
    num_samples=500,
    num_nodes=20,
    train_ratio=0.8
)

# Create model
model = create_mpnn_model(config)

# Create trainer
trainer = create_trainer(model, train_dataset, val_dataset, config)

# Train
trainer.train(config['num_epochs'])

# Evaluate
results, predictions, targets = trainer.evaluate()
```

### Single Mesh Prediction

```python
from dataset import FEDataset
from model import create_mpnn_model
import torch

# Create a single mesh
dataset = FEDataset(num_samples=1, num_nodes=20)
mesh_data = dataset[0]

# Create and load model
config = {
    'node_input_dim': 3,
    'edge_input_dim': 4,
    'hidden_dim': 64,
    'latent_dim': 32,
    'num_message_passing': 3,
    'output_dim': 4
}

model = create_mpnn_model(config)
model.load_state_dict(torch.load('gnn_outputs/final_model.pt'))

# Make prediction
model.eval()
with torch.no_grad():
    prediction = model(mesh_data)
    displacements = model.predict_displacements(mesh_data)
    von_mises = model.predict_von_mises(mesh_data)
```

## Visualization

### Mesh Visualization

```python
from utils import visualize_mesh_3d, visualize_field_on_mesh

# Visualize mesh structure
visualize_mesh_3d(mesh_data, output_path='mesh_3d.png')

# Visualize field on mesh
von_mises_values = mesh_data.y[:, 3].numpy()
visualize_field_on_mesh(mesh_data, von_mises_values, 'Von Mises Stress', 
                       output_path='von_mises_field.png')
```

### Training Results

The trainer automatically generates:
- Training history plots
- Prediction vs target scatter plots
- Error distribution histograms
- Detailed comparison plots for each output

## Dataset

The synthetic dataset generates:
- **Random 3D coordinates** for nodes in [0, 10]³
- **Tetrahedral-like connectivity** using nearest neighbor connections
- **Smooth displacement fields** based on trigonometric functions
- **Von Mises stress** derived from displacement gradients

### Dataset Properties
- Configurable number of nodes (default: 20)
- Configurable number of samples (default: 500)
- Train/validation split (default: 80/20)

## Model Configuration

### Key Parameters
- `hidden_dim`: Hidden dimension for all MLPs (default: 64)
- `latent_dim`: Latent space dimension (default: 32)
- `num_message_passing`: Number of message passing operations (default: 3)
- `batch_size`: Training batch size (default: 16)
- `learning_rate`: Learning rate (default: 0.001)

### Architecture Details
- **Encoder MLPs**: 3 layers for nodes, 2 layers for edges
- **Message MLPs**: 2 layers
- **Update MLPs**: 2 layers
- **Decoder MLPs**: 3 layers
- **Activation**: ReLU
- **Normalization**: BatchNorm1d
- **Aggregation**: Mean pooling

## Output Files

Training generates files within the `gnn/` directory:
- `gnn_outputs/final_model.pt`: Trained model weights
- `gnn_outputs/config.json`: Training configuration
- `gnn_outputs/training_history.png`: Loss curves
- `gnn_outputs/predictions_vs_targets.png`: Scatter plots
- `gnn_outputs/error_distributions.png`: Error histograms
- `gnn_outputs/comparison_*.png`: Detailed comparison plots
- `gnn_outputs/checkpoint_epoch_*.pt`: Training checkpoints
- `demo_outputs/`: Demo training outputs
- `demo_mesh_3d.png`: 3D mesh visualization
- `demo_von_mises_field.png`: Von Mises stress field visualization

## Performance Metrics

The trainer evaluates:
- **MSE Loss**: Mean squared error
- **MAE Loss**: Mean absolute error
- **Relative Error**: Mean relative error for each output
- **R² Score**: Correlation coefficient for predictions vs targets

## Extending the Implementation

### Custom Datasets
To use real FE data, modify the `FEDataset` class:
1. Load your mesh data
2. Compute edge features
3. Generate target values
4. Return PyTorch Geometric `Data` objects

### Custom Architectures
To modify the model architecture:
1. Extend the `MPNN` class
2. Modify encoder/decoder MLPs
3. Add custom message passing layers
4. Implement custom loss functions

### Physics-Informed Loss
To add physics constraints:
1. Implement physics-based loss terms
2. Add gradient computation for PDEs
3. Combine with data-driven loss
4. Modify the training loop

## Dependencies

- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.1.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- tqdm >= 4.64.0

## Device Handling

The code is designed to work with multi-GPU environments:
- **Automatic GPU 0 selection**: Uses `cuda:0` if available, falls back to CPU
- **Device consistency**: Ensures all tensors are on the same device
- **Non-blocking transfers**: Uses `non_blocking=True` for efficient GPU transfers
- **Debug utilities**: Device consistency checking and debugging tools

If you encounter device mismatch errors, the code includes automatic fixes and debugging tools.

## License

This implementation is provided as-is for educational and research purposes.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{gnn_fe_trainer,
  title={Graph Neural Network Trainer for Finite Element Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/gnn-fe-trainer}
}
``` 