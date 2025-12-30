![INN](/Figure1.png)


## Interpolating Neural Network

This is the github repo for the paper ["Unifying machine learning and interpolation theory via interpolating neural networks"](https://www.nature.com/articles/s41467-025-63790-8), published in Nature Communications.

INN is a lightweight yet precise network architecture that can replace MLPs or any AI/ML models for data training, partial differential equation (PDE) solving, and parameter calibration. 

## Features

* Less trainable parameters than MLP without sacrificing accuracy
* Faster and proven convergent behavior
* Fully differntiable and GPU-optimized

## Project Structure

```
├── pyinn/                  # Main training module
│   ├── main.py             # Entry point for training
│   ├── train.py            # Training loops (INN, MLP, KAN, FNO)
│   ├── model.py            # Model architectures
│   ├── dataset_regression.py    # Regression data loading
│   ├── dataset_classification.py # Classification data loading
│   ├── plot.py             # Loss visualization
│   └── Interpolator.py     # Linear/Nonlinear interpolation
├── config/                 # Training configurations
├── data/                   # Datasets (CSV files)
├── plots/                  # Generated plots
└── requirements.txt        # Python dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd pyinn_als

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 12)
pip install jax[cuda12]
```

## Quick Start

### Training a Model

```bash
cd pyinn

# Train with default settings (linear INN on 1s2p_1o dataset)
python main.py

# Train with specific dataset and method
python main.py --data_name 2D_1D_sine --interp_method linear
python main.py --data_name mnist --run_type classification --interp_method MLP

# Available methods: linear, nonlinear, MLP, KAN, FNO
```

### Running Benchmarks

```bash
# From project root
python -m benchmark.main --config 1s1p_1o   # 2D benchmark
python -m benchmark.main --config 2s1p_1o   # 3D benchmark
python -m benchmark.main --config 2s2p_1o   # 4D benchmark
```

## Configuration

Training configurations are YAML files in `/config`. Example:

```yaml
MODEL_PARAM:
  nmode: 20          # Number of CP modes
  nseg: 20           # Grid segments per dimension
  s_patch: 2         # Nonlinear patch size
  p_order: 2         # Polynomial order

DATA_PARAM:
  input_col: [0,1,2] # Input column indices
  output_col: [3]    # Output column indices
  bool_normalize: False
  bool_shuffle: True

TRAIN_PARAM:
  num_epochs_INN: 1000
  batch_size: 128
  learning_rate: 1e-3
  validation_period: 10
  patience: 50
```

## Supported Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `linear` | Linear INN with CP decomposition | Fast training, smooth functions |
| `nonlinear` | Nonlinear INN with radial basis | Complex nonlinear patterns |
| `MLP` | Multi-Layer Perceptron | General-purpose baseline |
| `KAN` | Kolmogorov-Arnold Network | Interpretable function learning |
| `FNO` | Fourier Neural Operator | Spectral/periodic patterns |

## Datasets

### Regression Datasets

| Dataset | Dimensions | Description |
|---------|------------|-------------|
| `1D_1D_sine` | 1 in, 1 out | Sine function |
| `2D_1D_sine` | 2 in, 1 out | 2D sine surface |
| `6D_4D_ansys` | 6 in, 4 out | Engineering simulation |
| `10D_5D_physics` | 10 in, 5 out | High-dimensional physics |
| `1s2p_1o` | 3 in, 1 out | Parametric function |

### Classification Datasets

| Dataset | Description |
|---------|-------------|
| `spiral` | 2D spiral classification |
| `mnist` | Handwritten digits (requires torchvision) |
| `fashion_mnist` | Fashion items (requires torchvision) |

## Benchmark Framework

The `/benchmark` module compares INN against CP tensor decomposition:

```bash
# Run N-D benchmark
python -m benchmark.main --config 1s2p_1o

# Available configurations:
#   1s1p_1o  - 2D (1 spatial + 1 parametric)
#   1s2p_1o  - 3D (1 spatial + 2 parametric)
#   2s1p_1o  - 3D (2 spatial + 1 parametric)
#   2s2p_1o  - 4D (2 spatial + 2 parametric)
#   3s3p_1o  - 6D (3 spatial + 3 parametric)
```

### Benchmark Metrics

- **RMSE**: Root Mean Square Error
- **Max Error**: Maximum absolute error
- **R² Score**: Coefficient of determination
- **Training Time**: Wall-clock training duration

## API Usage

```python
import yaml
from pyinn.dataset_regression import Data_regression
from pyinn.train import Regression_INN

# Load configuration
with open('config/2D_1D_sine.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['interp_method'] = 'linear'
config['TD_type'] = 'CP'

# Load data
data = Data_regression('2D_1D_sine', config)

# Train model
regressor = Regression_INN(data, config)
regressor.train()

# Access trained parameters
params = regressor.params
test_error = regressor.error_test
```

## License
This repository is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use only.  
This code is protected by U.S. Patent Pending.
For commercial use or licensing, contact: chanwookpark2024@u.northwestern.edu

## Citations
If you found this library useful in academic or industry work, we appreciate your support if you consider 1) starring the project on Github, and 2) citing relevant papers:

```bibtex
@article{park2024engineering,
  title={Engineering software 2.0 by interpolating neural networks: unifying training, solving, and calibration},
  author={Park, Chanwook and Saha, Sourav and Guo, Jiachen and Zhang, Hantao and Xie, Xiaoyu and Bessa, Miguel A and Qian, Dong and Chen, Wei and Wagner, Gregory J and Cao, Jian and others},
  journal={arXiv preprint arXiv:2404.10296},
  year={2024}
}
```
