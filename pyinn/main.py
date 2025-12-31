"""
INN Training Entry Point
----------------------------------------------------------------------------------
Main entry point for training INN-based models and other neural network architectures.
Supports regression and classification tasks with multiple model types.

Copyright (C) 2024  Chanwook Park
Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu

Usage:
    python main.py                          # Use default settings
    python main.py --data_name 2D_1D_sine   # Specify dataset
    python main.py --interp_method MLP      # Specify model type
    python main.py --help                   # Show all options

Supported Models:
    - linear:    Linear INN (Interpolating Neural Network)
    - nonlinear: Nonlinear INN with radial basis functions
    - MLP:       Multi-Layer Perceptron
    - KAN:       Kolmogorov-Arnold Network
    - FNO:       Fourier Neural Operator
"""

import os
import sys
import argparse
import yaml
from jax import config as jax_config
import jax.numpy as jnp
jax_config.update("jax_enable_x64", True)

# Local imports (development mode - no package installation)
import dataset_classification
import dataset_regression
import train
import plot
from model_utils import save_model_data, save_errors_val


# =============================================================================
# DEFAULT SETTINGS (previously in settings.yaml)
# =============================================================================

DEFAULT_SETTINGS = {
    'GPU': {
        'gpu_idx': 0  # GPU device index (set to 0 for single GPU)
    },
    'PROBLEM': {
        'run_type': 'regression',      # 'regression' or 'classification'
        'TD_type': 'CP',               # Tensor decomposition type: 'CP' or 'Tucker'
        'interp_method': 'nonlinear',  # 'linear', 'nonlinear', 'MLP', 'KAN', 'FNO'
    },
    'DATA': {
        'data_name': '1D_1D_sine',     # Dataset name (must match config file in /config)
    }
}

# Available datasets for reference
REGRESSION_DATASETS = [
    '1D_1D_sine', '1D_1D_exp', '1D_2D_sine_exp',
    '2D_1D_sine', '2D_1D_exp', '3D_1D_exp',
    '6D_4D_ansys', '8D_1D_physics', '10D_5D_physics',
]

CLASSIFICATION_DATASETS = [
    'spiral', 'mnist', 'fashion_mnist'
]

SUPPORTED_METHODS = ['linear', 'nonlinear', 'MLP', 'KAN', 'FNO']


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train INN-based models for regression and classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data_name 2D_1D_sine --interp_method linear
  python main.py --data_name mnist --run_type classification --interp_method MLP
  python main.py --data_name 10D_5D_physics --interp_method nonlinear
        """
    )

    parser.add_argument('--gpu_idx', type=int, default=DEFAULT_SETTINGS['GPU']['gpu_idx'],
                        help='GPU device index (default: 0)')
    parser.add_argument('--run_type', type=str, choices=['regression', 'classification'],
                        default=DEFAULT_SETTINGS['PROBLEM']['run_type'], help='Problem type')
    parser.add_argument('--interp_method', type=str, choices=SUPPORTED_METHODS,
                        default=DEFAULT_SETTINGS['PROBLEM']['interp_method'], help='Interpolation/model method')
    parser.add_argument('--data_name', type=str, default=DEFAULT_SETTINGS['DATA']['data_name'],
                        help='Dataset name (must match config file in /config)')
    parser.add_argument('--TD_type', type=str, choices=['CP', 'Tucker'],
                        default=DEFAULT_SETTINGS['PROBLEM']['TD_type'], help='Tensor decomposition type')
    parser.add_argument('--config_dir', type=str, default='./config',
                        help='Directory containing dataset config files')

    return parser.parse_args()


def get_settings(args):
    """
    Build settings dictionary from defaults and command-line arguments.

    Priority: command-line args > environment variables > defaults
    """
    settings = {
        'GPU': {'gpu_idx': DEFAULT_SETTINGS['GPU']['gpu_idx']},
        'PROBLEM': dict(DEFAULT_SETTINGS['PROBLEM']),
        'DATA': dict(DEFAULT_SETTINGS['DATA']),
    }

    # Override with environment variables if set
    if os.environ.get('PYINN_GPU_IDX'):
        settings['GPU']['gpu_idx'] = int(os.environ['PYINN_GPU_IDX'])
    if os.environ.get('PYINN_RUN_TYPE'):
        settings['PROBLEM']['run_type'] = os.environ['PYINN_RUN_TYPE']
    if os.environ.get('PYINN_INTERP_METHOD'):
        settings['PROBLEM']['interp_method'] = os.environ['PYINN_INTERP_METHOD']
    if os.environ.get('PYINN_DATA_NAME'):
        settings['DATA']['data_name'] = os.environ['PYINN_DATA_NAME']
    if os.environ.get('PYINN_TD_TYPE'):
        settings['PROBLEM']['TD_type'] = os.environ['PYINN_TD_TYPE']

    # Override with command-line arguments if provided
    if args.gpu_idx is not None:
        settings['GPU']['gpu_idx'] = args.gpu_idx
    if args.run_type is not None:
        settings['PROBLEM']['run_type'] = args.run_type
    if args.interp_method is not None:
        settings['PROBLEM']['interp_method'] = args.interp_method
    if args.data_name is not None:
        settings['DATA']['data_name'] = args.data_name
    if args.TD_type is not None:
        settings['PROBLEM']['TD_type'] = args.TD_type

    return settings


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_model(settings, config_dir=None):
    """
    Main training function.

    Args:
        settings: Dictionary with GPU, PROBLEM, and DATA settings
        config_dir: Directory containing dataset configuration files

    Returns:
        Trained model (regressor or classifier)
    """
    # Handle config_dir path (cross-platform compatible)
    if config_dir is None:
        # Default: look for config in parent directory relative to this file
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
        if not os.path.exists(config_dir):
            # Fallback: look in current working directory
            config_dir = os.path.join(os.getcwd(), 'config')
    # Normalize path for cross-platform compatibility
    config_dir = os.path.normpath(config_dir)

    # Extract settings
    gpu_idx = settings['GPU']['gpu_idx']
    run_type = settings['PROBLEM']['run_type']
    interp_method = settings['PROBLEM']['interp_method']
    data_name = settings['DATA']['data_name']
    TD_type = settings['PROBLEM']['TD_type']

    # Configure GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

    print("=" * 60)
    print("INN Training Pipeline")
    print("=" * 60)
    print(f"Run type:     {run_type}")
    print(f"Method:       {interp_method}")
    print(f"Dataset:      {data_name}")
    print(f"TD type:      {TD_type}")
    print(f"GPU index:    {gpu_idx}")
    print("=" * 60)

    # Load dataset configuration
    config_path = os.path.join(config_dir, f'{data_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Add settings to config
    config['data_name'] = data_name
    config['interp_method'] = interp_method
    config['TD_type'] = TD_type

    # =========================== REGRESSION ===========================
    if run_type == "regression":
        # Load data
        data = dataset_regression.Data_regression(data_name, config)

        # Select trainer based on method
        if interp_method in ["linear", "nonlinear"]:
            # Handle sequential training for INN
            if isinstance(config['MODEL_PARAM']['nmode'], list):
                nmode_list = config['MODEL_PARAM']['nmode']
                params = None
                errors_train, errors_val, errors_epoch = [], [], []

                for i, nmode in enumerate(nmode_list):
                    config['MODEL_PARAM']['nmode'] = nmode

                    if i == 0:
                        regressor = train.Regression_INN(data, config)
                        regressor.train()
                        params = regressor.params
                        errors_train = list(regressor.errors_train)
                        errors_val = list(regressor.errors_val)
                        errors_epoch = list(regressor.errors_epoch)
                    else:
                        regressor_seq = train.Regression_INN_sequential(data, config, params)
                        regressor_seq.train()

                        # Concatenate parameters
                        params_current = regressor_seq.params
                        if isinstance(params_current, list):
                            params = [jnp.concatenate([p, pc], axis=0)
                                     for p, pc in zip(params, params_current)]
                        else:
                            params = jnp.concatenate([params, params_current], axis=0)

                        # Concatenate error histories
                        errors_train += list(regressor_seq.errors_train)
                        errors_val += list(regressor_seq.errors_val)
                        epoch_offset = i * config['TRAIN_PARAM']['num_epochs_INN']
                        errors_epoch += [e + epoch_offset for e in regressor_seq.errors_epoch]

                # Update regressor with final params and errors
                regressor.params = params
                regressor.errors_train = errors_train
                regressor.errors_val = errors_val
                regressor.errors_epoch = errors_epoch
            else:
                regressor = train.Regression_INN(data, config)
                regressor.train()

        elif interp_method == "MLP":
            regressor = train.Regression_MLP(data, config)
            regressor.train()

        elif interp_method == "KAN":
            regressor = train.Regression_KAN(data, config)
            regressor.train()

        elif interp_method == "FNO":
            regressor = train.Regression_FNO(data, config)
            regressor.train()

        else:
            raise ValueError(f"Unknown interpolation method: {interp_method}")

        # Save model if configured
        if config['TRAIN_PARAM'].get('bool_save_model', False):
            save_model_data(config, data, regressor.params, data_name, interp_method)
            save_errors_val(regressor.errors_val, data_name, interp_method)

        # Plot results
        plot.plot_regression(regressor, data, config)

        return regressor

    # ========================= CLASSIFICATION =========================
    elif run_type == "classification":
        # Load data
        data = dataset_classification.Data_classification(data_name, config)

        # Select trainer based on method
        if interp_method in ["linear", "nonlinear"]:
            classifier = train.Classification_INN(data, config)
        elif interp_method == "MLP":
            classifier = train.Classification_MLP(data, config)
        elif interp_method == "KAN":
            classifier = train.Classification_KAN(data, config)
        elif interp_method == "FNO":
            classifier = train.Classification_FNO(data, config)
        else:
            raise ValueError(f"Unknown interpolation method: {interp_method}")

        classifier.train()

        # Save model if configured
        if config['TRAIN_PARAM'].get('bool_save_model', False):
            save_model_data(config, data, classifier.params, data_name, interp_method)

        # Plot results
        plot.plot_classification(classifier, data, config)

        return classifier

    else:
        raise ValueError(f"Unknown run type: {run_type}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    settings = get_settings(args)

    # Use command-line config_dir if provided, otherwise let train_model use defaults
    config_dir = None
    if hasattr(args, 'config_dir') and args.config_dir:
        # Normalize user-provided path for cross-platform compatibility
        config_dir = os.path.normpath(args.config_dir)

    try:
        model = train_model(settings, config_dir)
        print("\nTraining completed successfully!")
        return model
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
