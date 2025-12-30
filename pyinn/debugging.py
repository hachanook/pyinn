"""
INN Debugging Script
----------------------------------------------------------------------------------
Batch testing script that runs all combinations of models and datasets
to verify the training pipeline works correctly.

Copyright (C) 2024  Chanwook Park
Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
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

# Test configurations
run_types = ['regression', 'classification']
interp_methods = ['nonlinear', 'linear', 'MLP']

data_names_regression = ['1D_1D_sine', '2D_1D_sine', '10D_5D_physics', '6D_4D_ansys']
data_names_classification = ['spiral', 'mnist', 'fashion_mnist']

# Default settings (previously from settings.yaml)
DEFAULT_SETTINGS = {
    'GPU': {'gpu_idx': 0},
    'PROBLEM': {'TD_type': 'CP'}
}

# Configure GPU
gpu_idx = DEFAULT_SETTINGS['GPU']['gpu_idx']
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)


def run_debugging():
    """Run all test configurations."""
    print("=" * 70)
    print("INN Debugging - Running all configurations")
    print("=" * 70)

    for run_type in run_types:
        for interp_method in interp_methods:
            if run_type == 'regression':
                for data_name in data_names_regression:
                    print(f"\n{'='*60}")
                    print(f"Testing: {run_type} / {interp_method} / {data_name}")
                    print(f"{'='*60}")

                    try:
                        with open(f'./config/{data_name}.yaml', 'r') as file_dataConfig:
                            config_data = yaml.safe_load(file_dataConfig)
                            config_data['interp_method'] = interp_method
                            config_data['TD_type'] = DEFAULT_SETTINGS['PROBLEM']['TD_type']
                            # Use minimal epochs for debugging
                            config_data['TRAIN_PARAM']['num_epochs_INN'] = 2
                            config_data['TRAIN_PARAM']['num_epochs_MLP'] = 2
                            # Handle sequential mode
                            if isinstance(config_data['MODEL_PARAM']['nmode'], list):
                                config_data['MODEL_PARAM']['nmode'] = config_data['MODEL_PARAM']['nmode'][0]

                        # Data import
                        data = dataset_regression.Data_regression(data_name, config_data)

                        # Train
                        if interp_method in ["linear", "nonlinear"]:
                            regressor = train.Regression_INN(data, config_data)
                        elif interp_method == "MLP":
                            regressor = train.Regression_MLP(data, config_data)
                        regressor.train()

                        print(f"SUCCESS: {run_type} / {interp_method} / {data_name}")

                    except Exception as e:
                        print(f"FAILED: {run_type} / {interp_method} / {data_name}")
                        print(f"Error: {e}")

            if run_type == 'classification':
                for data_name in data_names_classification:
                    # Skip nonlinear for mnist (too slow for debugging)
                    if 'mnist' in data_name and interp_method == 'nonlinear':
                        continue

                    print(f"\n{'='*60}")
                    print(f"Testing: {run_type} / {interp_method} / {data_name}")
                    print(f"{'='*60}")

                    try:
                        with open(f'./config/{data_name}.yaml', 'r') as file_dataConfig:
                            config_data = yaml.safe_load(file_dataConfig)
                            config_data['interp_method'] = interp_method
                            config_data['TD_type'] = DEFAULT_SETTINGS['PROBLEM']['TD_type']
                            # Use minimal epochs for debugging
                            config_data['TRAIN_PARAM']['num_epochs_INN'] = 2
                            config_data['TRAIN_PARAM']['num_epochs_MLP'] = 2

                        # Data import
                        data = dataset_classification.Data_classification(data_name, config_data)

                        # Train
                        if interp_method in ["linear", "nonlinear"]:
                            classifier = train.Classification_INN(data, config_data)
                        elif interp_method == "MLP":
                            classifier = train.Classification_MLP(data, config_data)
                        classifier.train()

                        print(f"SUCCESS: {run_type} / {interp_method} / {data_name}")

                    except Exception as e:
                        print(f"FAILED: {run_type} / {interp_method} / {data_name}")
                        print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("Debugging complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_debugging()
