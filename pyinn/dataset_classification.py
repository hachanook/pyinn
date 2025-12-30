"""
INN Data Module - Classification
----------------------------------------------------------------------------------
Data loading and preprocessing for classification tasks.
Uses NumPy arrays for efficient CPU-to-GPU transfer with JAX.

Copyright (C) 2024  Chanwook Park
Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import numpy as np
import os
import sys
import pandas as pd


def one_hot(labels, num_classes):
    """Convert integer labels to one-hot encoding."""
    one_hot_arr = np.zeros((len(labels), num_classes), dtype=np.float64)
    one_hot_arr[np.arange(len(labels)), np.squeeze(labels)] = 1
    return one_hot_arr


class Data_classification:
    """
    Data container for classification tasks.

    Stores data as NumPy arrays for efficient batch transfer to JAX/GPU.
    No PyTorch dependencies for training - pure NumPy/JAX workflow.
    """

    def __init__(self, data_name: str, config: dict, *args: list) -> None:
        """
        Initialize data container.

        Args:
            data_name: Name of the dataset
            config: Configuration dictionary from YAML file
            *args: Optional additional arguments
        """
        if not os.path.exists('data'):
            os.makedirs('data')

        self.data_dir = 'data/'
        self.data_name = data_name
        self.bool_data_generation = config['DATA_PARAM']['bool_data_generation']
        self.bool_normalize = config['DATA_PARAM']['bool_normalize']
        self.bool_image = config['DATA_PARAM']['bool_image']
        self.bool_shuffle = config['DATA_PARAM']['bool_shuffle']
        self.batch_size = config['TRAIN_PARAM']['batch_size']

        # Load data
        if self.bool_data_generation:
            data_file = self.data_dir + data_name + '.csv'

            try:
                data = np.loadtxt(data_file, delimiter=",", dtype=np.float32, skiprows=1)
            except:
                print(f"Data file {data_file} does not exist. Creating data...")
                data_generation_classification(data_name, config)
                data = np.loadtxt(data_file, delimiter=",", dtype=np.float32, skiprows=1)

            ndata = len(data)

            if self.bool_shuffle:
                np.random.shuffle(data)

            # Split data
            data_train, data_val, data_test = self._split_data(data, config)
        else:
            print("External classification data loading not yet implemented.")
            sys.exit()

        # Set up columns
        if 'mnist' in self.data_name:
            self.input_col = np.arange(1, data.shape[1])
            self.output_col = [0]
        else:
            self.input_col = config['DATA_PARAM']['input_col']
            self.output_col = config['DATA_PARAM']['output_col']

        self.nclass = config['DATA_PARAM']['nclass']
        self.dim = len(self.input_col)
        self.var = self.nclass

        # Extract input/output data
        self.x_data_org = data[:, self.input_col]
        self.u_data_org = data[:, self.output_col].astype(np.int32)
        x_data_train_org = data_train[:, self.input_col]
        u_data_train_org = data_train[:, self.output_col].astype(np.int32)
        x_data_val_org = data_val[:, self.input_col]
        u_data_val_org = data_val[:, self.output_col].astype(np.int32)
        x_data_test_org = data_test[:, self.input_col]
        u_data_test_org = data_test[:, self.output_col].astype(np.int32)

        # Compute normalization bounds
        if self.bool_image:
            self.x_data_minmax = {
                "min": np.zeros(self.x_data_org.shape[1], dtype=np.float64),
                "max": np.ones(self.x_data_org.shape[1], dtype=np.float64) * np.max(self.x_data_org)
            }
        else:
            self.x_data_minmax = {
                "min": self.x_data_org.min(axis=0),
                "max": self.x_data_org.max(axis=0)
            }

        # Normalize if requested
        if self.bool_normalize:
            self.x_data_train = self._normalize(x_data_train_org)
            self.x_data_val = self._normalize(x_data_val_org)
            self.x_data_test = self._normalize(x_data_test_org)
        else:
            self.x_data_train = x_data_train_org.astype(np.float64)
            self.x_data_val = x_data_val_org.astype(np.float64)
            self.x_data_test = x_data_test_org.astype(np.float64)

        # Convert labels to one-hot encoding
        self.u_data_train = one_hot(u_data_train_org, self.nclass)
        self.u_data_val = one_hot(u_data_val_org, self.nclass)
        self.u_data_test = one_hot(u_data_test_org, self.nclass)

        # Store sizes
        self.n_train = len(self.x_data_train)
        self.n_val = len(self.x_data_val)
        self.n_test = len(self.x_data_test)

        print(f'Loaded {ndata} datapoints from {data_name} dataset')
        print(f'  Train: {self.n_train}, Val: {self.n_val}, Test: {self.n_test}')

    def _normalize(self, data):
        """Normalize data to [0, 1] range."""
        return ((data - self.x_data_minmax["min"]) /
                (self.x_data_minmax["max"] - self.x_data_minmax["min"])).astype(np.float64)

    def _split_data(self, data, config):
        """Split data according to split_ratio."""
        ndata = len(data)
        split_ratio = config['DATA_PARAM']['split_ratio']

        # Handle float ratios
        if all(isinstance(item, float) for item in split_ratio):
            if len(split_ratio) == 2:
                train_end = int(split_ratio[0] * ndata)
                return data[:train_end], data[train_end:], data[train_end:]
            elif len(split_ratio) == 3:
                train_end = int(split_ratio[0] * ndata)
                val_end = train_end + int(split_ratio[1] * ndata)
                return data[:train_end], data[train_end:val_end], data[val_end:]

        # Handle integer counts
        elif all(isinstance(item, int) for item in split_ratio):
            if len(split_ratio) == 2:
                train_end = split_ratio[0]
                return data[:train_end], data[train_end:], data[train_end:]
            elif len(split_ratio) == 3:
                train_end = split_ratio[0]
                val_end = train_end + split_ratio[1]
                return data[:train_end], data[train_end:val_end], data[val_end:]

        print("Error: Invalid split ratio")
        sys.exit()

    def __len__(self):
        return len(self.x_data_org)


def data_generation_classification(data_name: str, config):
    """Generate or download classification data."""

    data_dir = './data'

    if data_name == 'mnist' or data_name == 'fashion_mnist':
        # Use torchvision only for downloading MNIST
        try:
            from torchvision import datasets, transforms

            transform = transforms.ToTensor()

            if data_name == 'mnist':
                train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
                test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
            else:
                train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
                test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

            # Convert to numpy and save as CSV
            combined_data = [(int(label), *image.numpy().astype('float16').flatten())
                            for image, label in train_dataset]
            combined_data += [(int(label), *image.numpy().astype('float16').flatten())
                             for image, label in test_dataset]

            columns = ["label"] + [f"pixel{i}" for i in range(784)]
            combined_df = pd.DataFrame(combined_data, columns=columns)
            combined_df.to_csv(os.path.join(data_dir, f'{data_name}.csv'), index=False, header=True)

        except ImportError:
            print("Error: torchvision required for MNIST download. Install with: pip install torchvision")
            sys.exit()

    elif data_name == 'spiral':
        halfSamples = 5000
        noise = 3
        N_SAMPLES = halfSamples * 2

        def genSpiral(deltaT, label, halfSamples, noise):
            points = np.zeros((halfSamples, 3), dtype=np.double)
            for i in range(halfSamples):
                r = i / halfSamples * 5
                t = 3.43 * i / halfSamples * 2 * np.pi + deltaT
                x = r * np.sin(t) + np.random.uniform(-0.1, 0.1) * noise
                y = r * np.cos(t) + np.random.uniform(-0.1, 0.1) * noise
                points[i] = np.array([x, y, label])
            return points

        points1 = genSpiral(0, 1, halfSamples, noise)
        points2 = genSpiral(np.pi, 0, halfSamples, noise)
        points = np.concatenate((points1, points2), axis=0)

        indices = np.arange(N_SAMPLES)
        np.random.shuffle(indices)
        points = points[indices, :]

        df = pd.DataFrame(points, columns=['x1', 'x2', 'u'])
        df.to_csv(os.path.join(data_dir, f'{data_name}.csv'), index=False)

    else:
        raise ValueError(f"Unknown classification dataset: {data_name}")
