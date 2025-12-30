"""
INN Data Module - Regression
----------------------------------------------------------------------------------
Data loading and preprocessing for regression tasks.
Uses NumPy arrays for efficient CPU-to-GPU transfer with JAX.

Copyright (C) 2024  Chanwook Park
Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import pandas as pd
from typing import Sequence
from scipy.stats import qmc


class Data_regression:
    """
    Data container for regression tasks.

    Stores data as NumPy arrays for efficient batch transfer to JAX/GPU.
    No PyTorch dependencies - pure NumPy/JAX workflow.
    """

    def __init__(self, data_name: str, config: dict, *args: list) -> None:
        """
        Initialize data container.

        Args:
            data_name: Name of the dataset (used for data generation)
            config: Configuration dictionary from YAML file
            *args[0]: Optional list of pre-loaded datasets
        """
        if not os.path.exists('data'):
            os.makedirs('data')

        self.data_dir = 'data/'
        self.data_name = data_name
        self.input_col = config['DATA_PARAM']['input_col']
        self.output_col = config['DATA_PARAM']['output_col']
        self.dim = len(self.input_col)
        self.var = len(self.output_col)
        self.bool_normalize = config['DATA_PARAM']['bool_normalize']
        self.bool_shuffle = config['DATA_PARAM']['bool_shuffle']
        self.bool_data_generation = config['DATA_PARAM']['bool_data_generation']
        self.batch_size = config['TRAIN_PARAM']['batch_size']

        # Load and split data
        data_train, data_val, data_test, data, ndata = self._load_data(config, args)

        # Divide into input and output
        x_data_org = data[:, self.input_col]
        u_data_org = data[:, self.output_col]
        x_data_train_org = data_train[:, self.input_col]
        u_data_train_org = data_train[:, self.output_col]
        x_data_val_org = data_val[:, self.input_col]
        u_data_val_org = data_val[:, self.output_col]
        x_data_test_org = data_test[:, self.input_col]
        u_data_test_org = data_test[:, self.output_col]

        # Compute normalization bounds
        self.x_data_minmax = {"min": x_data_org.min(axis=0), "max": x_data_org.max(axis=0)}
        self.u_data_minmax = {"min": u_data_org.min(axis=0), "max": u_data_org.max(axis=0)}

        # Normalize if requested
        if self.bool_normalize:
            self.x_data_train = self._normalize(x_data_train_org, self.x_data_minmax)
            self.u_data_train = self._normalize(u_data_train_org, self.u_data_minmax)
            self.x_data_val = self._normalize(x_data_val_org, self.x_data_minmax)
            self.u_data_val = self._normalize(u_data_val_org, self.u_data_minmax)
            self.x_data_test = self._normalize(x_data_test_org, self.x_data_minmax)
            self.u_data_test = self._normalize(u_data_test_org, self.u_data_minmax)
        else:
            self.x_data_train = x_data_train_org.astype(np.float64)
            self.u_data_train = u_data_train_org.astype(np.float64)
            self.x_data_val = x_data_val_org.astype(np.float64)
            self.u_data_val = u_data_val_org.astype(np.float64)
            self.x_data_test = x_data_test_org.astype(np.float64)
            self.u_data_test = u_data_test_org.astype(np.float64)

        # Store sizes
        self.n_train = len(self.x_data_train)
        self.n_val = len(self.x_data_val)
        self.n_test = len(self.x_data_test)

        print(f'Loaded {ndata} datapoints from {data_name} dataset')
        print(f'  Train: {self.n_train}, Val: {self.n_val}, Test: {self.n_test}')

    def _normalize(self, data, minmax):
        """Normalize data to [0, 1] range."""
        return ((data - minmax["min"]) / (minmax["max"] - minmax["min"])).astype(np.float64)

    def _load_data(self, config, args):
        """Load and split data based on configuration."""

        if self.bool_data_generation:
            # Generate or load synthetic data
            self.data_size = config['DATA_PARAM']['data_size']
            data_file = self.data_dir + self.data_name + '_' + str(self.data_size) + '.csv'

            try:
                data = np.loadtxt(data_file, delimiter=",", dtype=np.float64, skiprows=1)
            except:
                print(f"Data file {data_file} does not exist. Creating data...")
                data_generation_regression(self.data_name, self.data_size, self.input_col)
                data = np.loadtxt(data_file, delimiter=",", dtype=np.float64, skiprows=1)

            ndata = len(data)
            if self.bool_shuffle:
                np.random.shuffle(data)

            data_train, data_val, data_test = self._split_data(data, config)

        elif not self.bool_data_generation and 'data_filenames' in config['DATA_PARAM']:
            # Load from files
            filenames = config['DATA_PARAM']['data_filenames']
            data, data_train, data_val, data_test, ndata = self._load_from_files(filenames, config)

        else:
            # Directly imported data
            data_list = args[0]
            data, data_train, data_val, data_test, ndata = self._load_from_args(data_list, config)

        return data_train, data_val, data_test, data, ndata

    def _split_data(self, data, config):
        """Split data according to split_ratio."""
        ndata = len(data)
        split_ratio = config['DATA_PARAM']['split_ratio']

        if len(split_ratio) == 2:
            train_end = int(split_ratio[0] * ndata)
            return data[:train_end], data[train_end:], data[train_end:]
        elif len(split_ratio) == 3:
            train_end = int(split_ratio[0] * ndata)
            val_end = train_end + int(split_ratio[1] * ndata)
            return data[:train_end], data[train_end:val_end], data[val_end:]
        elif len(split_ratio) == 1 and split_ratio[0] == 1.0:
            return data, data, data
        else:
            print("Error: Invalid split ratio")
            sys.exit()

    def _load_from_files(self, filenames, config):
        """Load data from CSV files."""
        def load_csv(filepath):
            if "mnt" not in filepath:
                filepath = self.data_dir + filepath
            df = pd.read_csv(filepath)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return df[numeric_cols].values.astype(np.float64)

        if len(filenames) == 1:
            data = load_csv(filenames[0])
            self.data = data  # Store for turbulence plotting
            ndata = len(data)
            if self.bool_shuffle:
                np.random.shuffle(data)
            data_train, data_val, data_test = self._split_data(data, config)

        elif len(filenames) == 2:
            data_train = load_csv(filenames[0])
            data_test = load_csv(filenames[1])
            data_val = data_test
            data = np.concatenate([data_train, data_val], axis=0)
            ndata = len(data)

        elif len(filenames) == 3:
            data_train = load_csv(filenames[0])
            data_val = load_csv(filenames[1])
            data_test = load_csv(filenames[2])
            data = np.concatenate([data_train, data_val, data_test], axis=0)
            ndata = len(data)

        return data, data_train, data_val, data_test, ndata

    def _load_from_args(self, data_list, config):
        """Load data from directly passed arrays."""
        if len(data_list) == 1:
            data = data_list[0]
            ndata = len(data)
            if self.bool_shuffle:
                np.random.shuffle(data)
            data_train, data_val, data_test = self._split_data(data, config)

        elif len(data_list) == 2:
            data_train = data_list[0]
            data_val = data_list[1]
            data_test = data_list[1]
            data = np.concatenate([data_train, data_val], axis=0)
            ndata = len(data)

        elif len(data_list) == 3:
            data_train = data_list[0]
            data_val = data_list[1]
            data_test = data_list[2]
            data = np.concatenate([data_train, data_val, data_test], axis=0)
            ndata = len(data)

        return data, data_train, data_val, data_test, ndata

    def __len__(self):
        return self.n_train + self.n_val + self.n_test

    def normalize(self, x_data=None, u_data=None):
        """Normalize data to [0, 1] range."""
        result = []
        if x_data is not None:
            x_norm = (x_data - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])
            result.append(x_norm)
        if u_data is not None:
            u_norm = (u_data - self.u_data_minmax["min"]) / (self.u_data_minmax["max"] - self.u_data_minmax["min"])
            result.append(u_norm)
        return result

    def denormalize(self, x_data=None, u_data=None):
        """Denormalize data from [0, 1] range to original scale."""
        result = []
        if x_data is not None:
            x_org = (self.x_data_minmax["max"] - self.x_data_minmax["min"]) * x_data + self.x_data_minmax["min"]
            result.append(x_org)
        if u_data is not None:
            u_org = (self.u_data_minmax["max"] - self.u_data_minmax["min"]) * u_data + self.u_data_minmax["min"]
            result.append(u_org)
        return result


# =============================================================================
# DATA GENERATION FUNCTIONS
# =============================================================================

def data_generation_regression(data_name: str, data_size: int, input_col: Sequence[int]):
    """Generate synthetic regression data."""

    # Latin Hypercube sampling
    sampler = qmc.LatinHypercube(d=len(input_col))
    x_data_org = sampler.random(n=data_size)

    if data_name == "1D_1D_sine":
        u_data_org = v_fun_1D_1D_sine(x_data_org)
        cols = ['x1', 'u']

    elif data_name == "1D_1D_exp":
        u_data_org = v_fun_1D_1D_exp(x_data_org)
        cols = ['x1', 'u']

    elif data_name == "1D_2D_sine_exp":
        u_data_org = v_fun_1D_2D_sine_exp(x_data_org)
        cols = ['x1', 'u1', 'u2']

    elif data_name == "2D_1D_sine":
        u_data_org = v_fun_2D_1D_sine(x_data_org)
        cols = ['x1', 'x2', 'u']

    elif data_name == "2D_1D_exp":
        u_data_org = v_fun_2D_1D_exp(x_data_org)
        cols = ['x1', 'x2', 'u']

    elif data_name == "3D_1D_exp":
        u_data_org = v_fun_3D_1D_exp(x_data_org)
        cols = ['x1', 'x2', 'x3', 'u']

    elif data_name == "4D_1D_heat_transfer":
        x_min = jnp.array([-2, -2, 0, 1], dtype=jnp.double)
        x_max = jnp.array([2, 2, 0.04, 4], dtype=jnp.double)
        x_data_org = x_data_org * (x_max - x_min) + x_min
        u_data_org = v_fun_4D_1D_heat_transfer(x_data_org)
        cols = ['x1', 'x2', 't', 'k', 'u']

    elif data_name == "8D_1D_physics":
        x_min = jnp.array([0.05, 100, 63_070, 990, 63.1, 700, 1120, 9_855], dtype=jnp.double)
        x_max = jnp.array([0.15, 50_000, 115_600, 1110, 116, 820, 1680, 12_045], dtype=jnp.double)
        x_data_org = x_data_org * (x_max - x_min) + x_min
        u_data_org = jax.vmap(fun_8D_1D_physics)(x_data_org)
        cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'u']

    elif data_name == "10D_5D_physics":
        # Multi-physics benchmark functions
        x_min_1 = jnp.array([0.05, 100, 63_070, 990, 63.1, 700, 1120, 9_855], dtype=jnp.double)
        x_max_1 = jnp.array([0.15, 50_000, 115_600, 1110, 116, 820, 1680, 12_045], dtype=jnp.double)
        x1_data_org = x_data_org[:, :8] * (x_max_1 - x_min_1) + x_min_1

        x_min_2 = jnp.array([30, 0.005, 0.002, 1000, 90_000, 290, 340], dtype=jnp.double)
        x_max_2 = jnp.array([60, 0.020, 0.010, 5000, 110_000, 296, 360], dtype=jnp.double)
        x2_data_org = x_data_org[:, :7] * (x_max_2 - x_min_2) + x_min_2

        x_min_3 = jnp.array([50, 25, 0.5, 1.2, 0.25, 50], dtype=jnp.double)
        x_max_3 = jnp.array([150, 70, 3.0, 2.5, 1.20, 300], dtype=jnp.double)
        x3_data_org = x_data_org[:, :6] * (x_max_3 - x_min_3) + x_min_3

        x_min_4 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.double)
        x_max_4 = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 1.0, 1.0, 1.0, 1.0], dtype=jnp.double)
        x4_data_org = x_data_org[:, :8] * (x_max_4 - x_min_4) + x_min_4

        x_min_5 = jnp.array([150, 220, 6, -10*jnp.pi/180, 16, 0.5, 0.08, 2.5, 1700, 0.025], dtype=jnp.double)
        x_max_5 = jnp.array([200, 300, 10, 10*jnp.pi/180, 45, 1.0, 0.18, 6.0, 2500, 0.080], dtype=jnp.double)
        x5_data_org = x_data_org[:, :10] * (x_max_5 - x_min_5) + x_min_5

        u_data_org = jax.vmap(fun_10D_5D_physics, in_axes=(0, 0, 0, 0, 0))(
            x1_data_org, x2_data_org, x3_data_org, x4_data_org, x5_data_org
        )
        cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
                'u1', 'u2', 'u3', 'u4', 'u5']

    elif data_name == "IGAMapping2D":
        x_min = jnp.array([0, 0], dtype=jnp.double)
        x_max = jnp.array([10, 10], dtype=jnp.double)
        x_data_org = x_data_org * (x_max - x_min) + x_min
        u_data_org = v_fun_IGAMapping2D(x_data_org)
        cols = ['x1', 'x2', 'u1', 'u2']

    else:
        raise ValueError(f"Unknown data_name: {data_name}")

    # Save to CSV
    data = np.concatenate((x_data_org, u_data_org), axis=1)
    df = pd.DataFrame(data, columns=cols)

    parent_dir = os.path.abspath(os.getcwd())
    path_data = os.path.join(parent_dir, 'data')
    csv_filename = f"{data_name}_{data_size}.csv"
    df.to_csv(os.path.join(path_data, csv_filename), index=False)


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def fun_1D_1D_exp(x_data_org):
    u_data_org = jnp.exp(4*x_data_org**2 - 2*x_data_org - 1)
    return u_data_org.reshape(1,)

v_fun_1D_1D_exp = jax.vmap(fun_1D_1D_exp, in_axes=(0))
vv_fun_1D_1D_exp = jax.vmap(v_fun_1D_1D_exp, in_axes=(0))


def fun_1D_1D_sine(x_data_org):
    u_data_org = jnp.sin(2*jnp.pi*x_data_org)
    return u_data_org.reshape(1,)

v_fun_1D_1D_sine = jax.vmap(fun_1D_1D_sine, in_axes=(0))
vv_fun_1D_1D_sine = jax.vmap(v_fun_1D_1D_sine, in_axes=(0))


def fun_1D_2D_sine_exp(x_data_org):
    u1 = jnp.sin(2*jnp.pi*x_data_org)
    u2 = jnp.exp(4*x_data_org**2 - 2*x_data_org - 1)
    return jnp.array([u1, u2], dtype=jnp.double).reshape(-1)

v_fun_1D_2D_sine_exp = jax.vmap(fun_1D_2D_sine_exp, in_axes=(0))
vv_fun_1D_2D_sine_exp = jax.vmap(v_fun_1D_2D_sine_exp, in_axes=(0))


def fun_2D_1D_sine(x_data_org):
    u_data_org = jnp.sin(x_data_org[0] - 2*x_data_org[1]) * jnp.cos(3*x_data_org[0] + x_data_org[1])
    return u_data_org.reshape(1,)

v_fun_2D_1D_sine = jax.vmap(fun_2D_1D_sine, in_axes=(0))
vv_fun_2D_1D_sine = jax.vmap(v_fun_2D_1D_sine, in_axes=(0))


def fun_2D_1D_exp(x_data_org):
    u_data_org = jnp.exp(x_data_org[0] + 2*x_data_org[1])
    return u_data_org.reshape(1,)

v_fun_2D_1D_exp = jax.vmap(fun_2D_1D_exp, in_axes=(0))
vv_fun_2D_1D_exp = jax.vmap(v_fun_2D_1D_exp, in_axes=(0))


def fun_3D_1D_exp(x_data_org):
    u_data_org = (2*x_data_org[2]*jnp.sin(x_data_org[1]) - 3*x_data_org[0]) / jnp.exp(x_data_org[0] - x_data_org[1]**2)
    return u_data_org.reshape(1,)

v_fun_3D_1D_exp = jax.vmap(fun_3D_1D_exp, in_axes=(0))
vv_fun_3D_1D_exp = jax.vmap(v_fun_3D_1D_exp, in_axes=(0))


def fun_4D_1D_heat_transfer(x_data_org):
    u_data_org = (1 - jnp.exp(-15*x_data_org[3]*x_data_org[2])) * jnp.exp(-25*x_data_org[0]**2 - 25*x_data_org[1]**2)
    return u_data_org.reshape(1,)

v_fun_4D_1D_heat_transfer = jax.vmap(fun_4D_1D_heat_transfer, in_axes=(0))
vv_fun_4D_1D_heat_transfer = jax.vmap(v_fun_4D_1D_heat_transfer, in_axes=(0))


def fun_8D_1D_physics(p):
    """Borehole function."""
    p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    u = 2*jnp.pi * p1 * (p4-p6) * (jnp.log(p2/p3) * (1 + 2*(p7*p1) / (jnp.log(p2/p3)*p3**2*p8) + p1/p5))**(-1)
    return jnp.array([u], dtype=jnp.double)


def fun_10D_5D_physics(x1, x2, x3, x4, x5):
    """Five physics functions: Borehole, Piston, OTL Circuit, Robot Arm, Wing Weight."""

    # u1: Borehole function
    p = x1
    p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    u1 = 2*jnp.pi * p1 * (p4-p6) * (jnp.log(p2/p3) * (1 + 2*(p7*p1) / (jnp.log(p2/p3)*p3**2*p8) + p1/p5))**(-1)

    # u2: Piston simulation function
    p = x2
    p1, p2, p3, p4, p5, p6, p7 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    A = p5*p2 + 19.62*p1 - p4*p3/p2
    V = p2/(2*p4) * ((A**2 + 4*p4*p5*p3*p6/p7)**0.5 - A)
    u2 = 2*jnp.pi * (p1/(p4 + p2**2*p5*p3*p6/p7/V**2))**0.5

    # u3: OTL circuit function
    p = x3
    p1, p2, p3, p4, p5, p6 = p[0], p[1], p[2], p[3], p[4], p[5]
    u3 = (((12*p2/(p1+p2) + 0.74) * p6*(p5+9) + 11.35*p3) / (p6*(p5+9)+p3)
          + (0.74*p3*p6*(p5+9)) / ((p6*(p5+9) + p3)*p4))

    # u4: Robot arm function
    p = x4
    x, y = 0.0, 0.0
    for i in range(4):
        angle = 0.0
        for j in range(i+1):
            angle += p[j]
        x += p[i+4] * jnp.cos(angle)
        y += p[i+4] * jnp.sin(angle)
    u4 = (x**2 + y**2)**0.5

    # u5: Wing weight function
    p = x5
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    u5 = (0.036*p1**0.758*p2**0.0035 * (p3/(jnp.cos(p4))**2)**0.6 * p5**0.006*p6**0.04
          * (100*p7/jnp.cos(p4))**(-0.3) * (p8*p9)**0.49 + p1*p10)

    return jnp.array([u1, u2, u3, u4, u5], dtype=jnp.double)


# =============================================================================
# IGA MAPPING FUNCTIONS
# =============================================================================

def in_range(xi, lb, ub):
    return jnp.heaviside(xi-lb, 1) * jnp.heaviside(ub-xi, 0)


def NBasis(xi, L):
    xi /= L
    N1 = (1-xi)**2
    N2 = 2*xi*(1-xi)
    N3 = xi**2
    return jnp.array([N1, N2, N3], dtype=jnp.float64)


def MBasis(eta, L):
    eta /= L
    M1 = (1-eta)**2
    M2 = 2*eta*(1-eta)
    M3 = eta**2
    return jnp.array([M1, M2, M3], dtype=jnp.float64)


def Sum_fun(xieta, L, weights):
    xi, eta = xieta[0], xieta[1]
    N_all = NBasis(xi, L)
    M_all = MBasis(eta, L)
    NM_all = jnp.tensordot(N_all, M_all, axes=0)
    return jnp.sum(NM_all * weights)


def fun_IGAMapping2D(xieta):
    """IGA mapping for 2D NURBS surface."""
    L = 10
    controlPts = np.zeros((3, 3, 2), dtype=np.double)
    controlPts[:, :, 0] = np.array([[0, 0, 0], [10, 15, 20], [10, 15, 20]], dtype=np.float64)
    controlPts[:, :, 1] = np.array([[10, 15, 20], [10, 15, 20], [0, 0, 0]], dtype=np.float64)
    controlPts = jnp.array(controlPts)
    weights = jnp.array([[1, 1, 1],
                         [0.5*jnp.sqrt(2), 0.5*jnp.sqrt(2), 0.5*jnp.sqrt(2)],
                         [1, 1, 1]], dtype=jnp.float64)

    xi, eta = xieta[0], xieta[1]
    N_all = NBasis(xi, L)
    M_all = MBasis(eta, L)
    NM_all = jnp.tensordot(N_all, M_all, axes=0)
    Sum = Sum_fun(xieta, L, weights)
    R_all = NM_all * weights / Sum
    xy = jnp.sum(R_all[:, :, None] * controlPts[:, :, :], axis=(0, 1))
    return xy

v_fun_IGAMapping2D = jax.vmap(fun_IGAMapping2D, in_axes=(0))
vv_fun_IGAMapping2D = jax.vmap(v_fun_IGAMapping2D, in_axes=(0))
