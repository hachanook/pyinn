"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)
import os
import importlib.util

# from flax.training import train_state

from dataset import Data_regression
from model import *
from train import *

# %% User Set up

# --------------------- Global setup -----------------------------
gpu_idx = 0  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = "regression"
# run_type = "classification"
# run_type = "calibration"
method = "Linear"
# method = "Nonlinear"

plot_bool = False
plot_axis = [0,1]  # for more than 2 inputs, select two axes to visualize [5, 20] for Gamma

# --------------------- data setup for regression --------------------------
if run_type == "regression" or run_type == "calibration":
    # data_name = "1D_1D_sine"  
    # data_name = '1D_1D_exp' 
    # data_name = "1D_2D_sine_exp"
    # data_name = '2D_1D_sine'
    # data_name = '2D_1D_exp'
    data_name = '8D_1D_physics' # borehole equation
    # data_name = '10D_5D_physics' # five physics equations

    data_size = 100_000  # numeric values
    split_ratio = [0.7,0.15,0.15]  # [training ratio, validation ratio], default: [0.7, 0.15]


# --------------------- data setup for classification --------------------------
elif run_type == "classification":
    # data_name  = 'spiral'
    # data_name = 'mnist'
    data_name = "fashion_mnist"
    data_size = 10_000  # 200, 1000, 10_000
    noise_level = 3  # 0, 1, 2, 3
    split_ratio = [0.7,0.15]  # [training ratio, validation ratio], default: [0.7, 0.15] / Gamma: [0.8,0.2]

# %%
if data_name == "1D_1D_sine" or data_name == "1D_1D_exp":
    nmode_H = 1
    nelem = 5
    num_epochs = 50
    batch_size = 50
    learning_rate = 1e-3
    
    data = Data_regression(data_name,data_size,input_col=[0],output_col=[1],
        split_ratio=split_ratio,bool_normalize=True)

if data_name == "1D_2D_sine_exp":
    nmode_H = 1
    nelem = 5
    num_epochs = 50
    batch_size = 50
    learning_rate = 1e-3
    
    data = Data_regression(data_name,data_size,input_col=[0],output_col=[1, 2],
        split_ratio=split_ratio,bool_normalize=True)
        
    
if data_name == "2D_1D_sine" or data_name == "2D_1D_exp":
    nmode_H = 3
    nelem = 10
    num_epochs = 50
    batch_size = 50
    learning_rate = 1e-3

    data = Data_regression(data_name,data_size,input_col=[0,1],output_col=[1],
        split_ratio=split_ratio,bool_normalize=True)

if data_name == "8D_1D_physics": # borehole equation
    nmode_H = 10
    nelem = 10
    num_epochs = 50
    batch_size = 200
    learning_rate = 1e-3

    data = Data_regression(data_name,data_size,input_col=list(range(8)),output_col=[8],
        split_ratio=split_ratio,bool_normalize=True)
    
if data_name == "10D_5D_physics": # five physics equation
    nmode_H = 10
    nelem = 10
    num_epochs = 50
    batch_size = 200
    learning_rate = 1e-3

    data = Data_regression(data_name,data_size,input_col=list(range(10)),output_col=list(range(10,15)),
        split_ratio=split_ratio,bool_normalize=True)


regressor = Regression(data, nmode_H, nelem)  # HiDeNN-TD regressor class
regressor.train(num_epochs, batch_size, learning_rate)  # Train module
