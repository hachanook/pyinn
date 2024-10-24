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
from plot import *

import yaml


# %% User Set up
with open('settings.yaml','r') as file:
    config = yaml.safe_load(file)

cfg_gpu = config['GPU']
gpu_idx = cfg_gpu['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = config['PROBLEM']["run_type"]
interp_method = config['PROBLEM']["interp_method"]
data_name = config['DATA']["data_name"]


with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
    config_dataConfig = yaml.safe_load(file_dataConfig)

cfg_model_param = config_dataConfig['MODEL_PARAM']
nmode = cfg_model_param['nmode']
nelem = cfg_model_param['nelem']
input_col = cfg_model_param['input_col']
output_col = cfg_model_param['output_col']
bool_normalize = cfg_model_param["bool_normalize"]
data_size = cfg_model_param["data_size"]
split_ratio = cfg_model_param["split_ratio"]

cfg_train_param = config_dataConfig['TRAIN_PARAM']
num_epochs = int(cfg_train_param['num_epochs'])
batch_size = int(cfg_train_param['batch_size'])
learning_rate = float(cfg_train_param['learning_rate'])


# # --------------------- data setup for classification --------------------------
# elif run_type == "classification":
#     # data_name  = 'spiral'
#     # data_name = 'mnist'
#     data_name = "fashion_mnist"
#     data_size = 10_000  # 200, 1000, 10_000
#     noise_level = 3  # 0, 1, 2, 3
#     split_ratio = [0.7,0.15]  # [training ratio, validation ratio], default: [0.7, 0.15] / Gamma: [0.8,0.2]

## data import
data = Data_regression(data_name, data_size, input_col=input_col, output_col=output_col,
                        split_ratio=split_ratio, bool_normalize=bool_normalize)

## train
regressor = Regression(data, nmode, nelem)  # HiDeNN-TD regressor class
regressor.train(num_epochs, batch_size, learning_rate)  # Train module

## plot
plot_regression(config_dataConfig['PLOT']['bool_plot'], regressor, config_dataConfig['PLOT']['plot_in_axis'], 
                config_dataConfig['PLOT']['plot_out_axis'], data)


