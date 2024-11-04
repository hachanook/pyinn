"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)
import os, sys
import importlib.util

# from flax.training import train_state

from dataset_regression import Data_regression
from dataset_classification import Data_classification
from model import *
from train import *
from plot import *

import yaml


# %% User Set up
with open('settings.yaml','r') as file:
    settings = yaml.safe_load(file)

gpu_idx = settings['GPU']['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = settings['PROBLEM']["run_type"]
interp_method = settings['PROBLEM']["interp_method"]
data_name = settings['DATA']["data_name"]

with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
    config = yaml.safe_load(file_dataConfig)

# # --------------------- data setup for classification --------------------------
# elif run_type == "classification":
#     # data_name  = 'spiral'
#     # data_name = 'mnist'
#     data_name = "fashion_mnist"
#     data_size = 10_000  # 200, 1000, 10_000
#     noise_level = 3  # 0, 1, 2, 3
#     split_ratio = [0.7,0.15]  # [training ratio, validation ratio], default: [0.7, 0.15] / Gamma: [0.8,0.2]

# --------------------- Regression --------------------------
if run_type == "regression":
    
    ## data import
    data = Data_regression(data_name, config)

    ## train
    if interp_method == "linear" or interp_method == "nonlinear":
        regressor = Regression_INN(interp_method, data, config)  # HiDeNN-TD regressor class
    elif interp_method == "MLP":
        regressor = Regression_MLP(interp_method, data, config)  # HiDeNN-TD regressor class
    regressor.train()  # Train module

    ## plot
    plot_regression(regressor, data, config)

# --------------------- Classification --------------------------
elif run_type == "classification":

    ## data import
    data = Data_classification(data_name, config)
    
    ## train
    if interp_method == "linear" or interp_method == "nonlinear":
        classifier = Classification_INN(interp_method, data, config)  # HiDeNN-TD regressor class
    elif interp_method == "MLP":
        classifier = Classification_MLP(interp_method, data, config)  # HiDeNN-TD regressor class    
    classifier.train()  # Train module

    ## plot
    plot_classification(classifier, data, config)


    


