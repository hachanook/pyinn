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
with open('./code/settings.yaml','r') as file:
    settings = yaml.safe_load(file)

gpu_idx = settings['GPU']['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = settings['PROBLEM']["run_type"]
interp_method = settings['PROBLEM']["interp_method"]
data_name = settings['DATA']["data_name"]

with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
    config = yaml.safe_load(file_dataConfig)
    config['interp_method'] = settings['PROBLEM']["interp_method"]
    config['TD_type'] = settings['PROBLEM']["TD_type"]
    

# --------------------- Regression --------------------------
if run_type == "regression":
    
    ## data import
    data = Data_regression(data_name, config)

    ## train
    if interp_method == "linear" or interp_method == "nonlinear":
        regressor = Regression_INN(data, config)  # HiDeNN-TD regressor class
    elif interp_method == "MLP":
        regressor = Regression_MLP(data, config)  # HiDeNN-TD regressor class
    regressor.train()  # Train module

    ## plot
    plot_regression(regressor, data, config)

# --------------------- Classification --------------------------
elif run_type == "classification":

    ## data import
    data = Data_classification(data_name, config)
    
    ## train
    if interp_method == "linear" or interp_method == "nonlinear":
        classifier = Classification_INN(data, config)  # HiDeNN-TD regressor class
    elif interp_method == "MLP":
        classifier = Classification_MLP(data, config)  # HiDeNN-TD regressor class    
    classifier.train()  # Train module

    ## plot
    plot_classification(classifier, data, config)




    


