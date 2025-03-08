"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""
# from pyinn import dataset_classification, dataset_regression, model, train, plot # with pyinn library
import dataset_classification, dataset_regression, model, train, plot # for debugging
from jax import config
config.update("jax_enable_x64", True)
import os
import yaml

# %% User Set up
with open('./pyinn/settings.yaml','r') as file:
    settings = yaml.safe_load(file)

gpu_idx = settings['GPU']['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = settings['PROBLEM']["run_type"]
interp_method = settings['PROBLEM']["interp_method"]
data_name = settings['DATA']["data_name"] 

with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
    config = yaml.safe_load(file_dataConfig)
    config['data_name'] = data_name
    config['interp_method'] = settings['PROBLEM']["interp_method"]
    config['TD_type'] = settings['PROBLEM']["TD_type"]
    

# --------------------- Regression --------------------------3
if run_type == "regression":
    
    ## data import
    data = dataset_regression.Data_regression(data_name, config)

    ## train
    if interp_method == "linear" or interp_method == "nonlinear":
        regressor = train.Regression_INN(data, config)  # HiDeNN-TD regressor class
    elif interp_method == "MLP":
        regressor = train.Regression_MLP(data, config)  # HiDeNN-TD regressor class
    regressor.train()  # Train module

    ## plot
    plot.plot_regression(regressor, data, config)
    
    if regressor.activation == "adaptive" and interp_method == "nonlinear":
        plot.plot_psi(regressor, data, config)


    


