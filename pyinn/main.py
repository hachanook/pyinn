"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""
from pyinn import dataset_classification, dataset_regression, model, train, plot # with pyinn library
# import dataset_classification, dataset_regression, model, train, plot # for debugging
from jax import config
import jax.numpy as jnp
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
    

# --------------------- Regression ----------------------------
if run_type == "regression":
    
    ## data import
    data = dataset_regression.Data_regression(data_name, config)

    ## train
    if interp_method == "linear" or interp_method == "nonlinear": # for INN
        if isinstance(config['MODEL_PARAM']['nmode'], list): # sequential training
            nmode_list = config['MODEL_PARAM']['nmode']
            for i, nmode in enumerate(nmode_list):
                config['MODEL_PARAM']['nmode'] = nmode
                if i == 0: # first case, use standard INN trainer
                    regressor = train.Regression_INN(data, config)  
                    regressor.train()  # Train module
                    params = regressor.params
                    errors_train, errors_val, errors_epoch = regressor.errors_train, regressor.errors_val, regressor.errors_epoch, 
                    # plot.plot_regression(regressor, data, config) # plot
                else:
                    
                    regressor_sequential = train.Regression_INN_sequential(data, config, params)
                    regressor_sequential.train()
                    ## concatenate params & loss landscape
                    params_current = regressor_sequential.params
                    if isinstance(params_current, list): # for varying discretization over dimension
                        params = [jnp.concatenate([param, param_current], axis=0) for param, param_current in zip(params, params_current)]
                    else:
                        params = jnp.concatenate([params, params_current], axis=0)
                    errors_train += regressor_sequential.errors_train
                    errors_val += regressor_sequential.errors_val
                    errors_epoch_current = [epoch + i * config['TRAIN_PARAM']['num_epochs_INN'] for epoch in regressor_sequential.errors_epoch]
                    errors_epoch += errors_epoch_current
            
            ## after sequential learning, save merged params and errors in the regressor branch
            regressor.params, regressor.errors_train, regressor.errors_val, regressor.errors_epoch = params, errors_train, errors_val, errors_epoch
            plot.plot_regression(regressor, data, config) # plot



        else: # regular TD training
            regressor = train.Regression_INN(data, config) 
            regressor.train()  # Train module

    elif interp_method == "MLP":
        regressor = train.Regression_MLP(data, config) 
        regressor.train()  # Train module
      
    ## plot
    plot.plot_regression(regressor, data, config)


# --------------------- Classification --------------------------
elif run_type == "classification": 

    ## data import
    data = dataset_classification.Data_classification(data_name, config)
    
    ## train
    if interp_method == "linear" or interp_method == "nonlinear":
        classifier = train.Classification_INN(data, config)  # HiDeNN-TD regressor class
    elif interp_method == "MLP":
        classifier = train.Classification_MLP(data, config)  # HiDeNN-TD regressor class    
    classifier.train()  # Train module

    ## plot
    plot.plot_classification(classifier, data, config)




    