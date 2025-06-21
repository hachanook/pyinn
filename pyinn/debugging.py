"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""
from pyinn import dataset_classification, dataset_regression, model, train, plot # with pyinn library
# import dataset_classification, dataset_regression, model, train, plot # for debugging
from jax import config
config.update("jax_enable_x64", True)
import os
import yaml

run_types = ['regression', 'classification']
interp_methods = ['nonlinear', 'linear','MLP']

data_names_regression = ['1D_1D_sine', '2D_1D_sine','10D_5D_physics', '6D_4D_ansys']
data_names_classification = ['spiral', 'mnist','fashion_mnist', ]


# %% User Set up
with open('./pyinn/settings.yaml','r') as file:
    settings = yaml.safe_load(file)

gpu_idx = settings['GPU']['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

for run_type in run_types:
    for interp_method in interp_methods:
        if run_type == 'regression':
            for data_name in data_names_regression:

                with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
                    config = yaml.safe_load(file_dataConfig)
                    config['interp_method'] = interp_method
                    config['TD_type'] = settings['PROBLEM']["TD_type"]
                    config['TRAIN_PARAM']['num_epochs_INN'] = 2
                    config['TRAIN_PARAM']['num_epochs_MLP'] = 2
                    if isinstance(config['MODEL_PARAM']['nmode'], list):
                        config['MODEL_PARAM']['nmode'] = config['MODEL_PARAM']['nmode'][0]

                # --------------------- Regression --------------------------3
                    
                ## data import
                data = dataset_regression.Data_regression(data_name, config)

                ## train
                if interp_method == "linear" or interp_method == "nonlinear":
                    regressor = train.Regression_INN(data, config)  
                elif interp_method == "MLP":
                    regressor = train.Regression_MLP(data, config)  
                regressor.train()  # Train module

                ## plot
                plot.plot_regression(regressor, data, config)

                # --------------------- Classification --------------------------
                
        if run_type == 'classification':
            for data_name in data_names_classification:
                if ('mnist' in data_name and interp_method=='nonlinear'):
                    break
                    
                with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
                    config = yaml.safe_load(file_dataConfig)
                    config['interp_method'] = interp_method
                    config['TD_type'] = settings['PROBLEM']["TD_type"]
                    config['TRAIN_PARAM']['num_epochs_INN'] = 2
                    config['TRAIN_PARAM']['num_epochs_MLP'] = 2

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




    