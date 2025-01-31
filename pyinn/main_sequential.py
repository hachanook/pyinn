"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""
# from pyinn import dataset_classification, dataset_regression, model, train, plot # with pyinn library
import dataset_classification, dataset_regression, model, train, plot # for debugging
from jax import config
import jax.numpy as jnp
config.update("jax_enable_x64", True)
import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split, Subset

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
    config['interp_method'] = settings['PROBLEM']["interp_method"]
    config['TD_type'] = settings['PROBLEM']["TD_type"]
    


## data import
data = dataset_regression.Data_regression(data_name, config)
## get original train/val/test dataloader
split_ratio = data.split_ratio
batch_size = int(config['TRAIN_PARAM']['batch_size'])
if config['DATA_PARAM']['bool_random_split'] == True and all(isinstance(item, float) for item in split_ratio):
    # random split with a split ratio
    generator = torch.Generator().manual_seed(42)
    split_data = random_split(dataset=data,lengths=data.split_ratio, generator=generator)
    if len(split_ratio) == 2:
        train_data = split_data[0]
        test_data = split_data[1]
    elif len(split_ratio) == 3:
        train_data = split_data[0]
        val_data = split_data[1]
        test_data = split_data[2]
        val_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
elif config['DATA_PARAM']['bool_random_split'] == False and all(isinstance(item, int) for item in split_ratio):
    # non-random split with a fixed number of data
    if len(split_ratio) == 2:
        train_data = Subset(data, list(range(split_ratio[0])))
        test_data = Subset(data, list(range(split_ratio[0], split_ratio[0]+split_ratio[1])))
    elif len(split_ratio) == 3:
        train_data = Subset(data, list(range(split_ratio[0])))
        test_data = Subset(data, list(range(split_ratio[0], split_ratio[0]+split_ratio[1])))
        val_data = Subset(data, list(range(split_ratio[0]+split_ratio[1], split_ratio[0]+split_ratio[1]+split_ratio[2])))
        val_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


## split sequential data
unique_values, indices = jnp.unique(data.x_data[:,3], return_inverse=True) # Get unique values and their indices
grouped_indices = {}
params = 0
test_errors = []
for sid in range(len(unique_values)): # sid: sequential ID
    print(f"Sequence {sid+1}")
    x_data_sid = data.x_data[jnp.where(indices==sid)]
    u_data_sid = data.u_data[jnp.where(indices==sid)]
    
    ## update data class for this sequence
    data_sid = dataset_regression.Data_regression_squential(data_name, config, x_data_sid, u_data_sid)

    ## Train
    ### define regressor model
    if interp_method == "linear" or interp_method == "nonlinear":
        regressor = train.Regression_INN(data_sid, config)
    elif interp_method == "MLP":
        regressor = train.Regression_MLP(data_sid, config)
    ### update test dataloader from the original data
    regressor.test_dataloader = test_dataloader
    ### update parameters from previous sequence
    if sid != 0:
        regressor.params = params
    regressor.train()
    params = regressor.params # Save current parameters
    test_errors.append(regressor.test_error)

print(test_errors)

## Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(6,5))
# plt.plot(test_errors)
# plt.ylabel('RMSE')
# plt.xlabel('step')
# plt.yscale('log')

plt.tight_layout()

gs = gridspec.GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0])
# plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

ax1.plot(test_errors,  label='Test errors')
ax1.set_xlabel(f"Step", fontsize=16)
ax1.set_ylabel(f"RMSE", fontsize=16)
ax1.tick_params(axis='both', labelsize=12)

parent_dir = os.path.abspath(os.getcwd())
path_figure = os.path.join(parent_dir, 'plots')
fig.savefig(os.path.join(path_figure, data.data_name + "_" + interp_method + "_sequential_test_err") , dpi=300)
plt.close()

# ## plot
# plot.plot_regression(regressor, data, config)



