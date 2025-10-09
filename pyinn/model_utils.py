"""
Model utilities for INN - Save, Load, and List models
----------------------------------------------------------------------------------
This module provides comprehensive functionality for saving, loading, and managing
saved INN models.
"""

import os
import csv
import pickle
import glob
import jax.numpy as jnp
import jax
import sys
from model import INN_linear, INN_nonlinear, MLP

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dataset_classification, dataset_regression, train

def get_linspace(xmin, xmax, nnode):
    return jnp.linspace(xmin,xmax,nnode, dtype=jnp.float64)
# v_get_linspace = jax.vmap(get_linspace, in_axes=(0,0,None))

def save_model_data(config, data, params, data_name, interp_method):
    """
    Save model data to pickle file
    
    Args:
        config (dict): Configuration dictionary
        data: Data instance (regression or classification)
        params: Model parameters
        data_name (str): Name of the dataset
        interp_method (str): Interpolation method used
    """
    # Create model_saved directory if it doesn't exist
    model_save_dir = './pyinn/model_saved'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Prepare model data for saving
    model_data = {
        'config': config,
        'x_data_minmax': data.x_data_minmax,
        'params': params
    }
    
    # Add u_data_minmax for regression models
    if hasattr(data, 'u_data_minmax'):
        model_data['u_data_minmax'] = data.u_data_minmax
    
    # Create filename with data name and method
    model_filename = f"{data_name}_{interp_method}_model.pkl"
    model_path = os.path.join(model_save_dir, model_filename)
    
    # Save model data
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_path}")

def save_errors_val(errors_val, data_name, interp_method):
    """
    Save validation errors to a CSV file with the same base name as the model .pkl.

    Args:
        errors_val (list): List of validation errors per epoch
        data_name (str): Name of the dataset
        interp_method (str): Interpolation method used
    """
    model_save_dir = './pyinn/model_saved'
    os.makedirs(model_save_dir, exist_ok=True)

    # Match the model filename used in save_model_data, but with .csv extension
    model_filename = f"{data_name}_{interp_method}_errors_val.pkl"
    csv_filename = os.path.splitext(model_filename)[0] + ".csv"
    csv_path = os.path.join(model_save_dir, csv_filename)

    # Write header and one value per row
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['errors_val'])
        for v in errors_val:
            try:
                writer.writerow([float(v)])
            except Exception:
                # Fallback for non-scalar types
                try:
                    writer.writerow([float(jnp.asarray(v))])
                except Exception:
                    writer.writerow([v])

    print(f"Validation errors saved to: {csv_path}")

def load_saved_model(data_name, interp_method):
    """
    Load saved model data from pickle file
    
    Args:
        data_name (str): Name of the dataset
        interp_method (str): Interpolation method used (linear, nonlinear, MLP)
    
    Returns:
        dict: Dictionary containing model data with keys:
            - config: Configuration dictionary
            - x_data_minmax: Input data normalization parameters
            - u_data_minmax: Output data normalization parameters (for regression)
            - params: Model parameters
    """
    model_save_dir = './pyinn/model_saved'
    model_filename = f"{data_name}_{interp_method}_model.pkl"
    model_path = os.path.join(model_save_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"Model loaded from: {model_path}")
    return model_data

def create_model_from_saved_data(model_data, run_type):
    """
    Create a model instance from loaded model data for inference
    
    Args:
        model_data (dict): Loaded model data from load_saved_model()
        run_type (str): 'regression' or 'classification'
    
    Returns:
        tuple: (model_instance, data_instance) for inference
    """
    config = model_data['config']
    data_name = config['data_name']
    interp_method = config['interp_method']
    
    
    # self.interp_method = config['interp_method']
    # self.cls_data = cls_data
    # self.config = config
    # self.key = int(time.time())
    # self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_INN'])
    
    ## Initialize trainable parameters for INN.
    if 'linear' in interp_method: # for INN
        nmode = int(config['MODEL_PARAM']['nmode'])
        if isinstance(config['MODEL_PARAM']['nseg'], int): # same discretization across dimension
            
            nseg = int(config['MODEL_PARAM']['nseg'])
            nnode = nseg + 1
            
            ## initialization of trainable parameters
            if config["DATA_PARAM"]["bool_normalize"]: # when the data is normalized
                grid_dms = jnp.linspace(0, 1, nnode, dtype=jnp.float64) # (nnode,) the most efficient way
            else: # when the data is not normalized
                grid_dms = [get_linspace(xmin, xmax, nnode) for (xmin, xmax) in zip(config["DATA_PARAM"]["x_data_minmax"]["min"], config["DATA_PARAM"]["x_data_minmax"]["max"])]
            # params = jax.random.uniform(jax.random.PRNGKey(self.key), (nmode, config["DATA_PARAM"]["dim"], 
            #                                     config["DATA_PARAM"]["var"], nnode), dtype=jnp.double)       
            # numParam = nmode*config["DATA_PARAM"]["dim"]*config["DATA_PARAM"]["var"]*nnode
        
        elif isinstance(config['MODEL_PARAM']['nseg'], list): # varying discretization across dimension

            nseg = jnp.array(config['MODEL_PARAM']['nseg'], dtype=jnp.int64) # (dim,) 1D array of integers
            nnode = nseg + 1

            # if len(self.nseg) != cls_data.dim:
            #     print(f"Error: lenth of nseg {len(self.nseg)} is different from input dimension {cls_data.dim}. Check config file.")
            #     sys.exit()

            ## initialization of trainable parameters
            grid_dms, params, numParam = [], [], 0
            for idm, nnode_idm in enumerate(nnode):
                if config["DATA_PARAM"]["bool_normalize"]: # when the data is normalized
                    grid_dms.append(jnp.linspace(0, 1, nnode_idm, dtype=jnp.float64))
                else: # when the data is not normalized
                    grid_dms.append(get_linspace(config["DATA_PARAM"]["x_data_minmax"]["min"][idm], config["DATA_PARAM"]["x_data_minmax"]["max"][idm], nnode_idm))
                # params.append(jax.random.uniform(jax.random.PRNGKey(self.key), (nmode, config["DATA_PARAM"]["var"], nnode_idm), dtype=jnp.double))
                # numParam += nmode*config["DATA_PARAM"]["var"]*nnode_idm 

    ## Define model
    if interp_method == "linear":
        model = INN_linear(grid_dms, config)
        # forward = model.forward
        # v_forward = model.v_forward
        # vv_forward = model.vv_forward
        
    elif interp_method == "nonlinear":
        model = INN_nonlinear(grid_dms, config)
        # forward = model.forward
        # v_forward = model.v_forward
        # vv_forward = model.vv_forward

    elif interp_method == "MLP":
        model = MLP(config['MODEL_PARAM']['activation'])

    
    # # Create data instance
    # if run_type == "regression":
    #     data = dataset_regression.Data_regression(data_name, config)
    # else:  # classification
    #     data = dataset_classification.Data_classification(data_name, config)
    
    # # Create model instance
    # if interp_method == "linear" or interp_method == "nonlinear":
    #     if run_type == "regression":
    #         model = train.Regression_INN(data, config)
    #     else:  # classification
    #         model = train.Classification_INN(data, config)
    # elif interp_method == "MLP":
    #     if run_type == "regression":
    #         model = train.Regression_MLP(data, config)
    #     else:  # classification
    #         model = train.Classification_MLP(data, config)
    
    # Set the loaded parameters
    # model.params = model_data['params']
    
    return model

def list_saved_models():
    """
    List all saved models in the model_saved directory
    """
    model_save_dir = './pyinn/model_saved'
    
    if not os.path.exists(model_save_dir):
        print(f"Model save directory does not exist: {model_save_dir}")
        print("No models have been saved yet.")
        return
    
    # Find all .pkl files in the model_saved directory
    model_files = glob.glob(os.path.join(model_save_dir, "*.pkl"))
    
    if not model_files:
        print(f"No saved models found in {model_save_dir}")
        return
    
    print(f"Found {len(model_files)} saved model(s) in {model_save_dir}:")
    print("-" * 80)
    
    for i, model_file in enumerate(sorted(model_files), 1):
        filename = os.path.basename(model_file)
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
        
        print(f"{i}. {filename}")
        print(f"   Size: {file_size:.2f} MB")
        
        # Try to load basic info from the model file
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            config = model_data['config']
            data_name = config.get('data_name', 'Unknown')
            interp_method = config.get('interp_method', 'Unknown')
            run_type = config.get('run_type', 'Unknown')
            
            print(f"   Dataset: {data_name}")
            print(f"   Method: {interp_method}")
            print(f"   Type: {run_type}")
            
            # Show parameter info
            params = model_data['params']
            if hasattr(params, 'shape'):
                print(f"   Parameters shape: {params.shape}")
            elif isinstance(params, list):
                print(f"   Parameters: list of {len(params)} arrays")
            else:
                print(f"   Parameters: {type(params)}")
                
        except Exception as e:
            print(f"   Error reading model info: {e}")
        
        print()

def main():
    """
    Main function to list saved models
    """
    print("Saved INN Models")
    print("=" * 80)
    list_saved_models()

if __name__ == "__main__":
    main() 