"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import optax
from functools import partial
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
import time
import torch
from sklearn.metrics import r2_score, classification_report
import importlib.util
import sys

# from .model import * ## when using pyinn
from model import * ## when debugging
# from pyinn.model import * ## when debugging on streamlit

if importlib.util.find_spec("GPUtil") is not None: # for linux & GPU
    ''' If you are funning on GPU, please install the following libraries on your anaconda environment via 
    $ conda install -c conda-forge humanize
    $ conda install -c conda-forge psutil
    $ conda install -c conda-forge gputil
    ''' 
    import humanize, psutil, GPUtil
    
    # memory report
    def mem_report(num, gpu_idx):
        ''' This function reports memory usage for both CPU and GPU'''
        print(f"-{num}-CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
        
        GPUs = GPUtil.getGPUs()
        gpu = GPUs[gpu_idx]
        # for i, gpu in enumerate(GPUs):
        print('---GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%\n'.format(gpu_idx, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

def get_linspace(xmin, xmax, nnode):
    return jnp.linspace(xmin,xmax,nnode, dtype=jnp.float64)
# v_get_linspace = jax.vmap(get_linspace, in_axes=(0,0,None))

class Regression_INN:
    def __init__(self, cls_data, config):

        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = int(time.time())
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_INN'])
        # self.scale = int(config['MODEL_PARAM']['nmode'])/100
        self.scale = 1
        
        ## Initialize trainable parameters for INN.
        if 'linear' in self.interp_method: # for INN
            self.nmode = int(config['MODEL_PARAM']['nmode'])
            if isinstance(config['MODEL_PARAM']['nseg'], int): # same discretization across dimension
                
                self.nseg = int(config['MODEL_PARAM']['nseg'])
                self.nnode = self.nseg + 1
                
                
                
                ## initialization of trainable parameters
                if cls_data.bool_normalize: # when the data is normalized
                    self.grid_dms = jnp.linspace(0, 1, self.nnode, dtype=jnp.float64) # (nnode,) the most efficient way
                else: # when the data is not normalized
                    self.grid_dms = [get_linspace(xmin, xmax, self.nnode) for (xmin, xmax) in zip(cls_data.x_data_minmax["min"], cls_data.x_data_minmax["max"])]
                self.params = jax.random.uniform(jax.random.PRNGKey(self.key), (self.nmode, self.cls_data.dim, 
                                                    self.cls_data.var, self.nnode), dtype=jnp.double) / self.scale
                numParam = self.nmode*self.cls_data.dim*self.cls_data.var*self.nnode
            
            elif isinstance(config['MODEL_PARAM']['nseg'], list): # varying discretization across dimension

                self.nseg = jnp.array(config['MODEL_PARAM']['nseg'], dtype=jnp.int64) # (dim,) 1D array of integers
                self.nnode = self.nseg + 1

                if len(self.nseg) != cls_data.dim:
                    print(f"Error: lenth of nseg {len(self.nseg)} is different from input dimension {cls_data.dim}. Check config file.")
                    sys.exit()

                ## initialization of trainable parameters
                self.grid_dms, self.params, numParam = [], [], 0
                for idm, nnode_idm in enumerate(self.nnode):
                    if cls_data.bool_normalize: # when the data is normalized
                        self.grid_dms.append(jnp.linspace(0, 1, nnode_idm, dtype=jnp.float64))
                    else: # when the data is not normalized
                        self.grid_dms.append(get_linspace(cls_data.x_data_minmax["min"][idm], cls_data.x_data_minmax["max"][idm], nnode_idm))
                    self.params.append(jax.random.uniform(jax.random.PRNGKey(self.key), (self.nmode, self.cls_data.var, nnode_idm), dtype=jnp.double) / self.scale)
                    numParam += self.nmode*self.cls_data.var*nnode_idm 

        ## Define model
        if self.interp_method == "linear":
            model = INN_linear(self.grid_dms, self.config)
            self.forward = model.forward
            self.v_forward = model.v_forward
            self.vv_forward = model.vv_forward
            
        elif self.interp_method == "nonlinear":
            model = INN_nonlinear(self.grid_dms, self.config)
            self.forward = model.forward
            self.v_forward = model.v_forward
            self.vv_forward = model.vv_forward

        # ## debug
        # x_idata = jnp.ones(self.cls_data.dim, dtype=jnp.float64) * 0.5
        # x_data = jnp.ones((2, self.cls_data.dim), dtype=jnp.float64) * 0.5
        # a = self.forward(self.params, x_idata) # (var,)
        # va = self.v_forward(self.params, x_data) # (ndata, var)
        
        if self.interp_method == "linear":
            print(f"------------ INN {config['TD_type']} {self.interp_method}, nmode: {config['MODEL_PARAM']['nmode']}, nseg: {config['MODEL_PARAM']['nseg']} -------------")
            print(f"# of training parameters: {numParam}")
        elif self.interp_method == "nonlinear":
            print(f"------------ INN {config['TD_type']} {self.interp_method}, nmode: {config['MODEL_PARAM']['nmode']}, nseg: {config['MODEL_PARAM']['nseg']}, s={config['MODEL_PARAM']['s_patch']}, P={config['MODEL_PARAM']['p_order']} -------------")
            print(f"# of training parameters: {numParam}")


    @partial(jax.jit, static_argnames=['self']) # jit necessary
    def get_loss(self, params, x_data, u_data):
        ''' Compute MSE loss value at (m)th mode given upto (m-1)th mode solution, which is u_pred_old
        --- input ---
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        u_pred = self.v_forward(params, x_data) # (ndata_train, var)
        loss = ((u_pred- u_data)**2).mean()
        return loss, u_pred
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])


    @partial(jax.jit, static_argnames=['self']) 
    def update_optax(self, params, opt_state, x_data, u_data):
        ((loss, u_pred), grads) = self.Grad_get_loss(params, x_data, u_data)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, u_pred    
    

    def get_error_metrics(self, u, u_pred, error_cum, ndata_cum):
        """ Get sum of squared error
        u: (batch_size, var)
        """
        if jnp.isnan(u_pred).any(): # if the prediction has nan value,
            print(f"[Error] INN prediction has NaN components")
            print(jnp.where(jnp.isnan(u_pred))[0])        

        if self.bool_denormalize:
            u = self.cls_data.denormalize(u, self.cls_data.u_data_minmax)
            u_pred = self.cls_data.denormalize(u_pred, self.cls_data.u_data_minmax)
        error_cum += jnp.sum((u - u_pred)**2)
        ndata_cum += u.shape[0]*u.shape[1]
        # ndata_cum += len(u)
        return error_cum, ndata_cum # sum of squared error

    def inference(self, x_test):
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            start_time_inference = time.time()
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")       

    def train(self):
        # self.batch_size = int(self.config['TRAIN_PARAM']['batch_size'])
        self.learning_rate = float(self.config['TRAIN_PARAM']['learning_rate'])
        self.validation_period = int(self.config['TRAIN_PARAM']['validation_period'])
        self.bool_denormalize = self.config['TRAIN_PARAM']['bool_denormalize']
        self.error_type = self.config['TRAIN_PARAM']['error_type']
        self.patience = int(self.config['TRAIN_PARAM']['patience'])


        ## Define optimizer
        params = self.params
        self.optimizer = optax.adam(self.learning_rate)
        opt_state = self.optimizer.init(params)
        
        ## Train
        start_time_train = time.time()
        self.errors_train = [] # training error for every epoch 
        self.errors_val = [] # validation error for every epoch
        self.errors_epoch = [] # epochs where the erros are stored
        time_per_epochs = [] # training time per epoch
        for epoch in range(self.num_epochs):
            epoch_list_loss, epoch_list_acc = [], [] 
            start_time_epoch = time.time()    

            error_cum, ndata_cum = 0,0 # measure train error
            for batch in self.cls_data.train_dataloader:
                
                x_train, u_train = jnp.array(batch[0]), jnp.array(batch[1])
                params, opt_state, loss_train, u_pred_train = self.update_optax(params, opt_state, x_train, u_train)
                error_cum, ndata_cum = self.get_error_metrics(u_train, u_pred_train, error_cum, ndata_cum)
            
            ## measure train error
            if self.error_type == 'rmse':
                err_train = jnp.sqrt(error_cum/ndata_cum)
            elif self.error_type == 'mse':
                err_train = error_cum/ndata_cum
            elif self.error_type == 'accuracy':
                err_train = error_cum/ndata_cum*100
            else:
                pass

            if (epoch+1)%1 == 0:
                print(f"Epoch {epoch+1}")
                print(f"\tTrain {self.error_type}: {err_train:.4e}")
                

            ## Validation
            if (epoch+1)%self.validation_period == 0:
                self.errors_train.append(err_train)
                self.errors_epoch.append(epoch+1)
                error_cum, ndata_cum = 0,0
                for batch in self.cls_data.val_dataloader:
                    x_val, u_val = jnp.array(batch[0]), jnp.array(batch[1])
                    _, _, loss_val, u_pred_val = self.update_optax(params, opt_state, x_val, u_val)
                    error_cum, ndata_cum = self.get_error_metrics(u_val, u_pred_val, error_cum, ndata_cum)
                
                if self.error_type == 'rmse':
                    print(f"\tVal RMSE: {jnp.sqrt(error_cum/ndata_cum):.4e}")
                    self.errors_val.append(jnp.sqrt(error_cum/ndata_cum))
                elif self.error_type == 'mse':
                    print(f"\tVal MSE: {error_cum/ndata_cum:.4e}")
                    self.errors_val.append(error_cum/ndata_cum)
                elif self.error_type == 'accuracy':
                    print(f"\tVal Accuracy: {error_cum/ndata_cum*100:.2f}%")
                    self.errors_val.append(error_cum/ndata_cum)
                else:
                    pass
            time_per_epochs.append(time.time() - start_time_epoch) # append training time per epoch
            print(f"\t{np.mean(time_per_epochs):.2f} seconds per epoch")
                

            ## Check early stopping
            if len(self.errors_val) > self.patience:
                if self.error_type == 'rmse' or self.error_type == 'mse':
                    early_stopping = np.all(np.subtract(self.errors_val[-self.patience:], self.errors_val[-self.patience-1:-1])>0)
                elif self.error_type == 'accuracy':
                    # for accuracy, we want to stop when the validation accuracy is not improving
                    early_stopping = np.all(np.subtract(self.errors_val[-self.patience:], self.errors_val[-self.patience-1:-1])<0)
                
                if early_stopping: # break the epoch and finish training
                    print(f"\tEarly stopping at {epoch+1}-th epoch")
                    print(f"\tValidation losses of the latest epochs are {self.errors_val[-self.patience:]}")
                    break
            if "stopping_loss_train" in self.config["TRAIN_PARAM"].keys():
                if err_train < float(self.config["TRAIN_PARAM"]["stopping_loss_train"]):
                    break

        print(f"Training took {time.time() - start_time_train:.4f} seconds/ {np.mean(time_per_epochs):.2f} seconds per epoch")
        # if importlib.util.find_spec("GPUtil") is not None: # report GPU memory usage
        #     mem_report('After training', gpu_idx)
        self.params = params

        ## Test 
        start_time_test = time.time()
        error_cum, ndata_cum = 0,0
        for batch in self.cls_data.test_dataloader:
            x_test, u_test = jnp.array(batch[0]), jnp.array(batch[1])
            _, _, _, u_pred_test = self.update_optax(params, opt_state, x_test, u_test)
            error_cum, ndata_cum = self.get_error_metrics(u_test, u_pred_test, error_cum, ndata_cum)
        print("Test")
        if self.error_type == 'rmse':
            error_test = jnp.sqrt(error_cum/ndata_cum)
            print(f"\tTest RMSE: {error_test:.4e}")
        elif self.error_type == 'mse':
            error_test = error_cum/ndata_cum
            print(f"\tTest MSE: {error_test:.4e}")
        elif self.error_type == 'accuracy':
            error_test = error_cum/ndata_cum*100
            print(f"\tTest Accuracy: {error_test:.2f}%")
        self.error_test = error_test
            
        print(f"\tTest took {time.time() - start_time_test:.4f} seconds") 

        ## Inference
        self.inference(x_test)

class Regression_INN_sequential(Regression_INN):
    def __init__(self, cls_data, config, params_prev):
        super().__init__(cls_data, config) # prob being dropout probability

        self.params_prev = params_prev # trained parameters from previous sequences
        
        ## set the params to be zero WE SHOULD NOT MAKE THIS ZERO BECUASE TD WILL MULTIPLY WITH ZEROS
        
        if isinstance(self.params, list):
            params = self.params # list
            scale = 0.1**(1/len(self.params)) # 1/dim
            self.params = [param*scale for param in params]
        else:
            scale = 0.1**(1/self.params.shape[1]) # 1/dim
            self.params = self.params*scale

    @partial(jax.jit, static_argnames=['self']) # jit necessary
    def get_loss(self, params, x_data, u_data):
        ''' Compute MSE loss value at (m)th mode given upto (m-1)th mode solution, which is u_pred_old
        --- input ---
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        ## augment params from previous sequence
        params = jnp.concatenate([self.params_prev, params], axis=0)

        u_pred = self.v_forward(params, x_data) # (ndata_train, var)
        loss = ((u_pred- u_data)**2).mean()
        return loss, u_pred
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])


    def inference(self, x_test):
        
        ## augment params from previous sequence
        params = jnp.concatenate([self.params_prev, self.params], axis=0)

        u_pred = self.forward(params, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(params, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(params, x_test[0]) # (ndata_train, var)
        start_time_inference = time.time()
        u_pred = self.forward(params, x_test[0]) # (ndata_train, var)
        print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")      


        



        


class Regression_MLP(Regression_INN):
    def __init__(self, cls_data, config):
        super().__init__(cls_data, config) # prob being dropout probability
        
        activation = config['MODEL_PARAM']["activation"]
        model = MLP(activation)
        self.forward = model.forward
        self.v_forward = model.v_forward
        self.vv_forward = model.vv_forward
        self.nlayers = config['MODEL_PARAM']["nlayers"]
        self.nneurons = config['MODEL_PARAM']["nneurons"]
        
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_MLP'])

        ### initialization of trainable parameters
        layer_sizes = [cls_data.dim] + self.nlayers * [self.nneurons] + [cls_data.var]
        self.params = self.init_network_params(layer_sizes, jax.random.PRNGKey(self.key))
        weights, biases = 0,0
        for layer in self.params:
            w, b = layer[0], layer[1]
            weights += w.shape[0]*w.shape[1]
            biases += b.shape[0]
        print(f"------------ MLP, {layer_sizes} -------------")
        print(f"# of training parameters: {weights+biases}")

    
    # def random_layer_params(self, m, n, key, scale=1e-2): # m input / n output neurons
    #   w_key, b_key = jax.random.split(key)
    #   return scale * jax.random.normal(w_key, (m, n)), scale * jax.random.normal(b_key, (n,))
    
    # def init_network_params(self, sizes, key):
    #     # original method
    #     keys = jax.random.split(key, len(sizes))
    #     return [self.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    def init_network_params(self, layer_sizes, key):
        # what ChatGPT suggested
        keys = jax.random.split(key, len(layer_sizes) - 1)
        params = []
        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
            weight_key, bias_key = jax.random.split(k)
            W = jax.random.normal(weight_key, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            b = jnp.zeros(out_dim)
            params.append((W, b))
        return params
    
    @partial(jax.jit, static_argnames=['self']) # jit necessary
    def get_loss(self, params, x_data, u_data):
        ''' Compute loss value at (m)th mode given upto (m-1)th mode solution, which is u_pred_old
        --- input ---
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        
        u_pred = self.v_forward(params, x_data) # (ndata_train, var)
        loss = ((u_pred- u_data)**2).mean()
        return loss, u_pred
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])

    def inference(self, x_test):
        u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
        start_time_inference = time.time()
        u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
        print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")    
    

class Regression_KAN(Regression_INN):
    def __init__(self, cls_data, config):
        """Initialize KAN trainer for regression."""
        # Don't call super().__init__ because we need custom initialization
        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = int(time.time())
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_KAN'])

        # KAN-specific parameters
        self.nlayers = config['MODEL_PARAM']['nlayers']
        self.hidden_dim = config['MODEL_PARAM']['hidden_dim']
        self.grid_size = config['MODEL_PARAM']['grid_size']
        self.spline_order = config['MODEL_PARAM']['spline_order']

        # Create KAN model
        layer_sizes = [cls_data.dim] + self.nlayers * [self.hidden_dim] + [cls_data.var]
        model = KAN(layer_sizes, self.grid_size, self.spline_order)
        self.forward = model.forward
        self.v_forward = model.v_forward
        self.vv_forward = model.vv_forward

        # Initialize parameters
        self.params = self.init_kan_params(layer_sizes, model.grid_size, model.spline_order, jax.random.PRNGKey(self.key))

        # Count parameters
        num_params = 0
        for spline_params, base_weights in self.params:
            num_params += spline_params.size + base_weights.size

        print(f"------------ KAN, {layer_sizes}, grid_size={self.grid_size}, spline_order={self.spline_order} -------------")
        print(f"# of training parameters: {num_params}")

    def init_kan_params(self, layer_sizes, grid_size, spline_order, key):
        """
        Initialize KAN parameters.

        Returns:
            List of (spline_params, base_weights) tuples for each layer
        """
        # Use grid_size directly (RBF basis count)
        num_basis = grid_size

        keys = jax.random.split(key, len(layer_sizes) - 1)
        params = []

        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
            spline_key, weight_key = jax.random.split(k)

            # Initialize basis coefficients (in_features, out_features, num_basis)
            spline_params = jax.random.normal(spline_key, (in_dim, out_dim, num_basis), dtype=jnp.float64) * 0.1

            # Initialize base weights for residual connection (in_features, out_features)
            base_weights = jax.random.normal(weight_key, (in_dim, out_dim), dtype=jnp.float64) * jnp.sqrt(1 / in_dim)

            params.append((spline_params, base_weights))

        return params

    @partial(jax.jit, static_argnames=['self'])
    def get_loss(self, params, x_data, u_data):
        """Compute MSE loss for KAN."""
        u_pred = self.v_forward(params, x_data)  # (ndata_train, var)
        loss = ((u_pred - u_data) ** 2).mean()
        return loss, u_pred

    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])

    def inference(self, x_test):
        """Inference timing for KAN."""
        # Warm-up
        u_pred = self.forward(self.params, x_test[0])
        u_pred = self.forward(self.params, x_test[0])
        u_pred = self.forward(self.params, x_test[0])

        # Timed inference
        start_time_inference = time.time()
        u_pred = self.forward(self.params, x_test[0])
        print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")


class Classification_INN(Regression_INN):
    def __init__(self, cls_data, config):
        super().__init__(cls_data, config) # prob being dropout probability

        ## classification problem always normalize inputs between 0 and 1
        self.x_dms_nds = jnp.tile(jnp.linspace(0,1,self.nnode, dtype=jnp.float64), (self.cls_data.dim,1)) # (dim,nnode)

        ## initialization of trainable parameters
        self.params = jax.random.uniform(jax.random.PRNGKey(self.key), (self.nmode, self.cls_data.dim,
                                                self.cls_data.var, self.nnode), dtype=jnp.double,
                                                minval=0.98, maxval=1.02) # for classification, we sould confine the params range
        # numParam = self.nmode*self.cls_data.dim*self.cls_data.var*self.nnode
        
        

    @partial(jax.jit, static_argnames=['self']) # jit necessary
    def get_loss(self, params, x_data, u_data):
        ''' Compute Cross Entropy loss value at (m)th mode given upto (m-1)th mode solution, which is u_pred_old
        --- input ---
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        
        u_pred = self.v_forward(params, x_data) # (ndata_train, var)
        prediction = u_pred - jax.scipy.special.logsumexp(u_pred, axis=1)[:,None] # (ndata, var = nclass)
        loss = -jnp.mean(jnp.sum(prediction * u_data, axis=1))
        return loss, u_pred
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])
 

    def get_error_metrics(self, u, u_pred, error_cum, ndata_cum):
        """ Get accuracy metrics for classification.
            This function cannot be jitted because it uses scipy library
        --- input ---
        u: (ndata, nclass) integer vector that indicates class of the data
        u_train: (ndata, nclass) integer vector that indicates predicted class
        """
        u_single = jnp.argmax(u, axis=1)
        u_pred_single = jnp.argmax(u_pred, axis=1)
        report = classification_report(np.array(u_single), np.array(u_pred_single), output_dict=True, zero_division=1)
        error = report["accuracy"]
        error_cum += error*len(u)
        ndata_cum += len(u)
        return error_cum, ndata_cum


class Regression_FNO(Regression_INN):
    def __init__(self, cls_data, config):
        """Initialize FNO trainer for regression."""
        # Custom initialization for FNO
        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = int(time.time())
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_FNO'])

        # FNO-specific parameters
        self.hidden_dim = config['MODEL_PARAM']['hidden_dim']
        self.num_layers = config['MODEL_PARAM']['num_layers']
        self.modes = config['MODEL_PARAM']['modes']

        # Create FNO model
        model = FNO(cls_data.dim, cls_data.var, self.hidden_dim, self.num_layers, self.modes)
        self.forward = model.forward
        self.v_forward = model.v_forward
        self.vv_forward = model.vv_forward

        # Initialize parameters
        self.params = self.init_fno_params(cls_data.dim, cls_data.var, self.hidden_dim,
                                            self.num_layers, self.modes, jax.random.PRNGKey(self.key))

        # Count parameters
        num_params = 0
        num_params += self.params['lift'].size + self.params['lift_bias'].size
        for spectral_w, linear_w, b in self.params['fourier_layers']:
            num_params += spectral_w.size + linear_w.size + b.size
        num_params += self.params['project'].size + self.params['project_bias'].size

        print(f"------------ FNO, hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, modes={self.modes} -------------")
        print(f"# of training parameters: {num_params}")

    def init_fno_params(self, input_dim, output_dim, hidden_dim, num_layers, modes, key):
        """
        Initialize FNO parameters.

        Returns:
            Dictionary with lifting, Fourier layers, and projection parameters
        """
        keys = jax.random.split(key, 2 + num_layers * 3)
        key_idx = 0

        # Lifting layer: input_dim -> hidden_dim
        lift = jax.random.normal(keys[key_idx], (input_dim, hidden_dim), dtype=jnp.float64) * jnp.sqrt(2 / input_dim)
        key_idx += 1
        lift_bias = jnp.zeros(hidden_dim, dtype=jnp.float64)

        # Fourier layers
        fourier_layers = []
        for _ in range(num_layers):
            # Spectral weights (complex-valued for Fourier domain)
            # Simplified: (modes, hidden_dim) instead of (modes, hidden_dim, hidden_dim)
            spectral_real = jax.random.normal(keys[key_idx], (modes, hidden_dim), dtype=jnp.float64) * 0.02
            key_idx += 1
            spectral_imag = jax.random.normal(keys[key_idx], (modes, hidden_dim), dtype=jnp.float64) * 0.02
            key_idx += 1
            spectral_weights = spectral_real + 1j * spectral_imag

            # Linear weights (physical space)
            linear_weights = jax.random.normal(keys[key_idx], (hidden_dim, hidden_dim), dtype=jnp.float64) * jnp.sqrt(2 / hidden_dim)
            key_idx += 1

            # Bias
            bias = jnp.zeros(hidden_dim, dtype=jnp.float64)

            fourier_layers.append((spectral_weights, linear_weights, bias))

        # Projection layer: hidden_dim -> output_dim
        project = jax.random.normal(keys[key_idx], (hidden_dim, output_dim), dtype=jnp.float64) * jnp.sqrt(2 / hidden_dim)
        key_idx += 1
        project_bias = jnp.zeros(output_dim, dtype=jnp.float64)

        params = {
            'lift': lift,
            'lift_bias': lift_bias,
            'fourier_layers': fourier_layers,
            'project': project,
            'project_bias': project_bias
        }

        return params

    @partial(jax.jit, static_argnames=['self'])
    def get_loss(self, params, x_data, u_data):
        """Compute MSE loss for FNO."""
        u_pred = self.v_forward(params, x_data)  # (ndata_train, var)
        loss = ((u_pred - u_data) ** 2).mean()
        return loss, u_pred

    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])

    def inference(self, x_test):
        """Inference timing for FNO."""
        # Warm-up
        u_pred = self.forward(self.params, x_test[0])
        u_pred = self.forward(self.params, x_test[0])
        u_pred = self.forward(self.params, x_test[0])

        # Timed inference
        start_time_inference = time.time()
        u_pred = self.forward(self.params, x_test[0])
        print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")


class Classification_MLP(Regression_MLP):

    def __init__(self, cls_data, config):
        super().__init__(cls_data, config) # prob being dropout probability

    @partial(jax.jit, static_argnames=['self']) # jit necessary
    def get_loss(self, params, x_data, u_data):
        ''' Compute Cross Entropy loss value at (m)th mode given upto (m-1)th mode solution, which is u_pred_old
        --- input ---
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        
        u_pred = self.v_forward(params, x_data) # (ndata_train, var)
        prediction = u_pred - jax.scipy.special.logsumexp(u_pred, axis=1)[:,None] # (ndata, var = nclass)
        loss = -jnp.mean(jnp.sum(prediction * u_data, axis=1))
        return loss, u_pred
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])
    # Grad_get_loss = jax.value_and_grad(get_loss, argnums=1, has_aux=True)

    def get_error_metrics(self, u, u_pred, error_cum, ndata_cum):
        """ Get accuracy metrics for classification.
            This function cannot be jitted because it uses scipy library
        --- input ---
        u: (ndata, nclass) integer vector that indicates class of the data
        u_train: (ndata, nclass) integer vector that indicates predicted class
        """
        u_single = jnp.argmax(u, axis=1)
        u_pred_single = jnp.argmax(u_pred, axis=1)
        report = classification_report(np.array(u_single), np.array(u_pred_single), output_dict=True, zero_division=1)
        error = report["accuracy"]
        error_cum += error*len(u)
        ndata_cum += len(u)
        return error_cum, ndata_cum