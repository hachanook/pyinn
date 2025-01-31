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

from .model import * ## when using pyinn
# from model import * ## when debugging

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
v_get_linspace = jax.vmap(get_linspace, in_axes=(0,0,None))

class Regression_INN:
    def __init__(self, cls_data, config):

        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = 3264

        self.nmode = int(config['MODEL_PARAM']['nmode'])
        self.nelem = int(config['MODEL_PARAM']['nelem'])
        self.nnode = self.nelem+1
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_INN'])
        
        ## initialization of trainable parameters
        if cls_data.bool_normalize: # when the data is normalized
            self.grid_dms = jnp.tile(jnp.linspace(0,1,self.nnode, dtype=jnp.float64), (self.cls_data.dim,1)) # (dim,nnode)
        else: # when the data is not normalized
            self.grid_dms = v_get_linspace(cls_data.x_data_minmax["min"], cls_data.x_data_minmax["max"], self.nnode)

        
        self.params = jax.random.uniform(jax.random.PRNGKey(self.key), (self.nmode, self.cls_data.dim, 
                                            self.cls_data.var, self.nnode), dtype=jnp.double)       
        numParam = self.nmode*self.cls_data.dim*self.cls_data.var*self.nnode

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
        
        if self.interp_method == "linear" or self.interp_method == "nonlinear":
            print(f"------------INN {config['TD_type']} {self.interp_method} -------------")
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


    def get_loss_r(self, grid_dms, params, x_data, u_data):
        ''' Compute MSE loss value for r-adaptivity
        --- input ---
        grid_dms: (dim, J)
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        # if self.interp_method == "linear":
        model = INN_linear(grid_dms, self.config)
            
        # elif self.interp_method == "nonlinear":
        #     model = INN_nonlinear(grid_dms, self.config)

        u_pred = model.v_forward(params, x_data) # (ndata_train, var)
        loss = ((u_pred- u_data)**2).mean()
        return loss, u_pred
    Grad_get_loss_r = jax.jit(jax.value_and_grad(get_loss_r, argnums=1, has_aux=True), static_argnames=['self'])
    

    @partial(jax.jit, static_argnames=['self']) 
    def update_optax(self, params, opt_state, x_data, u_data):
        ((loss, u_pred), grads) = self.Grad_get_loss(params, x_data, u_data)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, u_pred    
    
    def data_split(self):
        split_ratio = self.cls_data.split_ratio
        if self.config['DATA_PARAM']['bool_random_split'] == True and all(isinstance(item, float) for item in split_ratio):
            # random split with a split ratio
            generator = torch.Generator().manual_seed(42)
            split_data = random_split(dataset=self.cls_data,lengths=self.cls_data.split_ratio, generator=generator)
            if len(split_ratio) == 2:
                self.split_type="TT" # train and test
                train_data = split_data[0]
                test_data = split_data[1]
            elif len(split_ratio) == 3:
                self.split_type="TVT" # train, validation, test
                train_data = split_data[0]
                val_data = split_data[1]
                test_data = split_data[2]
                self.val_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
            
        elif self.config['DATA_PARAM']['bool_random_split'] == False and all(isinstance(item, int) for item in split_ratio):
            # non-random split with a fixed number of data
            if len(split_ratio) == 2:
                self.split_type="TT" # train and test
                train_data = Subset(self.cls_data, list(range(split_ratio[0])))
                test_data = Subset(self.cls_data, list(range(split_ratio[0], split_ratio[0]+split_ratio[1])))
            elif len(split_ratio) == 3:
                self.split_type="TVT" # train, validation, test
                train_data = Subset(self.cls_data, list(range(split_ratio[0])))
                test_data = Subset(self.cls_data, list(range(split_ratio[0], split_ratio[0]+split_ratio[1])))
                val_data = Subset(self.cls_data, list(range(split_ratio[0]+split_ratio[1], split_ratio[0]+split_ratio[1]+split_ratio[2])))
                self.val_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)

    def get_error_metrics(self, u, u_pred, error, ndata):
        """ Get sum of squared error
        """
        if jnp.isnan(u_pred).any(): # if the prediction has nan value,
            print(f"[Error] INN prediction has NaN components")
            print(jnp.where(jnp.isnan(u_pred))[0])        

        if self.bool_denormalize:
            u = self.cls_data.denormalize(u, self.cls_data.u_data_minmax)
            u_pred = self.cls_data.denormalize(u_pred, self.cls_data.u_data_minmax)
        error += jnp.sum((u - u_pred)**2)
        ndata += len(u)
        return error, ndata # sum of squared error

    def inference(self, x_test):
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            start_time_inference = time.time()
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")       

    def train(self):
        self.batch_size = int(self.config['TRAIN_PARAM']['batch_size'])
        self.learning_rate = float(self.config['TRAIN_PARAM']['learning_rate'])
        self.validation_period = int(self.config['TRAIN_PARAM']['validation_period'])
        self.bool_denormalize = self.config['TRAIN_PARAM']['bool_denormalize']
        self.error_type = self.config['TRAIN_PARAM']['error_type']
        self.patience = int(self.config['TRAIN_PARAM']['patience'])

        
        ## Split data and create dataloader
        self.data_split()
        
        ## Define optimizer
        params = self.params
        self.optimizer = optax.adam(self.learning_rate)
        opt_state = self.optimizer.init(params)
        
        ## Train
        start_time_train = time.time()
        errors_val = [] # validation error for every epoch
        for epoch in range(self.num_epochs):
            epoch_list_loss, epoch_list_acc = [], [] 
            start_time_epoch = time.time()    
            error, ndata = 0,0 # measure train error
            for batch in self.train_dataloader:
                
                x_train, u_train = jnp.array(batch[0]), jnp.array(batch[1])
                params, opt_state, loss_train, u_pred_train = self.update_optax(params, opt_state, x_train, u_train)
                error, ndata = self.get_error_metrics(u_train, u_pred_train, error, ndata)
                
            self.params = params
            print(f"Epoch {epoch+1}")
            if self.error_type == 'rmse':
                print(f"\tTrain RMSE: {jnp.sqrt(error/ndata):.4e}")
            elif self.error_type == 'mse':
                print(f"\tTrain MSE: {error/ndata:.4e}")
            elif self.error_type == 'accuracy':
                print(f"\tTrain Accuracy: {error/ndata*100:.2f}%")
            else:
                pass
            print(f"\tTrain took {time.time() - start_time_epoch:.4f} seconds")

            ## Validation
            if (epoch+1)%self.validation_period == 0:
                if self.split_type == "TT": # when there are only train & test data
                    self.val_dataloader = self.test_dataloader # deal test data as validation data
                error, ndata = 0,0
                for batch in self.val_dataloader:
                    x_val, u_val = jnp.array(batch[0]), jnp.array(batch[1])
                    _, _, loss_val, u_pred_val = self.update_optax(params, opt_state, x_val, u_val)
                    error, ndata = self.get_error_metrics(u_val, u_pred_val, error, ndata)
                
                if self.error_type == 'rmse':
                    print(f"\tVal RMSE: {jnp.sqrt(error/ndata):.4e}")
                    errors_val.append(jnp.sqrt(error/ndata))
                elif self.error_type == 'mse':
                    print(f"\tVal MSE: {error/ndata:.4e}")
                    errors_val.append(error/ndata)
                elif self.error_type == 'accuracy':
                    print(f"\tVal Accuracy: {error/ndata*100:.2f}%")
                    errors_val.append(error/ndata)
                else:
                    pass

            ## Check early stopping
            if len(errors_val) > self.patience:
                early_stopping = np.all(np.subtract(errors_val[-self.patience:], errors_val[-self.patience-1:-1])>0)
                if early_stopping: # break the epoch and finish training
                    print(f"\tEarly stopping at {epoch+1}-th epoch")
                    print(f"\tValidation losses of the latest epochs are {errors_val[-self.patience:]}")
                    break

        print(f"INN training took {time.time() - start_time_train:.4f} seconds")
        # if importlib.util.find_spec("GPUtil") is not None: # report GPU memory usage
        #     mem_report('After training', gpu_idx)

        ## Test 
        start_time_test = time.time()
        error, ndata = 0,0
        for batch in self.test_dataloader:
            x_test, u_test = jnp.array(batch[0]), jnp.array(batch[1])
            _, _, _, u_pred_test = self.update_optax(params, opt_state, x_test, u_test)
            error, ndata = self.get_error_metrics(u_test, u_pred_test, error, ndata)
        print("Test")
        if self.error_type == 'rmse':
            print(f"\tTest RMSE: {jnp.sqrt(error/ndata):.4e}")
        elif self.error_type == 'mse':
            print(f"\tTest MSE: {error/ndata:.4e}")
        elif self.error_type == 'accuracy':
            print(f"\tTest Accuracy: {error/ndata*100:.2f}%")
            
        print(f"\tTest took {time.time() - start_time_test:.4f} seconds") 

        ## Inference
        self.inference(x_test)

    def train_r(self):
        self.batch_size = int(self.config['TRAIN_PARAM']['batch_size']) * 100 # set r-batch as large as possible
        self.learning_rate = float(self.config['TRAIN_PARAM']['learning_rate'])
        self.validation_period = int(self.config['TRAIN_PARAM']['validation_period'])
        
        ## Split data and create dataloader
        self.data_split()
        
        ## Define variables
        grid_dms = self.grid_dms
        
        ## Train
        start_time_train = time.time()
        for batch in self.train_dataloader:
            # time_batch = time.time()
            x_train, u_train = jnp.array(batch[0]), jnp.array(batch[1])
            ((loss, u_pred), grads) = self.Grad_get_loss_r(grid_dms, self.params, x_train, u_train)

            ## Updated nodal coordinates - r-adaptivity
            grads = grads.at[:,[0,-1]].set(0) # fix boundary nodes
            elem_size = grid_dms[:,1:] - grid_dms[:,0:-1]  # (dim, J-1)
            elem_size_min = jnp.min(elem_size, axis=1) # (dim,) minimum element size for each dimension
            learning_rate = 0.05 * elem_size_min / jnp.max(jnp.abs(grads), axis=1)
            grid_dms -= learning_rate[:,None] * grads

        ## Save updated nodal coordinates 
        self.grid_dms = grid_dms
        # print(grid_dms)
        
        ## Re-define models
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




class Regression_MLP(Regression_INN):
    def __init__(self, cls_data, config):
        super().__init__(cls_data, config) # prob being dropout probability
        
        self.forward = forward_MLP
        self.v_forward = v_forward_MLP
        self.vv_forward = vv_forward_MLP
        self.nlayers = config['MODEL_PARAM']["nlayers"]
        self.nneurons = config['MODEL_PARAM']["nneurons"]
        self.activation = config['MODEL_PARAM']["activation"]
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_MLP'])

        ### initialization of trainable parameters
        layer_sizes = [cls_data.dim] + self.nlayers * [self.nneurons] + [cls_data.var]
        self.params = self.init_network_params(layer_sizes, jax.random.PRNGKey(self.key))
        weights, biases = 0,0
        for layer in self.params:
            w, b = layer[0], layer[1]
            weights += w.shape[0]*w.shape[1]
            biases += b.shape[0]
        print("------------MLP-------------")
        print(f"# of training parameters: {weights+biases}")

        
    def random_layer_params(self, m, n, key, scale=1e-2): # m input / n output neurons
      w_key, b_key = jax.random.split(key)
      return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(b_key, (n,))
    
    def init_network_params(self, sizes, key):
        keys = jax.random.split(key, len(sizes))
        return [self.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    
    @partial(jax.jit, static_argnames=['self']) # jit necessary
    def get_loss(self, params, x_data, u_data):
        ''' Compute loss value at (m)th mode given upto (m-1)th mode solution, which is u_pred_old
        --- input ---
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        
        u_pred = self.v_forward(params, self.activation, x_data) # (ndata_train, var)
        loss = ((u_pred- u_data)**2).mean()
        return loss, u_pred
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])

    def inference(self, x_test):
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
        start_time_inference = time.time()
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
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
    # Grad_get_loss = jax.value_and_grad(get_loss, argnums=1, has_aux=True)

    
    def get_error_metrics(self, u, u_pred, error, ndata):
        """ Get accuracy metrics. For regression, either RMSE or MSE will be returned.
            This function cannot be jitted because it uses scipy library
        --- input ---
        u: (ndata, nclass) integer vector that indicates class of the data
        u_train: (ndata, nclass) integer vector that indicates predicted class
        """
        u_single = jnp.argmax(u, axis=1)
        u_pred_single = jnp.argmax(u_pred, axis=1)
        report = classification_report(np.array(u_single), np.array(u_pred_single), output_dict=True, zero_division=1)
        error = report["accuracy"]
        ndata += len(u)
        return error*ndata, ndata


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
        
        u_pred = self.v_forward(params, self.activation, x_data) # (ndata_train, var)
        prediction = u_pred - jax.scipy.special.logsumexp(u_pred, axis=1)[:,None] # (ndata, var = nclass)
        loss = -jnp.mean(jnp.sum(prediction * u_data, axis=1))
        return loss, u_pred
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])
    # Grad_get_loss = jax.value_and_grad(get_loss, argnums=1, has_aux=True)

    def get_error_metrics(self, u, u_pred, error, ndata):
        """ Get accuracy metrics. For regression, either RMSE or MSE will be returned.
            This function cannot be jitted because it uses scipy library
        --- input ---
        u: (ndata, nclass) integer vector that indicates class of the data
        u_train: (ndata, nclass) integer vector that indicates predicted class
        """
        u_single = jnp.argmax(u, axis=1)
        u_pred_single = jnp.argmax(u_pred, axis=1)
        report = classification_report(np.array(u_single), np.array(u_pred_single), output_dict=True, zero_division=1)
        error = report["accuracy"]
        ndata += len(u)
        return error*ndata, ndata