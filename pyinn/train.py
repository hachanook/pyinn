"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax
import jax.numpy as jnp
from jax import config
import numpy as np
config.update("jax_enable_x64", True)
import optax
from functools import partial
from torch.utils.data import DataLoader, random_split, Subset
import time
import torch
from sklearn.metrics import r2_score, classification_report
import importlib.util
import pandas as pd
# from .model import * ## when using pyinn
from model import * ## when debugging
from tqdm import tqdm

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
            self.x_dms_nds = jnp.tile(jnp.linspace(0,1,self.nnode, dtype=jnp.float64), (self.cls_data.dim,1)) # (dim,nnode)
        else: # when the data is not normalized
            self.x_dms_nds = v_get_linspace(cls_data.x_data_minmax["min"], cls_data.x_data_minmax["max"], self.nnode)

        if self.interp_method == "linear":
            model = INN_linear(self.x_dms_nds[0,:], config)
            self.forward = model.forward
            self.v_forward = model.v_forward
            self.vv_forward = model.vv_forward
        elif self.interp_method == "nonlinear":
            model = INN_nonlinear(self.x_dms_nds[0,:], config)
            self.forward = model.forward
            self.v_forward = model.v_forward
            self.vv_forward = model.vv_forward
        
        
        self.params = jax.random.uniform(jax.random.PRNGKey(self.key), (self.nmode, self.cls_data.dim, 
                                            self.cls_data.var, self.nnode), dtype=jnp.double)       
        numParam = self.nmode*self.cls_data.dim*self.cls_data.var*self.nnode

        if config['TD_type'] == 'Tucker':
            # Create a grid of indices for the tensor
            indices = jnp.arange(self.nmode)

            # Create a tensor of zeros
            shape = (self.nmode,) * self.cls_data.dim
            indices = jnp.arange(self.nmode)

            # Use a boolean mask to set diagonal elements
            core = jnp.zeros(shape, dtype=jnp.float64)
            core = core.at[tuple([indices] * self.cls_data.dim)].set(1.0)
            self.params = [core, self.params] # core tensor and factor matrices
            numParam += len(self.params[0].reshape(-1))
        
        
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
    
    @partial(jax.jit, static_argnames=['self']) 
    def update_optax(self, params, opt_state, x_data, u_data):
        ((loss, u_pred), grads) = self.Grad_get_loss(params, x_data, u_data)
        # print(f"\tgrad shape: {jax.tree_map(lambda x: x.shape, grads)}")
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, u_pred    
    
    def data_split(self):
        split_ratio = self.cls_data.split_ratio
        if self.config['DATA_PARAM']['bool_load_data'] == True:
            self.split_type = 'TVT' # test as validation data
            self.train_dataloader = DataLoader(self.cls_data.pre_split_train_data, batch_size=self.batch_size, shuffle=True)
            self.test_dataloader = DataLoader(self.cls_data.pre_split_test_data, batch_size=self.batch_size, shuffle=True)
            self.val_dataloader = DataLoader(self.cls_data.pre_split_val_data, batch_size=self.batch_size, shuffle=True)
        elif self.config['DATA_PARAM']['bool_random_split'] == True and all(isinstance(item, float) for item in split_ratio):
            # random split with a split ratio
            generator = torch.Generator().manual_seed(42)
            retained_data, _ = random_split(dataset=self.cls_data, lengths=[0.8, 0.2], generator=generator)
            # Lists to store features and labels
            features_list = []
            labels_list = []

            # Iterate over the retained_data
            for data_item in retained_data:
                # Split data_item into features and labels
                features, label = data_item 
                
                # If features is a tensor, convert to list or numpy array
                features_list.append(features.tolist())
                
                # If label is a tensor, convert to scalar (assuming it's a single label per item)
                labels_list.append(label.item()) 

            split_data = random_split(dataset=retained_data, lengths=self.cls_data.split_ratio, generator=generator)
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

            self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
                        
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

    def get_acc_metrics(self, u, u_pred, type0="test"):
        """ Get accuracy metrics. For regression, R2 will be returned.
        """
        if jnp.isnan(u_pred).any(): # if the prediction has nan value,
            print(f"[Error] INN prediction has NaN components")
            print(jnp.where(jnp.isnan(u_pred))[0])        

        if type0 == "train":
            bool_train_acc = self.config['TRAIN_PARAM']['bool_train_acc']
            if bool_train_acc:
                acc_metrics = "R2"
                acc = r2_score(np.array(u), np.array(u_pred))
            else:
                acc, acc_metrics = 0,"R2"
                
        elif type0 == "val" or type0 == "test":
            acc_metrics = "R2"
            acc = r2_score(np.array(u), np.array(u_pred))
        return acc, acc_metrics

    def inference(self, x_test):
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            start_time_inference = time.time()
            u_pred = self.forward(self.params, x_test[0]) # (ndata_train, var)
            print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")       

    def train(self):
        self.batch_size = int(self.config['TRAIN_PARAM']['batch_size'])
        self.learning_rate = float(self.config['TRAIN_PARAM']['learning_rate'])
        self.validation_period = int(self.config['TRAIN_PARAM']['validation_period'])
        
        ## Split data and create dataloader
        self.data_split()
        
        ## Define optimizer
        params = self.params
        self.optimizer = optax.adam(self.learning_rate)
        opt_state = self.optimizer.init(params)
        early_stopping = False
        
        # loss_train_list, loss_test_list = [], []
        # acc_train_list, acc_test_list = [], []
        # if self.split_type == "TVT":
        #     loss_val_list, acc_val_list = [], []
        
        ## Train
        total_epochs = self.num_epochs  # Assuming self.num_epochs is defined
        interval = total_epochs // 10  # This calculates 10% of the total epochs
        start_time_train = time.time()
        rmses_val = [] # validation loss for every epoch
        for epoch in tqdm(range(self.num_epochs), desc='Epochs'):
            
            ## Training one epoch
            losses_train_epoch = [] # for one epoch, list of losses from all batches
            for idx, batch in enumerate(self.train_dataloader):
                x_train, u_train = jnp.array(batch[0]), jnp.array(batch[1])
                params, opt_state, loss_train, u_pred_train = self.update_optax(params, opt_state, x_train, u_train)
            #     losses_train_epoch.append(jnp.sqrt(loss_train))  #root mean square error in normalized space
            # loss_train_epoch = np.mean(losses_train_epoch) #root mean square error
            # losses_train_epoch.append(loss_train_epoch)
            self.params = params # save trained parameters


            ## Check validation set in normalized scale
            rmse_val, n_samples = 0,0
            for batch in self.val_dataloader:
                x_val, u_val = jnp.array(batch[0]), jnp.array(batch[1])
                _, _, loss_val, u_pred_val = self.update_optax(params, opt_state, x_val, u_val)
                rmse_val += loss_val * len(x_val) # sum of squared error
                n_samples += len(x_val)
            rmse_val = jnp.sqrt(rmse_val/n_samples)
            rmses_val.append(rmse_val)
            
            ## Check early stopping
            patience = int(self.config['TRAIN_PARAM']['patience'])
            if len(rmses_val) > patience:
                early_stopping = np.all(np.subtract(rmses_val[-patience:], rmses_val[-patience-1:-1])>0)

            ## Print out training results
            if (early_stopping or (epoch % 200 == 0 or epoch == self.num_epochs-1)):
                print(f"\tEpoch {epoch+1}")
                
                ## Train error in original scale
                rmse_train, n_samples = 0, 0
                for batch in self.train_dataloader:
                    x_train, u_train = jnp.array(batch[0]), jnp.array(batch[1])
                    _, _, _, u_pred_train = self.update_optax(params, opt_state, x_train, u_train)
                    u_train_org = self.cls_data.denormalize(u_train, self.cls_data.u_data_minmax)
                    u_pred_train_org = self.cls_data.denormalize(u_pred_train, self.cls_data.u_data_minmax)
                    rmse_train += jnp.sum((u_train_org-u_pred_train_org)**2) # sum of squared errors
                    n_samples += len(x_train)
                rmse_train = jnp.sqrt(rmse_train/n_samples)
                print(f"\tTrain denormalized loss (RMSE): {rmse_train:.4e}")

                ## Test error in original scale
                rmse_test, n_samples = 0, 0
                for batch in self.test_dataloader:
                    x_test, u_test = jnp.array(batch[0]), jnp.array(batch[1])
                    _, _, _, u_pred_test = self.update_optax(params, opt_state, x_test, u_test)
                    u_test_org = self.cls_data.denormalize(u_test, self.cls_data.u_data_minmax)
                    u_pred_test_org = self.cls_data.denormalize(u_pred_test, self.cls_data.u_data_minmax)
                    rmse_test += jnp.sum((u_test_org-u_pred_test_org)**2) # sum of squared errors
                    n_samples += len(x_test)
                rmse_test = jnp.sqrt(rmse_test/n_samples)
                print(f"\tTest denormalized loss (RMSE): {rmse_test:.4e}")

                ## Early stopping
                if early_stopping:
                    print(f"\tEarly stopping at {epoch+1}-th epoch")
                    print(f"\tValidation losses of the latest epochs are {losses_val[-patience:]:.4e}")
                    break
            else:
                pass
            
        
        print(f"INN training took {time.time() - start_time_train:.4f} seconds")

        ## Inference
        self.inference(x_test)
        


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
        print(layer_sizes)
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
    # Grad_get_loss = jax.value_and_grad(get_loss, argnums=1, has_aux=True)
    Grad_get_loss = jax.jit(jax.value_and_grad(get_loss, argnums=1, has_aux=True), static_argnames=['self'])

    def inference(self, x_test):
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
        start_time_inference = time.time()
        u_pred = self.forward(self.params, self.activation, x_test[0]) # (ndata_train, var)
        print(f"\tInference time: {time.time() - start_time_inference:.4f} seconds")    
    
class Regression_CPMLP(Regression_INN):
    def __init__(self, cls_data, config):
        super().__init__(cls_data, config) # prob being dropout probability
        
        self.forward = forward_CPMLP
        self.v_forward = v_forward_CPMLP
        self.vv_forward = vv_forward_CPMLP
        self.ninput = cls_data.dim
        self.nmode = int(config['MODEL_PARAM']['nmode'])
        self.nlayers = config['MODEL_PARAM']["nlayers_cp"]
        self.nneurons = config['MODEL_PARAM']["nneurons_cp"]
        self.activation = config['MODEL_PARAM']["activation_cp"]
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_MLP'])
        
        assert self.cls_data.var == 1, "CPMLP only works for 1d output for now"
        ### initialization of trainable parameters
        layer_sizes = [1] + [self.nmode] + self.nlayers * [self.nneurons] + [self.nmode]
        print(layer_sizes)
        # print(param.shape for param in self.params)
        self.params = {}
        key = jax.random.PRNGKey(self.key)
        for i in range(self.ninput):
            key, subkey = jax.random.split(key)
            self.params[i] = self.init_network_params(layer_sizes, subkey)
        print(f"num_mode {jax.tree.map(lambda x: x.shape, self.params)}")
        # jax.debug.print("num_mode {params}", params = jax.tree.map(lambda x: x.shape, self.params))
         
        weights, biases = 0,0
        for layer in self.params[0]:
            w, b = layer[0], layer[1]
            weights += w.shape[0]*w.shape[1]
            biases += b.shape[0]
        print("------------CPMLP-------------")
        print(jax.tree_map(lambda x: x.shape, self.params))
        print(f"# of training parameters: {(weights+biases) * self.ninput}")

# class Regression_CPMLP(Regression_INN):
#     def __init__(self, cls_data, config):
#         super().__init__(cls_data, config) # prob being dropout probability
        
#         self.forward = forward_CPMLP
#         self.v_forward = v_forward_CPMLP
#         self.vv_forward = vv_forward_CPMLP
#         self.ninput = cls_data.dim
#         self.nmode = int(config['MODEL_PARAM']['nmode'])
#         self.nlayers = config['MODEL_PARAM']["nlayers"]
#         self.nneurons = config['MODEL_PARAM']["nneurons"]
#         self.activation = config['MODEL_PARAM']["activation"]
#         self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_MLP'])
        
#         assert self.cls_data.var == 1, "CPMLP only works for 1d output for now"
#         ### initialization of trainable parameters
#         layer_sizes = [1, 10, 10, 1]
#         # layer_sizes = [1] + [self.nmode] + self.nlayers * [self.nneurons] + [self.nmode] #only works for 1d output for now
#         print(layer_sizes)
#         # print(param.shape for param in self.params)
#         self.params = self.init_network_params(layer_sizes, jax.random.PRNGKey(self.key))
            
#         # weights, biases = 0,0
#         # for layer in self.params[0]:
#         #     w, b = layer[0], layer[1]
#         #     weights += w.shape[0]*w.shape[1]
#         #     biases += b.shape[0]
#         print("------------CPMLP-------------")
#         # print(jax.tree_map(lambda x: x.shape, self.params))
#         # print(f"# of training parameters: {(weights+biases) * self.ninput}")
        
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
    # 
    # Grad_get_loss = jax.value_and_grad(get_loss, argnums=1, has_aux=True)
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
        numParam = self.nmode*self.cls_data.dim*self.cls_data.var*self.nnode
        
        

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

    
    def get_acc_metrics(self, u, u_pred, type="test"):
        """ Get accuracy metrics. For regression, R2 will be returned.
            This function cannot be jitted because it uses scipy library
        --- input ---
        u: (ndata, nclass) integer vector that indicates class of the data
        u_train: (ndata, nclass) integer vector that indicates predicted class
        """
        if type == "train":
            bool_train_acc = self.config['TRAIN_PARAM']['bool_train_acc']
            if bool_train_acc:
                u_single = jnp.argmax(u, axis=1)
                u_pred_single = jnp.argmax(u_pred, axis=1)
                report = classification_report(np.array(u_single), np.array(u_pred_single), output_dict=True, zero_division=1)
                acc = report["accuracy"]
                acc_metrics = "Accuracy"
            else:
                acc, acc_metrics = 0,"Accuracy"
                
        elif type == "val" or type == "test":
            u_single = jnp.argmax(u, axis=1)
            u_pred_single = jnp.argmax(u_pred, axis=1)
            report = classification_report(np.array(u_single), np.array(u_pred_single), output_dict=True, zero_division=1)
            acc = report["accuracy"]
            acc_metrics = "Accuracy"
        return acc, acc_metrics


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

    def get_acc_metrics(self, u, u_pred, type="test"):
        """ Get accuracy metrics. For regression, R2 will be returned.
            This function cannot be jitted because it uses scipy library
        --- input ---
        u: (ndata, nclass) integer vector that indicates class of the data
        u_train: (ndata, nclass) integer vector that indicates predicted class
        """
        if type == "train":
            bool_train_acc = self.config['TRAIN_PARAM']['bool_train_acc']
            if bool_train_acc:
                u_single = jnp.argmax(u, axis=1)
                u_pred_single = jnp.argmax(u_pred, axis=1)
                report = classification_report(u_single, u_pred_single, output_dict=True, zero_division=1)
                acc = report["accuracy"]
                acc_metrics = "Accuracy"
            else:
                acc, acc_metrics = 0,"Accuracy"
                
        elif type == "val" or type == "test":
            u_single = jnp.argmax(u, axis=1)
            u_pred_single = jnp.argmax(u_pred, axis=1)
            report = classification_report(u_single, u_pred_single, output_dict=True, zero_division=1)
            acc = report["accuracy"]
            acc_metrics = "Accuracy"
        return acc, acc_metrics