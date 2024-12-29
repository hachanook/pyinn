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

from .model import *

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

    def get_acc_metrics(self, u, u_pred, type="test"):
        """ Get accuracy metrics. For regression, R2 will be returned.
        """
        if jnp.isnan(u_pred).any(): # if the prediction has nan value,
            print(f"[Error] INN prediction has NaN components")
            print(jnp.where(jnp.isnan(u_pred))[0])        

        if type == "train":
            bool_train_acc = self.config['TRAIN_PARAM']['bool_train_acc']
            if bool_train_acc:
                acc_metrics = "R2"
                acc = r2_score(u, u_pred)
            else:
                acc, acc_metrics = 0,"R2"
                
        elif type == "val" or type == "test":
            acc_metrics = "R2"
            acc = r2_score(u, u_pred)
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
        
        # loss_train_list, loss_test_list = [], []
        # acc_train_list, acc_test_list = [], []
        # if self.split_type == "TVT":
        #     loss_val_list, acc_val_list = [], []
        
        ## Train
        start_time_train = time.time()
        for epoch in range(self.num_epochs):
            epoch_list_loss, epoch_list_acc = [], [] 
            start_time_epoch = time.time()    
            count = 0
            for batch in self.train_dataloader:
                
                
                
                # time_batch = time.time()
                x_train, u_train = jnp.array(batch[0]), jnp.array(batch[1])
                # print(f"\t data transfer {time.time() - time_batch:.4f} seconds")

                ## Optimization step (or update)
                # time_batch = time.time()
                params, opt_state, loss_train, u_pred_train = self.update_optax(params, opt_state, x_train, u_train)
                # print(f"\t update {time.time() - time_batch:.4f} seconds")

                # print(f"\t\tbatch {count+1}")
                # print("\t\t\t", jnp.where(jnp.isnan(u_pred_train))[0])
                
                # time_batch = time.time()
                acc_train, acc_metrics = self.get_acc_metrics(u_train, u_pred_train, "train")
                epoch_list_loss.append(loss_train)
                epoch_list_acc.append(acc_train)
                # print(f"\t append {time.time() - time_batch:.4f} seconds")

                
            batch_loss_train = np.mean(epoch_list_loss)
            batch_acc_train = np.mean(epoch_list_acc)
                
            print(f"Epoch {epoch+1}")
            print(f"\tTraining loss: {batch_loss_train:.4e}")
            if self.config['TRAIN_PARAM']['bool_train_acc']:
                print(f"\tTraining {acc_metrics}: {batch_acc_train:.4f}")
            else:
                pass
            print(f"\tEpoch {epoch+1} training took {time.time() - start_time_epoch:.4f} seconds")

            ## Validation
            if (epoch+1)%self.validation_period == 0:
                epoch_list_loss, epoch_list_acc = [], [] 
                if self.split_type == "TT": # when there are only train & test data
                    self.val_dataloader = self.test_dataloader # deal test data as validation data
                for batch in self.val_dataloader:
                    x_val, u_val = jnp.array(batch[0]), jnp.array(batch[1])
                    _, _, loss_val, u_pred_val = self.update_optax(params, opt_state, x_val, u_val)
                    
                    # ## debug
                    # if jnp.isnan(u_pred_val).any():
                    #     idx = jnp.where(jnp.isnan(u_pred_val))[0][0]
                    #     print(idx)   
                    #     print(x_val[idx])
                    
                    acc_val, acc_metrics = self.get_acc_metrics(u_val, u_pred_val)
                    epoch_list_loss.append(loss_val)
                    epoch_list_acc.append(acc_val)
                
                batch_loss_val = np.mean(epoch_list_loss)
                batch_acc_val = np.mean(epoch_list_acc)
                print(f"\tValidation loss: {batch_loss_val:.4e}")
                print(f"\tValidation {acc_metrics}: {batch_acc_val:.4f}")

                if self.cls_data.data_name == "IGAMapping2D" and batch_loss_val < 1e-3:
                    # stopping criteria for the IGA inverse mapping; multi-CAD-patch C-IGA paper
                    break
            if (self.cls_data.data_name == "8D_1D_physics" or self.cls_data.data_name == "10D_5D_physics") and batch_loss_train < 4e-4:
                break
            
        self.params = params
        print(f"INN training took {time.time() - start_time_train:.4f} seconds")
        # if importlib.util.find_spec("GPUtil") is not None: # report GPU memory usage
        #     mem_report('After training', gpu_idx)

        ## Test 
        start_time_test = time.time()
        epoch_list_loss, epoch_list_acc = [], [] 
        for batch in self.test_dataloader:
            x_test, u_test = jnp.array(batch[0]), jnp.array(batch[1])
            _, _, loss_test, u_pred_test = self.update_optax(params, opt_state, x_test, u_test)
            acc_test, acc_metrics = self.get_acc_metrics(u_test, u_pred_test)
            epoch_list_loss.append(loss_test)
            epoch_list_acc.append(acc_test)
        
        batch_loss_test = np.mean(epoch_list_loss)
        batch_acc_test = np.mean(epoch_list_acc)
        print("Test")
        print(f"\tTest loss: {batch_loss_test:.4e}")
        print(f"\tTest {acc_metrics}: {batch_acc_test:.4f}")
        print(f"\tTest took {time.time() - start_time_test:.4f} seconds") 

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