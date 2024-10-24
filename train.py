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
from torch.utils.data import DataLoader, random_split
import time
import torch
from sklearn.metrics import r2_score
import importlib.util

# from flax.training import train_state

# from dataset import *
from model import forward, v_forward
# from settings import *
# from dataset import v_fun_2D_1D_sine, vv_fun_2D_1D_sine
# from model import v_forward, vv_forward   

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

class Regression:
    def __init__(self, cls_data, nmode, nelem, prob=0.0):
        self.cls_data = cls_data
        self.nmode = nmode
        self.nelem = nelem
        self.nnode = nelem+1
        if cls_data.bool_normalize: # when the data is normalized
            self.x_dms_nds = jnp.tile(jnp.linspace(0,1,self.nnode, dtype=jnp.float64), (self.cls_data.dim,1)) # (dim,nnode)
        else: # when the data is not normalized
            self.x_dms_nds = jax.vmap(get_linspace, in_axes=(0,0,None))(cls_data.x_data_minmax["min"], cls_data.x_data_minmax["max"], self.nnode)

        self.key = 3264
        self.prob = prob # dropout prob
        # self.u_p_modes = jax.random.uniform(jax.random.PRNGKey(self.key), (self.nmode, self.cls_hidenn.cls_data.dim, 
        #                                           self.cls_hidenn.nnode, self.cls_hidenn.cls_data.var), dtype=jnp.double,
        #                                     minval=0.9, maxval=1.1)
        
        self.params = jax.random.uniform(jax.random.PRNGKey(self.key), (self.nmode, self.cls_data.dim, 
                                                  self.cls_data.var, self.nnode), dtype=jnp.double)
        
        numParam = self.nmode*self.cls_data.dim*self.cls_data.var*self.nnode
        print(f"# of training parameters: {numParam}")
        # self.u_p_modes_list = []
        
            
    @partial(jax.jit, static_argnames=['self']) # jit necessary
    def get_loss_TD(self, params, x_data, u_data):
        ''' Compute loss value at (m)th mode given upto (m-1)th mode solution, which is u_pred_old
        --- input ---
        u_p_modes: (nmode, dim, nnode, var)
        u_data: exact u from the data. (ndata_train, var)
        shape_vals_data, patch_nodes_data: defined in "get_HiDeNN_shape_fun"
        '''
        
        u_pred = v_forward(params, self.x_dms_nds, x_data) # (ndata_train, var)
        loss = ((u_pred- u_data)**2).mean()
        return loss, u_pred
    
    Grad_get_loss_TD = jax.jit(jax.value_and_grad(get_loss_TD, argnums=1, has_aux=True), static_argnames=['self'])
    
    @partial(jax.jit, static_argnames=['self']) # This will slower the function
    def update_optax(self, params, opt_state, x_data, u_data):
        ((loss, u_pred), grads) = self.Grad_get_loss_TD(params, x_data, u_data)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, u_pred    
    
    def train(self, num_epochs, batch_size, learning_rate):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        ## Split data and create dataloader
        generator = torch.Generator().manual_seed(42)
        split_data = random_split(dataset=self.cls_data,lengths=self.cls_data.split_ratio, generator=generator)
        
        if len(split_data) == 2:
            self.split_type="TT" # train and test
            train_data = split_data[0]
            test_data = split_data[1]
        elif len(split_data) == 3:
            self.split_type="TVT" # train, validation, test
            train_data = split_data[0]
            val_data = split_data[1]
            test_data = split_data[2]
            val_dataloader = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        # ## debug
        # x_data, u_data = self.cls_data.x_data, self.cls_data.u_data, 


        ## Define optimizer
        params = self.params
        self.optimizer = optax.adam(learning_rate)
        # self.optimizer = optax.rmsprop(start_learning_rate)
        opt_state = self.optimizer.init(params)
        
        loss_train_list, loss_test_list = [], []
        r2_train_list, r2_test_list = [], []
        if self.split_type == "TVT":
            loss_val_list, r2_val_list = [], []
        
        ## Train
        start_time_train = time.time()
        for epoch in range(num_epochs):
            epoch_list_loss, epoch_list_r2 = [], [] 
            start_time_epoch = time.time()    
            for batch in train_dataloader:
                x_train, u_train = jnp.array(batch[0]), jnp.array(batch[1])
                
                ## Optimization step (or update)
                params, opt_state, loss_train, u_pred_train = self.update_optax(params, opt_state, x_train, u_train)
                r2_train = r2_score(u_train, u_pred_train)
                epoch_list_loss.append(loss_train)
                epoch_list_r2.append(r2_train)
                
            batch_loss_train = np.mean(epoch_list_loss)
            batch_r2_train = np.mean(epoch_list_r2)
                
            print(f"Epoch {epoch}")
            print(f"\tTraining loss: {batch_loss_train:.4e}")
            print(f"\tTraining R2: {batch_r2_train:.4f}")
            print(f"\tEpoch {epoch} training took {time.time() - start_time_epoch:.4f} seconds")

            ## Validation
            if (epoch+1)%1 == 0:
                epoch_list_loss, epoch_list_r2 = [], [] 
                if self.split_type == "TT": # when there are only train & test data
                    val_dataloader = test_dataloader # deal test data as validation data
                for batch in val_dataloader:
                    x_val, u_val = jnp.array(batch[0]), jnp.array(batch[1])
                    _, _, loss_val, u_pred_val = self.update_optax(params, opt_state, x_val, u_val)
                    r2_val = r2_score(u_val, u_pred_val)
                    epoch_list_loss.append(loss_val)
                    epoch_list_r2.append(r2_val)
                
                batch_loss_val = np.mean(epoch_list_loss)
                batch_r2_val = np.mean(epoch_list_r2)
                print(f"\tValidation loss: {batch_loss_val:.4e}")
                print(f"\tValidation R2: {batch_r2_val:.4f}")
            
        self.params = params
        print(f"INN training took {time.time() - start_time_train:.4f} seconds")
        # if importlib.util.find_spec("humanize") is not None: # report GPU memory usage
        #     mem_report('After training', gpu_idx)

        ## Test 
        
        start_time_test = time.time()
        epoch_list_loss, epoch_list_r2 = [], [] 
        for batch in test_dataloader:
            x_test, u_test = jnp.array(batch[0]), jnp.array(batch[1])
            _, _, loss_test, u_pred_test = self.update_optax(params, opt_state, x_test, u_test)
            r2_test = r2_score(u_test, u_pred_test)
            epoch_list_loss.append(loss_test)
            epoch_list_r2.append(r2_test)

             
            # U_pred = v_forward(self.params, self.x_dms_nds, x_test) # (ndata,var)
            # U_exact = v_fun_2D_1D_sine(x_test) # (ndata,var)


        
        batch_loss_test = np.mean(epoch_list_loss)
        batch_r2_test = np.mean(epoch_list_r2)
        print("Test")
        print(f"\tTest loss: {batch_loss_test:.4e}")
        print(f"\tTest R2: {batch_r2_test:.4f}")
        print(f"\tTest took {time.time() - start_time_test:.4f} seconds")



