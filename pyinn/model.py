"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax
import jax.numpy as jnp
from jax.nn.initializers import uniform
from jax import config
config.update("jax_enable_x64", True)
from functools import partial
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)
from jax.scipy.interpolate import RegularGridInterpolator
import sys

from .Interpolator import LinearInterpolator, NonlinearInterpolator ## when using pyinn package
# from Interpolator import LinearInterpolator, NonlinearInterpolator ## when debugging
# from pyinn.Interpolator import LinearInterpolator, NonlinearInterpolator ## when debugging on streamlit

class INN_linear:
    def __init__(self, grid_dms, config): 
        """ 1D linear interpolation
        --- input --- 
        grid_dms: (dim,J) 1D vector of the grid and dimension or a "dim" componented list of (J,) arrays
        values: (J,) 1D vector of nodal values
        """
        self.grid_dms = grid_dms
        self.config = config
        
        if isinstance(grid_dms, jnp.ndarray): # same grids over dimension and normalized
            self.interpolate = LinearInterpolator(grid_dms) # we will use the same interpolator for all dimensions
            self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_dms_vars
        elif isinstance(grid_dms, list):
            self.interpolate = [LinearInterpolator(grid) for grid in grid_dms] # interpolate_dms
            
            all_same_length = all(len(grid) == len(grid_dms[0]) for grid in grid_dms) # check if all grids have the same length
            if all_same_length: # same grids over dimension but unnormalized
                self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_Vdms_vars
            else: # varying grids over dimension and normalized/unnormalized
                self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_Vdms_vars_varying_grid
        else:
            print("Error: check the grid type")
            exit()
    

    ## Case 1: the same grids over dimension and normalized
    @partial(jax.jit, static_argnames=['self', 'interpolate'])
    def get_Ju_idata_imd_idm_ivar(self, x_idata_idm, interpolate, u_imd_idm_ivar_nds):
        """ compute interpolation for a single mode, 1D function
        --- input ---
        x_idata_idm: scalar, jnp value / this can be any input
        interpolate: a 1D interpolator
        u_imd_idm_ivar_nds: (J,) jnp 1D array
        --- output ---
        Ju_idata_imd_idm_ivar: scalar, 1D interpolated value
        """
        Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        # Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        return Ju_idata_imd_idm_ivar
    get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,None,0)) # input: scalar, fun, (var,J) / output: (var,)
    get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,0,None,0)) # input: (dim,), fun, (dim,var,J) / output: (dim,var)
    get_Ju_idata_mds_dms_vars = jax.vmap(get_Ju_idata_imd_dms_vars, in_axes = (None,None,None,0)) # input: (dim,), fun, (M,dim,var,J) / output: (M,dim,var)
    get_Ju_idata_mds_idm_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,None,None,0)) # input: scalar, (M,var,J) / output: (M,var)
    
    ## Case 2: the same grids over dimension but unnormalized
    def get_Ju_idata_mds_Vdms_vars(self, x_idata, interpolate_dms, params): # this function cannot be jitted because interpolate_dms is a list of interpolators
        """ Prediction function
            --- input ---
            x_idata: x_idata_dms (dim,)
            interpolate_dms: a "dim" componented list of 1D interpolators
            params: (M, dim, var, J)
            --- output ---
            predicted output (M,dim,var)
        """
        Ju_idata_mds_Vdms_vars = jnp.zeros((params.shape[0], params.shape[1], params.shape[2]), dtype=jnp.float64) # (M,dim,var)
        for idm, (x_idm, interpolate) in enumerate(zip(x_idata, interpolate_dms)):
            u_idata_imd_idm_vars = self.get_Ju_idata_mds_idm_vars(x_idm, interpolate, params[:,idm,:,:]) # (M,var)
            Ju_idata_mds_Vdms_vars = Ju_idata_mds_Vdms_vars.at[:,idm,:].set(u_idata_imd_idm_vars)
        return Ju_idata_mds_Vdms_vars # (M,dim,var)


    ## Case 3: varying grids over dimension 
    def get_Ju_idata_mds_Vdms_vars_varying_grid(self, x_idata, interpolate_dms, params): # this function cannot be jitted because interpolate_dms is a list of interpolators
        """ Prediction function
            --- input ---
            x_idata: x_idata_dms (dim,)
            interpolate_dms: a "dim" componented list of 1D interpolators
            params: dim-componented list of (M, var, J)
            --- output ---
            predicted output (M,dim,var)
        """
        Ju_idata_mds_Vdms_vars = jnp.zeros((params[0].shape[0], len(params), params[0].shape[1]), dtype=jnp.float64) # (M,dim,var)
        for idm, (x_idm, interpolate, param) in enumerate(zip(x_idata, interpolate_dms, params)):
            u_idata_imd_idm_vars = self.get_Ju_idata_mds_idm_vars(x_idm, interpolate, param) # (M,var)
            Ju_idata_mds_Vdms_vars = Ju_idata_mds_Vdms_vars.at[:,idm,:].set(u_idata_imd_idm_vars)
        return Ju_idata_mds_Vdms_vars # (M,dim,var)


    @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """ Prediction function
            run one forward pass on given input data
            --- input ---
            params: (M, dim, var, J)  or a "dim" componented list of (M, var, J)
            x_idata: x_idata_dms (dim,)
            --- output ---
            predicted output (var,)
        """
        pred = self.interpolate_mds_dms_vars(x_idata, self.interpolate, params) # input: (dim,), (dim,J), (M,dim,var,J) / output: (M,dim,var)
        pred = jnp.prod(pred, axis=1) # output: (M,var)
        pred = jnp.sum(pred, axis=0) # output: (var,)
        return pred 

    v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
    vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)
    g_forward = jax.jacrev(forward, argnums=2) # returns (var, dim)
    gg_forward = jax.jacfwd(g_forward, argnums=2) # returns (var, dim, dim)
    v_g_forward = jax.vmap(g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)
    vv_g_forward = jax.vmap(v_g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)

        

class INN_nonlinear(INN_linear):
    def __init__(self, grid_dms, config):
        super().__init__(grid_dms, config) # prob being dropout probability

        self.s_patch = config['MODEL_PARAM']['s_patch']
        self.alpha_dil = config['MODEL_PARAM']['alpha_dil'] 
        self.p_order = config['MODEL_PARAM']['p_order']
        p_dict = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6} 
        self.mbasis = p_dict[self.p_order] 
        self.radial_basis = config['MODEL_PARAM']['radial_basis']
        self.activation = config['MODEL_PARAM']['INNactivation']

                    
        if isinstance(grid_dms, jnp.ndarray): # same grids over dimension and normalized
            self.interpolate = NonlinearInterpolator(grid_dms, 
                                                        self.s_patch, self.alpha_dil, self.p_order, 
                                                    self.mbasis, self.radial_basis, self.activation) # we will use the same interpolator for all dimensions
            self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_dms_vars
        elif isinstance(grid_dms, list):
            self.interpolate = [NonlinearInterpolator(grid, 
                                                        self.s_patch, self.alpha_dil, self.p_order, 
                                                    self.mbasis, self.radial_basis, self.activation) 
                                                    for grid in grid_dms] # interpolate_dms
        else:
            print("Error: check the grid type")
            exit()

## MLP

class MLP:
    def __init__(self, activation): 
        self.activation = activation

    # def relu(x):
    #     return jnp.maximum(0, x)

    @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """ Prediction function
            run one forward pass on given input data
            --- input ---
            params: (nlayer, nnode) or a "dim" componented list of (nmode, var, nnode)
            x_idata: x_idata_dms (dim,)
            --- return ---
            predicted output (var,)
        """
        
        # activations = x_idata
        # for w, b in params[:-1]:
        #     outputs = jnp.dot(w, activations) + b
        #     if self.activation == 'relu':
        #         activations = jax.nn.relu(outputs)
        #     elif self.activation == 'sigmoid':
        #         activations = jax.nn.sigmoid(outputs)
        #     elif self.activation == 'tanh':
        #         activations = jax.nn.tanh(outputs)
        #     elif self.activation == 'softplus':
        #         activations = jax.nn.softplus(outputs)
        #     elif self.activation == 'leaky_relu':
        #         activations = jax.nn.leaky_relu(outputs)
        #     else:
        #         raise ValueError(f"Unsupported activation function: {self.activation}")
        # final_w, final_b = params[-1]
        # return jnp.dot(final_w, activations) + final_b

        activations = x_idata
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            if self.activation == 'relu':
                activations = jax.nn.relu(outputs)
            elif self.activation == 'sigmoid':
                activations = jax.nn.sigmoid(outputs)
            elif self.activation == 'tanh':
                activations = jax.nn.tanh(outputs)
            elif self.activation == 'softplus':
                activations = jax.nn.softplus(outputs)
            elif self.activation == 'leaky_relu':
                activations = jax.nn.leaky_relu(outputs)
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
        final_w, final_b = params[-1]
        return jnp.dot(activations, final_w) + final_b

    v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata, var)
    vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata, var)
    g_forward = jax.jacrev(forward, argnums=2) # returns (var, dim)
    gg_forward = jax.jacfwd(g_forward, argnums=2) # returns (var, dim, dim)
    v_g_forward = jax.vmap(g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)
    vv_g_forward = jax.vmap(v_g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)