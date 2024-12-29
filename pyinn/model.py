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
from .Interpolator import LinearInterpolator, NonlinearInterpolator

class INN_linear:
    def __init__(self, grid, config):
        """ 1D linear interpolation
        --- input --- 
        grid: (J,) 1D vector of the grid
        values: (J,) 1D vector of nodal values
        """
        self.grid = grid
        self.config = config
        self.interpolate = LinearInterpolator(grid)
        
    @partial(jax.jit, static_argnames=['self'])
    def get_Ju_idata_imd_idm_ivar(self, x_idata_idm, u_imd_idm_ivar_nds):
        """ compute interpolation for a single mode, 1D function
        --- input ---
        x_idata_idm: scalar, jnp value / this can be any input
        u_imd_idm_ivar_nds: (J,) jnp 1D array
        --- output ---
        Ju_idata_imd_idm_ivar: scalar, 1D interpolated value
        """
        Ju_idata_imd_idm_ivar = self.interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        return Ju_idata_imd_idm_ivar
    get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,0)) # input: scalar, (var,J) / output: (var,)
    get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,0,0)) # input: (dim,), (dim,var,J) / output: (dim,var)
    get_Ju_idata_mds_dms_vars = jax.vmap(get_Ju_idata_imd_dms_vars, in_axes = (None,None,0)) # input: (dim,), (M,dim,var,J) / output: (M,dim,var)

    ## CP decomposition
    # def get_Ju_idata_imd(self, x_idata_dms, u_imd_dms_vars_nds):
    #     # x_idata_dms: (dim,)
    #     # x_dms_nds: (dim, nnode)
    #     # u_imd_dms_vars_nds: (dim, var, nnode)
        
    #     Ju_idata_imd_dims_vars = self.get_Ju_idata_imd_dms_vars(x_idata_dms, u_imd_dms_vars_nds) # output: (dim, var)
    #     Ju_idata_imd = jnp.prod(Ju_idata_imd_dims_vars, axis=0)
    #     return Ju_idata_imd # (var,)
    # get_Ju_idata_mds = jax.vmap(get_Ju_idata_imd, in_axes = (None,None,0)) # output: (mds,var)

    # def get_Ju_idata(self, x_idata_dms, u_mds_dms_vars_nds):
    #     Ju_idata_mds = self.get_Ju_idata_mds(x_idata_dms, u_mds_dms_vars_nds) # (mds,var)
    #     Ju_idata = jnp.sum(Ju_idata_mds, axis=0) # returns (var,)
    #     return Ju_idata

    # # @partial(jax.jit, static_argnames=[]) # jit necessary
    # # @jax.jit
    # @partial(jax.jit, static_argnames=['self'])
    # def forward(self, params, x_idata):
    #     """ Prediction function
    #         run one forward pass on given input data
    #         --- input ---
    #         params: u_mds_dms_vars_nds, (nmode, dim, var, nnode)
    #         x_dms_nds: nodal coordinates (dim, nnode)
    #         x_idata: x_idata_dms (dim,)
    #         --- return ---
    #         predicted output (var,)
    #     """
    #     pred = self.get_Ju_idata(x_idata, params)
    #     return pred
    # v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
    # vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)

    def tucker(self, G, factors):
        """ serior computation of tucker decomposition 
        --- input ---
        G: core tensor, (M, M, ..., M) 
        factors: factor matrices, (dim, M)"""
        for factor in factors:
            G = jnp.tensordot(G, factor, axes=[0,0])
        return jnp.squeeze(G)
    v_tucker = jax.vmap(tucker, in_axes=(None,None,0)) # returns (var,)

    @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """ Prediction function
            run one forward pass on given input data
            --- input ---
            params: u_mds_dms_vars_nds, (nmode, dim, var, nnode)
            x_dms_nds: nodal coordinates (dim, nnode)
            x_idata: x_idata_dms (dim,)
            --- return ---
            predicted output (var,)
        """
        # if self.config['TD_type']=='CP':
        pred = self.get_Ju_idata_mds_dms_vars(x_idata, params) # output: (M,dim,var)
        pred = jnp.prod(pred, axis=1) # output: (M,var)
        pred = jnp.sum(pred, axis=0) # output: (var,)
        
        # if self.config['TD_type']=='Tucker':
            
        #     G = params[0]
        #     factors = self.get_Ju_idata_mds_dms_vars(x_idata, params[1]) # output: (M,dim,var)
        #     # for factor in pred.transpose(1,0,2): # factor: (M,var)
        #     #     G = jnp.tensordot(G, factor, axes=[0,0])
        #     factors = factors.transpose(2,1,0) # (var,dim,M)
        #     # print(factors.shape)
        #     pred = self.v_tucker(G, factors)

        return pred 

    v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
    vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)
        

class INN_nonlinear(INN_linear):
    def __init__(self, grid, config):
        super().__init__(grid, config) # prob being dropout probability

        self.interpolate = NonlinearInterpolator(grid, self.config)

## MLP
def relu(x):
    return jnp.maximum(0, x)

@partial(jax.jit, static_argnames=['activation'])
def forward_MLP(params, activation, x_idata):
    # per-example predictions
    activations = x_idata
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        if activation == 'relu':
            activations = relu(outputs)
        elif activation == 'sigmoid':
            activations = jax.nn.sigmoid(outputs)
    final_w, final_b = params[-1]
    return jnp.dot(final_w, activations) + final_b
v_forward_MLP = jax.vmap(forward_MLP, in_axes=(None,None, 0)) # returns (ndata,)
vv_forward_MLP = jax.vmap(v_forward_MLP, in_axes=(None,None, 0)) # returns (ndata,)