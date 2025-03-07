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
        # self.interpolate = LinearInterpolator(grid)
        if isinstance(grid_dms, jnp.ndarray): # for the same grids over dimension
            self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_dms_vars
        elif isinstance(grid_dms, list):
            self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_Vdms_vars
        else:
            print("Error: check the grid type")
            import sys
            exit()
        
    @partial(jax.jit, static_argnames=['self'])
    def get_Ju_idata_imd_idm_ivar(self, x_idata_idm, grid, u_imd_idm_ivar_nds):
        """ compute interpolation for a single mode, 1D function
        --- input ---
        x_idata_idm: scalar, jnp value / this can be any input
        grid: (J,) 1D vector of the grid
        u_imd_idm_ivar_nds: (J,) jnp 1D array
        --- output ---
        Ju_idata_imd_idm_ivar: scalar, 1D interpolated value
        """
        interpolate = LinearInterpolator(grid)
        # Ju_idata_imd_idm_ivar = self.interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        return Ju_idata_imd_idm_ivar
    ## for the same discretization over dimension
    get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,None,0)) # input: scalar, (J,), (var,J) / output: (var,)
    get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,0,0,0)) # input: (dim,), (dim,J) (dim,var,J) / output: (dim,var)
    get_Ju_idata_mds_dms_vars = jax.vmap(get_Ju_idata_imd_dms_vars, in_axes = (None,None,None,0)) # input: (dim,), (dim,J), (M,dim,var,J) / output: (M,dim,var)
    ## for variable discretization over dimension
    get_Ju_idata_mds_idm_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,None,None,0)) # input: scalar, (J,), (M,var,J) / output: (M,var)
    
    def get_Ju_idata_mds_Vdms_vars(self, x_idata, grid_dms, params):
        """ Prediction function
            --- input ---
            params: u_mds_dms_vars_nds, a "dim" componented list of (nmode, var, nnode) or (M,var,J)
            x_idata: x_idata_dms (dim,)
            --- return ---
            predicted output (M,dim,var)
        """
        Ju_idata_mds_Vdms_vars = jnp.zeros((params[0].shape[0], len(params), params[0].shape[1]), dtype=jnp.float64) # (M,dim,var)
        for idm, (x_idm, grid_idm, params_idm) in enumerate(zip(x_idata, grid_dms, params)):
            u_idata_imd_idm_vars = self.get_Ju_idata_mds_idm_vars(x_idm, grid_idm, params_idm) # (M,var)
            Ju_idata_mds_Vdms_vars = Ju_idata_mds_Vdms_vars.at[:,idm,:].set(u_idata_imd_idm_vars)
        return Ju_idata_mds_Vdms_vars

    
    def tucker(self, G, factors):
        """ serior computation of tucker decomposition 
        --- input ---
        G: core tensor, (M, M, ..., M) 
        factors: factor matrices, (dim, M)"""
        for factor in factors:
            G = jnp.tensordot(G, factor, axes=[0,0])
        return jnp.squeeze(G)
    v_tucker = jax.vmap(tucker, in_axes=(None,None,0)) # returns (var,)

    # @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """ Prediction function
            run one forward pass on given input data
            --- input ---
            params: u_mds_dms_vars_nds, (nmode, dim, var, nnode) or a "dim" componented list of (nmode, var, nnode)
            x_idata: x_idata_dms (dim,)
            --- return ---
            predicted output (var,)
        """
        pred = self.interpolate_mds_dms_vars(x_idata, self.grid_dms, params) # input: (dim,), (dim,J), (M,dim,var,J) / output: (M,dim,var)
        pred = jnp.prod(pred, axis=1) # output: (M,var)
        pred = jnp.sum(pred, axis=0) # output: (var,)
   
        return pred 

    v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
    vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)
        

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

        # self.interpolate = NonlinearInterpolator(grid, self.config)

        # if isinstance(grid_dms, jnp.ndarray): # for the same grids over dimension
        #     self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_dms_vars
        # elif isinstance(grid_dms, list):
        #     self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_Vdms_vars
        # else:
        #     print("Error: check the grid type")
        #     import sys
        #     exit()

    @partial(jax.jit, static_argnames=['self']) #    , 's_patch', 'alpha_dil', 'p_order', 'radial_basis', 'INNactivation'])
    def get_Ju_idata_imd_idm_ivar(self, x_idata_idm, grid, u_imd_idm_ivar_nds):
        """ compute interpolation for a single mode, 1D function
        --- input ---
        x_idata_idm: scalar, jnp value / this can be any input
        grid: (J,) 1D vector of the grid
        u_imd_idm_ivar_nds: (J,) jnp 1D array
        --- output ---
        Ju_idata_imd_idm_ivar: scalar, 1D interpolated value
        """
        interpolate = NonlinearInterpolator(grid, 
                                            self.s_patch, self.alpha_dil, self.p_order, 
                                            self.mbasis, self.radial_basis, self.activation)
        # Ju_idata_imd_idm_ivar = self.interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        return Ju_idata_imd_idm_ivar
    ## for the same discretization over dimension
    get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,None,0)) # input: scalar, (J,), (var,J) / output: (var,)
    get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,0,0,0)) # input: (dim,), (dim,J) (dim,var,J) / output: (dim,var)
    get_Ju_idata_mds_dms_vars = jax.vmap(get_Ju_idata_imd_dms_vars, in_axes = (None,None,None,0)) # input: (dim,), (dim,J), (M,dim,var,J) / output: (M,dim,var)
    ## for variable discretization over dimension
    get_Ju_idata_mds_idm_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,None,None,0)) # input: scalar, (J,), (M,var,J) / output: (M,var)
    

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