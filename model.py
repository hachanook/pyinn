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
from jax import lax
# from flax import linen as nn
# from flax.linen.dtypes import promote_dtype
from jax.scipy.interpolate import RegularGridInterpolator



@jax.jit
def get_Ju_idata_imd_idm_ivar(x_idata_idm, x_idm_nds, u_imd_idm_ivar_nds):
    """ compute interpolation for a single mode, 1D function
    --- input ---
    x_idata_ivar: scalar, jnp value / this can be any input
    x_idm_nds: (J,) tuple with one jnp element
    u_imd_idm_ivar_nds: (J,) 1 modal solution
    --- output ---
    Ju_idata_imd_idm_ivar: scalar
    """
    interpolate = jax.scipy.interpolate.RegularGridInterpolator((x_idm_nds,), u_imd_idm_ivar_nds, method='linear') # reformat x_nds
    Ju_idata_imd_idm_ivar = interpolate(x_idata_idm.reshape(1))[0]
    return Ju_idata_imd_idm_ivar

get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,0)) # output: (var,)
get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (0,0,0)) # output: (dim,var)

def get_Ju_idata_imd(x_idata_dms, x_dms_nds, u_imd_dms_vars_nds):
    # x_idata_dms: (dim,)
    # x_dms_nds: (dim, nnode)
    # u_imd_dms_vars_nds: (dim, var, nnode)
    
    Ju_idata_imd_dims_vars = get_Ju_idata_imd_dms_vars(x_idata_dms, x_dms_nds, u_imd_dms_vars_nds) # (dim, var)
    Ju_idata_imd = jnp.prod(Ju_idata_imd_dims_vars, axis=0)
    return Ju_idata_imd # (var,)
get_Ju_idata_mds = jax.vmap(get_Ju_idata_imd, in_axes = (None,None,0)) # output: (mds,var)

def get_Ju_idata(x_idata_dms, x_dms_nds, u_mds_dms_vars_nds):
    Ju_idata_mds = get_Ju_idata_mds(x_idata_dms, x_dms_nds, u_mds_dms_vars_nds) # (mds,var)
    Ju_idata = jnp.sum(Ju_idata_mds, axis=0) # returns (var,)
    return Ju_idata

# @partial(jax.jit, static_argnames=[]) # jit necessary
@jax.jit
def forward_INN(params, x_dms_nds, x_idata):
    """ Prediction function
        run one forward pass on given input data
        --- input ---
        params: u_mds_dms_vars_nds, (nmode, dim, var, nnode)
        x_dms_nds: nodal coordinates (dim, nnode)
        x_idata: x_idata_dms (dim,)
        --- return ---
        predicted output (var,)
    """
    pred = get_Ju_idata(x_idata, x_dms_nds, params)
    return pred
v_forward_INN = jax.vmap(forward_INN, in_axes=(None,None, 0)) # returns (ndata,)
vv_forward_INN = jax.vmap(v_forward_INN, in_axes=(None,None, 0)) # returns (ndata,)





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