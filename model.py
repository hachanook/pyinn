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
from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from jax.scipy.interpolate import RegularGridInterpolator



# @jax.jit
def get_Ju_idata_imd_idm_ivar(x_idata_idm, x_idm_nds, u_imd_idm_ivar_nds):
    # compute interpolation for a single mode, 1D function
    # x_idata_ivar: scalar, jnp value / this can be any input
    # x_idm_nds: (J,) tuple with one jnp element
    # u_imd_idm_ivar_nds: (J,) 1 modal solution
    
    interpolate = jax.scipy.interpolate.RegularGridInterpolator((x_idm_nds,), u_imd_idm_ivar_nds, method='linear') # reformat x_nds
    Ju_idata_imd_idm_ivar = interpolate(x_idata_idm.reshape(1))[0]
    return Ju_idata_imd_idm_ivar

get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,0)) # output: (var,)
get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (0,0,0)) # output: (dim,var)

# x_idata_idm = jnp.array(0.5, dtype=jnp.float64)
# x_idm_nds = jnp.array([0,1,2], dtype=jnp.float64)
# u_imd_idm_ivar_nds = jnp.array([0,1,2], dtype=jnp.float64)
# Ju_idata_imd_idm_ivar = get_Ju_idata_imd_idm_ivar(x_idata_idm, x_idm_nds, u_imd_idm_ivar_nds)
# print(Ju_idata_imd_idm_ivar)
# # a=aa

# u_imd_idm_vars_nds = jnp.array([[0,1,2], [0,2,4]], dtype=jnp.float64)
# Ju_idata_imd_idm_vars = get_Ju_idata_imd_idm_vars(x_idata_idm, x_idm_nds, u_imd_idm_vars_nds)
# print(Ju_idata_imd_idm_vars)
# # a=aa


def get_Ju_idata_imd(x_idata_dms, x_dms_nds, u_imd_dms_vars_nds):
    # x_idata_dms: (dim,)
    # x_dms_nds: (dim, nnode)
    # u_imd_dms_vars_nds: (dim, var, nnode)
    
    Ju_idata_imd_dims_vars = get_Ju_idata_imd_dms_vars(x_idata_dms, x_dms_nds, u_imd_dms_vars_nds) # (dim, var)
    Ju_idata_imd = jnp.prod(Ju_idata_imd_dims_vars, axis=0)
    return Ju_idata_imd # (var,)
get_Ju_idata_mds = jax.vmap(get_Ju_idata_imd, in_axes = (None,None,0)) # output: (mds,var)


# x_idata_dms = jnp.array([0.5,1.0], dtype=jnp.float64) # (2,)
# x_dms_nds = jnp.array([[0,1,2], [0,2,4]], dtype=jnp.float64) # (2,3)
# u_imd_dms_vars_nds = jnp.ones((2,4,3), dtype=jnp.float64) # (2,4,3)
# Ju_idata_imd = get_Ju_idata_imd(x_idata_dms, x_dms_nds, u_imd_dms_vars_nds)
# # print(Ju_idata_imd)
# # a=aa

def get_Ju_idata(x_idata_dms, x_dms_nds, u_mds_dms_vars_nds):
    Ju_idata_mds = get_Ju_idata_mds(x_idata_dms, x_dms_nds, u_mds_dms_vars_nds) # (mds,var)
    Ju_idata = jnp.sum(Ju_idata_mds, axis=0) # returns (var,)
    return Ju_idata
# get_Ju_datas = jax.vmap(get_Ju_idata, in_axes = (0,None,None)) # output: (ndata,var)

# x_dms_nds = jnp.array([[0,1,2], [0,2,4]], dtype=jnp.float64) # (2,3)
# u_mds_dms_vars_nds = jnp.ones((5,2,4,3), dtype=jnp.float64) # (5,2,4,3)
# Ju_idata = get_Ju_idata(x_idata_dms, x_dms_nds, u_mds_dms_vars_nds)
# print(Ju_idata)

# x_datas_dms = jnp.array([[0.5,1.0], [0.5,1.0]], dtype=jnp.float64) # (2,)
# x_dms_nds = jnp.array([[0,1,2], [0,2,4]], dtype=jnp.float64) # (2,3)
# u_mds_dms_vars_nds = jnp.ones((5,2,4,3), dtype=jnp.float64) # (5,2,4,3)
# Ju_datas = get_Ju_datas(x_datas_dms, x_dms_nds, u_mds_dms_vars_nds)
# print(Ju_datas)
# a=aa

@partial(jax.jit, static_argnames=[]) # jit necessary
def forward(params, x_dms_nds, x_idata):
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
v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)


# x_idata_dms = jnp.array([0.5,1.0], dtype=jnp.float64) # (2,)
# x_dms_nds = jnp.array([[0,1,2], [0,2,4]], dtype=jnp.float64) # (2,3)
# u_mds_dms_vars_nds = jnp.ones((5,2,4,3), dtype=jnp.float64) # (5,2,4,3)
# Ju_idata = forward(u_mds_dms_vars_nds,  x_dms_nds, x_idata_dms)
# print(Ju_idata)
# a=aa