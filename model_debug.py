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
from Interpolator import LinearInterpolator



@jax.jit
def get_Ju_idata_imd_idm_ivar(x_idata_idm, x_idm_nds, u_imd_idm_ivar_nds):
    """ compute interpolation for a single mode, 1D function
    --- input ---
    x_idata_idm: scalar, jnp value / this can be any input
    x_idm_nds: (J,) jnp 1D array
    u_imd_idm_ivar_nds: (J,) jnp 1D array
    --- output ---
    Ju_idata_imd_idm_ivar: scalar
    """
    # interpolate = RegularGridInterpolator((x_idm_nds,), u_imd_idm_ivar_nds, method='linear') # reformat x_nds
    Ju_idata_imd_idm_ivar = interpolate(x_idata_idm)
    return Ju_idata_imd_idm_ivar

get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,0)) # output: (var,)
get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (0,0,0)) # output: (dim,var)


dim = 2
nnode = 11
ndata = 3
nmode = 2
var = 1

x_data = jnp.array([[0.51,0.42],[0.28,0.18],[0.78,0.68]], dtype=jnp.float64) # (ndata = 3, dim = 2)
u_data = jnp.ones((nmode,dim,var,nnode), dtype=jnp.float64) # (nmode = 2, dim=2, var=1, nnode=11)
x_dms_nds = jnp.tile(jnp.linspace(0,1,nnode, dtype=jnp.float64), (dim,1)) # (dim,nnode)


idata = 0
idm = 0
imd = 0
ivar = 0
x_idata_idm = x_data[idata, idm] # scalar
x_idm_nds = x_dms_nds[idm,:] # (nnode,)
u_imd_idm_ivar_nds = u_data[imd, idm, ivar, :] # (nnode,)
print(x_idata_idm)
print(x_idm_nds)
print(u_imd_idm_ivar_nds)

interpolate = RegularGridInterpolator((x_idm_nds,), u_imd_idm_ivar_nds, method='linear')
Ju_idata_imd_idm_ivar = interpolate(x_idata_idm.reshape(1))[0]
print(Ju_idata_imd_idm_ivar)


# interpolate = RegularGridInterpolator_inhouse((x_idm_nds,), u_imd_idm_ivar_nds)
# Ju_idata_imd_idm_ivar = interpolate(x_idata_idm.reshape(1))[0]
# print(Ju_idata_imd_idm_ivar)

interpolate = LinearInterpolator(x_idm_nds, u_imd_idm_ivar_nds)

Ju_idata_imd_idm_ivar = interpolate(x_idata_idm)
print(Ju_idata_imd_idm_ivar)

Ju_idata_imd_idm_ivar = get_Ju_idata_imd_idm_ivar(x_idata_idm, x_idm_nds, u_imd_idm_ivar_nds)
print(Ju_idata_imd_idm_ivar)












