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
from Interpolator import *
from model import *




# @jax.jit
# def get_Ju_idata_imd_idm_ivar(x_idata_idm, x_idm_nds, u_imd_idm_ivar_nds):
#     """ compute interpolation for a single mode, 1D function
#     --- input ---
#     x_idata_idm: scalar, jnp value / this can be any input
#     x_idm_nds: (J,) jnp 1D array
#     u_imd_idm_ivar_nds: (J,) jnp 1D array
#     --- output ---
#     Ju_idata_imd_idm_ivar: scalar
#     """
#     # interpolate = RegularGridInterpolator((x_idm_nds,), u_imd_idm_ivar_nds, method='linear') # reformat x_nds
#     Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
#     return Ju_idata_imd_idm_ivar

# get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,0)) # output: (var,)
# get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (0,0,0)) # output: (dim,var)


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
# print(x_idata_idm)
# print(x_idm_nds)
# print(u_imd_idm_ivar_nds)

# interpolate = RegularGridInterpolator((x_idm_nds,), u_imd_idm_ivar_nds, method='linear')
# Ju_idata_imd_idm_ivar = interpolate(x_idata_idm.reshape(1))[0]
# print(Ju_idata_imd_idm_ivar)


# # interpolate = RegularGridInterpolator_inhouse((x_idm_nds,), u_imd_idm_ivar_nds)
# # Ju_idata_imd_idm_ivar = interpolate(x_idata_idm.reshape(1))[0]
# # print(Ju_idata_imd_idm_ivar)

# interpolate = LinearInterpolator(x_idm_nds)

# Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
# print(Ju_idata_imd_idm_ivar)

# Ju_idata_imd_idm_ivar = get_Ju_idata_imd_idm_ivar(x_idata_idm, x_idm_nds, u_imd_idm_ivar_nds)
# print(Ju_idata_imd_idm_ivar)


import yaml, os


# %% User Set up
with open('settings.yaml','r') as file:
    settings = yaml.safe_load(file)

gpu_idx = settings['GPU']['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = settings['PROBLEM']["run_type"]
interp_method = settings['PROBLEM']["interp_method"]
data_name = settings['DATA']["data_name"]

with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
    config = yaml.safe_load(file_dataConfig)


model = INN_nonlinear(x_idm_nds, config)
interpolate = NonlinearInterpolator(x_idm_nds, config)
# xi = x_data[idata, idm] # scalar
# interpolate(xi, u_imd_idm_ivar_nds)



u_imd_idm_ivar_nds = jnp.array([0, 0.1, 0.3, 0.2, 0.6, 0.7, 0.8, 0.99, 0.3, 0.2, 0.1])
x_nds = jnp.linspace(0, 1, 101, dtype=jnp.float64) # (101,)
v_interpolate = jax.vmap(interpolate, in_axes = (0,None))
u_nds = v_interpolate(x_nds, u_imd_idm_ivar_nds)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# U_exact = globals()["v_fun_"+cls_data.data_name](x_nds) # (101,L)    
fig = plt.figure(figsize=(6,5))
gs = gridspec.GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0])
# plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

ax1.plot(x_idm_nds, u_imd_idm_ivar_nds, 'o', color='k', linewidth = 4,  label='Original data')
ax1.plot(x_nds, u_nds, '-', color='g', linewidth = 4,  label='Interpolated')
ax1.set_xlabel("x", fontsize=16)
ax1.set_ylabel("u", fontsize=16)
ax1.tick_params(axis='both', labelsize=12)
# ax1.set_title('INN prediction', fontsize=16)
ax1.legend(shadow=True, borderpad=1, fontsize=14, loc='best')
ax1.set_ylim([0,2])
plt.tight_layout()

parent_dir = os.path.abspath(os.getcwd())
path_figure = os.path.join(parent_dir, 'plots')
fig.savefig(os.path.join(path_figure, "debug") , dpi=300)
plt.close()


