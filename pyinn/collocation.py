from pyinn import dataset_classification, dataset_regression, model, train, plot
from jax import config
config.update("jax_enable_x64", True)
import os
import yaml
import jax.numpy as jnp
import jax
# from model import INN_linear, INN_nonlinear

# %% User Set up
with open('./pyinn/settings.yaml','r') as file:
    settings = yaml.safe_load(file)

gpu_idx = settings['GPU']['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = settings['PROBLEM']["run_type"]
interp_method = settings['PROBLEM']["interp_method"]
data_name = settings['DATA']["data_name"]

with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
    config = yaml.safe_load(file_dataConfig)
    config['interp_method'] = settings['PROBLEM']["interp_method"]
    config['TD_type'] = settings['PROBLEM']["TD_type"]



nmode = int(config['MODEL_PARAM']['nmode'])
nelem = int(config['MODEL_PARAM']['nelem'])
nnode = nelem+1
key = 234
dim = 2 # number of inputs
var = 1 # number of outputs

params = jax.random.uniform(jax.random.PRNGKey(key), (nmode, dim, 
                                            var, nnode), dtype=jnp.double)       

x_dms_nds = jnp.tile(jnp.linspace(0,1,nnode, dtype=jnp.float64), (2,1)) # (dim,nnode)
mdl = model.INN_linear(x_dms_nds[0,:], config)

x = jnp.array([0.5,0.4], dtype=jnp.float64)
xs = jnp.array([[0.5,0.4], [0.1,0.2]], dtype=jnp.float64)

# y = mdl.forward(params, x)
# ys = mdl.v_forward(params, xs)
grad_y = mdl.grad_forward(params, x)
# shape (var, dim):
# [[du1-dx1, du1-dx2, ..., du1-dx_dim],
#  [du2-dx1, du2-dx2, ..., du2-dx_dim],
#  [ ... ]
#  [du_var-dx1, du_var-dx2, ..., du_var-dx_dim]]

print(grad_y)
