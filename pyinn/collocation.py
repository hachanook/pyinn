import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import optax
from functools import partial
import numpy as np
from scipy.stats import qmc
from torch.utils.data import DataLoader, random_split, Subset
import time
import torch
import importlib.util
import sys, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from .model import * ## when using pyinn
from model import * ## when debugging
# from pyinn.model import * ## when debugging on streamlit

import FEM, gauss



# Define the neural network

class PINN:
    def __init__(self, model):
        self.model = model
        

    # @partial(jax.jit, statIC_argnames=["PDE_func"])
    def PDE_residual(self, params, x, PDE_func, forcing_func):
        u = self.model.forward(params, x) # (var,)
        g_u = self.model.g_forward(params, x) # (var, dim)
        gg_u = self.model.gg_forward(params, x) # (var, dim, dim)
        f = forcing_func(x) # (var,)
        PDE_residual = PDE_func(x, u, g_u, gg_u, f) # scalar
        return PDE_residual # scalar
    v_PDE_residual = jax.vmap(PDE_residual, in_axes=(None, None, 0, None, None)) # (n_CP_PDE,) 
    
    def IC_residual(self, params, x, IC_func):
        u = self.model.forward(params, x) # (var,)
        g_u = self.model.g_forward(params, x)
        gg_u = self.model.gg_forward(params, x)
        IC_residual = IC_func(u, g_u, gg_u)
        return IC_residual
    v_IC_residual = jax.vmap(IC_residual, in_axes=(None, None, 0, None)) # (n_CP_IC,)

    def BC_D_residual(self, params, x, BC_D_func):
        u = self.model.forward(params, x) # (var,)
        # g_u = self.model.g_forward(params, x)
        # gg_u = self.model.gg_forward(params, x)
        BC_D_residual = BC_D_func(u)
        return BC_D_residual
    v_BC_D_residual = jax.vmap(BC_D_residual, in_axes=(None, None, 0, None)) # (n_CP_BC_D,)

    def BC_N_residual(self, params, x, BC_N_func):
        # u = self.model.forward(params, x) # (var,)
        g_u = self.model.g_forward(params, x)
        # gg_u = self.model.gg_forward(params, x)
        BC_N_residual = BC_N_func(g_u)
        return BC_N_residual
    v_BC_N_residual = jax.vmap(BC_N_residual, in_axes=(None, None, 0, None)) # (n_CP_BC_N,)


    # Loss function
    # @jax.jit
    def loss_fn(self, params, cp_PDE, cp_IC, cp_BC_D, cp_BC_N, PDE_func, IC_func, BC_D_func, BC_N_func, forcing_func):
        # PDE residual loss
        PDE_residuals = self.v_PDE_residual(params, cp_PDE, PDE_func, forcing_func)
        PDE_loss = jnp.mean(PDE_residuals**2)
        
        # Initial condition loss
        IC_residuals = self.v_IC_residual(params, cp_IC, IC_func)
        IC_loss = jnp.mean(IC_residuals**2)

        # Boundary condition loss
        BC_D_residuals = self.v_BC_D_residual(params, cp_BC_D, BC_D_func)
        BC_D_loss = jnp.mean(BC_D_residuals**2)
        # BC_D_loss = jnp.linalg.norm(BC_D_residuals, ord=2) / jnp.sqrt(cp_BC.shape[0]) # L2 norm
        
        # Boundary condition loss
        BC_N_residuals = self.v_BC_N_residual(params, cp_BC_N, BC_N_func)
        BC_N_loss = jnp.mean(BC_N_residuals**2)
        
        return PDE_loss + IC_loss + BC_D_loss + BC_N_loss # + 100*IC_loss + 100*BC_loss
    
    # Training loop
    def train_pinn(self, params, cp_PDE, cp_IC, cp_BC_D, cp_BC_N, PDE_func, IC_func, BC_D_func, BC_N_func, forcing_func, lr=1e-3, epochs=20_000, batch_size=1000):
        
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)

        # ### Degugging block starts 
        # for cp_PDE_i in cp_PDE:
        #     # if cp_PDE_i > 0.4:
        #     #     b = 1
        #     u = self.model.forward(params, cp_PDE_i) # (var,)
        #     a = self.PDE_residual(params, cp_PDE_i, PDE_func, forcing_func) # debug
        #     if jnp.abs(a) > 1e-1:
        #         print(a)
        #     print(cp_PDE_i, a)
        #     if jnp.isnan(a):
        #         raise ValueError("PDE residual contains NaN values.")
        # a = self.loss_fn(params, cp_PDE, cp_IC, cp_BC_D, cp_BC_N, PDE_func, IC_func, BC_D_func, BC_N_func, forcing_func) # debug
        # ### Degugging block ends
        
        ## divide the collocation points into batches
        n_batches = max(len(cp_PDE) // batch_size, 1)
        cp_PDE_batches = jnp.array_split(cp_PDE, n_batches)
        if n_batches >= len(cp_IC): # when the number of IC collocation points is less than the number of batches
            cp_IC_batches = n_batches * [cp_IC]
        else:
            cp_IC_batches = jnp.array_split(cp_IC, n_batches)
        if n_batches >= len(cp_BC_D): # when the number of BC_D collocation points is less than the number of batches
            cp_BC_D_batches = n_batches * [cp_BC_D]
        else:
            cp_BC_D_batches = jnp.array_split(cp_BC_D, n_batches)
        if n_batches >= len(cp_BC_N):
            cp_BC_N_batches = n_batches * [cp_BC_N]
        else:
            cp_BC_N_batches = jnp.array_split(cp_BC_N, n_batches)

        @jax.jit
        def step_MLP(params, opt_state, cp_PDE, cp_IC, cp_BC_D, cp_BC_N):
            loss, grads = jax.value_and_grad(self.loss_fn)(
                params, cp_PDE, cp_IC, cp_BC_D, cp_BC_N, PDE_func, IC_func, BC_D_func, BC_N_func, forcing_func
            )
            # grads = grads.at[:,:,:,[0,-1]].set(0.0) # set the Dirichlet BC
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        @jax.jit
        def step_INN(params, opt_state, cp_PDE, cp_IC, cp_BC_D, cp_BC_N):
            loss, grads = jax.value_and_grad(self.loss_fn)(
                params, cp_PDE, cp_IC, cp_BC_D, cp_BC_N, PDE_func, IC_func, BC_D_func, BC_N_func, forcing_func
            )
            grads = grads.at[:,:,:,[0,-1]].set(0.0) # set the Dirichlet BC
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss


        for epoch in range(epochs):

            for batch in zip(cp_PDE_batches, cp_IC_batches, cp_BC_D_batches, cp_BC_N_batches):
                cp_PDE_batch, cp_IC_batch, cp_BC_D_batch, cp_BC_N_batch = batch
                if model_type.startswith("MLP"):
                    params, opt_state, loss = step_MLP(params, opt_state, cp_PDE_batch, cp_IC_batch, cp_BC_D_batch, cp_BC_N_batch)
                if model_type.startswith("INN"):
                    params, opt_state, loss = step_INN(params, opt_state, cp_PDE_batch, cp_IC_batch, cp_BC_D_batch, cp_BC_N_batch)
                
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        self.params = params # save trained parameters

        # self.loss_fn(params, cp_PDE, cp_IC, cp_BC_D, cp_BC_N, PDE_func, IC_func, BC_D_func, BC_N_func)

        return params
    
    

# Example usage: Solving u_t = u_xx with u(0, t) = u(1, t) = 0 and u(x, 0) = sin(pi * x)
if __name__ == "__main__":
    
    
    ############################### Configuration ############################

    # GPU settings
    gpu_idx = 5
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

    # PDE settings
    PDE_type = "1D_static_heat_conduction" # PDE type
    # PDE_type = "4D_SPT_heat_conduction" # PDE type

    
    # Model settings

    # model_type = "MLP"
    model_type = "INN_nonlinear"

    config_MLP = {
        'nlayers' : 2,
        'nneurons' : 20,
        'activation' : "tanh", # Activation functions: sigmoid, tanh, softplus
    }

    config_chidenn = {
        "MODEL_PARAM" : {
            'nelem' : 100, # number of elements
            'nmode' : 1, # number of modes
            's_patch' : 2,
            'alpha_dil' : 20,
            'p_order' : 2,
            'radial_basis' : "cubicSpline", # Activataion functions
            # radial_basis : 'gaussian1'
            'INNactivation' : 'polynomial'
        },
    }

    # Training settings
    learning_rate = 1e-4 # 1e-2 works well for MLP
    batch_size = 100 # 1000 works well for MLP
    epochs = 200_000 # 10_000 works well for MLP
    n_cp_PDE = 100 

##########################################################################
    
           
    if PDE_type == "1D_static_heat_conduction":
        

        # input dimension
        dim = 1 # number of input dimensions
        var = 1 # number of output variables
        x_minmax = [[-2.0, 2.0]] # list of minmax for each dimension
        xmin, xmax = x_minmax[0][0], x_minmax[0][1]

        # Define collocation points
        ## PDE collocation points
        cp_sampler = qmc.LatinHypercube(d=dim)
        cp_PDE = cp_sampler.random(n=n_cp_PDE) # (n_cp_PDE, dim) in normalized space btw 0 and 1
        for idim, minmax in enumerate(x_minmax):
            cp_PDE[:, idim] = cp_PDE[:, idim] * (minmax[1] - minmax[0]) + minmax[0]
        cp_PDE = jnp.array(cp_PDE, dtype=jnp.float64) # (n_cp_PDE, dim)
        # cp_PDE = jnp.linspace(xmin, xmax, n_cp_PDE, dtype=jnp.float64).reshape(-1,1) # or we can linearly smaple it

        # @Matthew: this block sets the collocation points to be on the gauss points
        if model_type == "INN_nonlinear":
            ## Meausre INN collocation points at the gauss points
            Gauss_num = 3
            elem_type = "D1LN2N" # 1D linear element
            shape_vals = FEM.get_shape_vals(Gauss_num, dim, elem_type) # (quad_num, nodes_per_elem)
            nelem = config_chidenn["MODEL_PARAM"]["nelem"] 
            nodes_per_elem = 2 # number of nodes per element
            xmin, xmax = x_minmax[0][0], x_minmax[0][1]
            XY_host, Elem_nodes_host, nelem, nnode, dof_global = FEM.uniform_mesh(xmin, xmax, nelem, dim, nodes_per_elem, elem_type)
            XY = jnp.array(XY_host, dtype=jnp.float64) # (nnode, dim)
            Elem_nodes = jnp.array(Elem_nodes_host, dtype=jnp.int64) # (nelem, nodes_per_elem)
            XY_elem = jnp.take(XY, Elem_nodes, axis=0) # (nelem, nodes_per_elem, dim)
            XY_GPs = jnp.sum(shape_vals[None,:,:,None] * XY_elem[:,None,:,:], axis=2) # (nelem, quad_num, dim)
            cp_PDE = XY_GPs.reshape(-1, dim) # (n_cp_PDE, dim)
    


        ## Initial collocation points
        n_cp_IC = 1
        cp_IC = jnp.array([[0]], dtype=jnp.float64) # (n_cp_IC, dim) in normalized space btw 0 and 1
        
        ## Dirichlet Boundary collocation points
        n_cp_BC_D = 2 # not important for 1D spatial problems
        cp_BC_D = jnp.array([[-2],
                             [2]], dtype=jnp.float64) # (n_cp_BC_D, dim) 

        n_cp_BC_N = 1
        cp_BC_N = jnp.array([[1]], dtype=jnp.float64) # (n_cp_BC_N, dim)


        def PDE_func(x, u, g_u, gg_u, f):
            """ Spring-mass-damper system PDE
            k/m : spring constant / mass, = 1.1
            c/m : damping coeffICient / mass, 0.1
            u: (var,)
            g_u: (var, dim)
            gg_u: (var, dim, dim)
            f: (var,) - source term
            --- output ---
            PDE residual: scalar
            """
            # u = u[0]
            # u_x = g_u[0, 0]
            u_xx = gg_u[0, 0, 0]
            f = f[0]
            return u_xx - f
        
        def forcing_func(x):
            """ Forcing function
            x: (dim,)
            --- output ---
            f: (var,)
            """
            f = 50.0 * (50.0 * x**2 - 1.0) * jnp.exp(-25.0 * x**2)
            return f
        
        def IC_func(u, g_u, gg_u):
            """ Initial condition
            u: (var,)
            g_u: (var, dim)
            gg_u: (var, dim, dim)
            --- output ---
            IC residual: scalar
            """

            return 0
        
        def BC_D_func(u):
            """ Boundary condition Dirichlet
            u: (var,)
            --- output ---
            BC residual: scalar
            """
            u = u[0]
            return u - 0
        
        def BC_N_func(g_u):
            """ Boundary condition Neumann
            g_u: (var, dim)
            --- output ---
            BC residual: scalar
            """
            # g_u = g_u[0,0]
            return 0
           
        @jax.jit
        def u_fun(x):
            # x is a scalar, a is a parameter, mostly being jnp.pi
            u = jnp.exp(-25.0 * x**2)
            return u # (var,)
        v_u_fun = jax.vmap(u_fun, in_axes=(0))
        vv_u_fun = jax.vmap(v_u_fun, in_axes=(0))
        Grad_u_fun_1D = jax.jacrev(u_fun, argnums=0) # returns (var, dim)
        vv_Grad_u_fun = jax.vmap(jax.vmap(Grad_u_fun_1D, in_axes = (0)), in_axes = (0))

    elif PDE_type == "4D_SPT_heat_conduction":
        

        # input dimension
        dim = 4 # number of input dimensions
        var = 1 # number of output variables
        x_s_idx = [0,1]
        x_p_idx = [2]
        x_t_idx = [3]
        x_minmax = [[-2.0, 2.0],
                    [-2.0, 2.0],
                    [0, 0.04],
                    [1, 4]] # list of minmax for each dimension

        # Define collocation points
        ## PDE collocation points
        cp_sampler = qmc.LatinHypercube(d=dim)
        cp_PDE = cp_sampler.random(n=n_cp_PDE) # (n_cp_PDE, dim) in normalized space btw 0 and 1
        for idim, minmax in enumerate(x_minmax):
            cp_PDE[:, idim] = cp_PDE[:, idim] * (minmax[1] - minmax[0]) + minmax[0]
        cp_PDE = jnp.array(cp_PDE, dtype=jnp.float64) # (n_cp_PDE, dim)

        ## Initial collocation points
        # n_cp_IC = 1
        cp_IC = cp_PDE.at[:, 3].set(0) # (n_cp_IC, dim) in normalized space btw 0 and 1
        
        ## Dirichlet Boundary collocation points
        # n_cp_BC_D = 2 # not important for 1D spatial problems
        n_cp_BC_D = int(n_cp_PDE/4)
        cp_BC_D_left   = cp_PDE[           :n_cp_BC_D  ].at[:, 0].set(-2) # (n_cp_BC_D, dim)
        cp_BC_D_right  = cp_PDE[n_cp_BC_D  :2*n_cp_BC_D].at[:, 0].set(2) # (n_cp_BC_D, dim)
        cp_BC_D_bottom = cp_PDE[2*n_cp_BC_D:3*n_cp_BC_D].at[:, 1].set(-2) # (n_cp_BC_D, dim)
        cp_BC_D_top    = cp_PDE[3*n_cp_BC_D:           ].at[:, 1].set(2) # (n_cp_BC_D, dim)
        cp_BC_D = jnp.concatenate((cp_BC_D_left, cp_BC_D_right, cp_BC_D_bottom, cp_BC_D_top), axis=0) # (n_cp_BC_D, dim)
        ## shuffle the collocation points
        cp_BC_D = cp_BC_D[jax.random.permutation(jax.random.PRNGKey(0), cp_BC_D.shape[0])]
        

        n_cp_BC_N = 1
        cp_BC_N = jnp.array([[0,0,0,2]], dtype=jnp.float64) # (n_cp_BC_N, dim)


        

        def PDE_func(x, u, g_u, gg_u, f):
            """ Spring-mass-damper system PDE
            k/m : spring constant / mass, = 1.1
            c/m : damping coeffICient / mass, 0.1
            u: (var,)
            g_u: (var, dim)
            gg_u: (var, dim, dim)
            f: (var,) - source term
            --- output ---
            PDE residual: scalar
            """
            k = x[2]
            u_t = g_u[0, 3]
            u_xx = gg_u[0, 0, 0]
            u_yy = gg_u[0, 1, 1]
            f = f[0]
            return u_t - k * (u_xx + u_yy) - f
        
        def forcing_func(xs):
            """ Forcing function
            xs: (dim,)
            --- output ---
            f: (var,)
            """
            x = xs[0]
            y = xs[1]
            k = xs[2]
            t = xs[3]

            f = k*(15 * jnp.exp(-15*k*t)
                   - (1 - jnp.exp(-15*k*t))*(2500*(x**2+y**2)-100)
                    ) * jnp.exp(-25*(x**2+y**2))
            return jnp.array([f], dtype=jnp.float64) # (var,)
        
        def IC_func(u, g_u, gg_u):
            """ Initial condition
            u: (var,)
            g_u: (var, dim)
            gg_u: (var, dim, dim)
            --- output ---
            IC residual: scalar
            """
            u = u[0]
            return u - 0
        
        def BC_D_func(u):
            """ Boundary condition Dirichlet
            u: (var,)
            --- output ---
            BC residual: scalar
            """
            u = u[0]
            return u - 0
        
        def BC_N_func(g_u):
            """ Boundary condition Neumann
            g_u: (var, dim)
            --- output ---
            BC residual: scalar
            """
            # g_u = g_u[0,0]
            return 0
           
        @jax.jit
        def u_fun(xs):
            # x is a scalar, a is a parameter, mostly being jnp.pi
            x = xs[0]
            y = xs[1]
            k = xs[2]
            t = xs[3]

            u = (1-jnp.exp(-15*k*t))*jnp.exp(-25*(x**2+y**2))

            return jnp.array([u], dtype=jnp.float64) # (var,)
        
        v_u_fun = jax.vmap(u_fun, in_axes=(0))
        vv_u_fun = jax.vmap(v_u_fun, in_axes=(0))
        Grad_u_fun_1D = jax.jacrev(u_fun, argnums=0) # returns (var, dim)
        vv_Grad_u_fun = jax.vmap(jax.vmap(Grad_u_fun_1D, in_axes = (0)), in_axes = (0))
    
    
 
############################# End of Configuration ############################


    # Define the model architecture

    if model_type == "MLP":

        def init_params(layer_sizes, key):
            keys = jax.random.split(key, len(layer_sizes) - 1)
            params = []
            for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
                weight_key, bias_key = jax.random.split(k)
                W = jax.random.normal(weight_key, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
                b = jnp.zeros(out_dim)
                params.append((W, b))
            return params
        
        # def init_params(layer_sizes, key):
        #     keys = jax.random.split(key, len(layer_sizes) - 1)
        #     params = []
        #     for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        #         weight_key, bias_key = jax.random.split(k)
        #         W = jax.random.normal(weight_key, (out_dim, in_dim)) * jnp.sqrt(2 / in_dim)
        #         b = jnp.zeros(out_dim)
        #         params.append((W, b))
        #     return params

        ### initialization of trainable parameters
        layer_sizes = [dim] + config_MLP['nlayers'] * [config_MLP['nneurons']] + [var]
        params = init_params(layer_sizes, jax.random.PRNGKey(0))

        weights, biases = 0,0
        for layer in params:
            w, b = layer[0], layer[1]
            weights += w.shape[0]*w.shape[1]
            biases += b.shape[0]
        print(f"------------ MLP {layer_sizes} -------------")
        print(f"# of training parameters: {weights+biases}")

        model = MLP(config_MLP['activation'])
    
    
    elif model_type == "INN_nonlinear":
        nnode = config_chidenn["MODEL_PARAM"]['nelem'] + 1 # number of nodes
        grid = jnp.linspace(0, 1, nnode, dtype=jnp.float64) # (nnode,)
        # grid_dms = grid_dms.at[:,-1].add(1e-5) # set the last node to 1.0

        ### denormalize the input space -> we will deal with denormalized input space
        for idim, minmax in enumerate(x_minmax):
            grid_dms = [grid * (minmax[1] - minmax[0]) + minmax[0] for minmax in x_minmax] # "dim" component list of (nnode,)
        
        ### define model
        model = INN_nonlinear(grid_dms, config_chidenn)
        
        ### initialization of trainable parameters
        key = int(time.time())
        # params = jax.random.uniform(jax.random.PRNGKey(key), (config_chidenn["MODEL_PARAM"]['nmode'], dim,
        #                                         var, nnode), dtype=jnp.float64)       
        params = jnp.zeros((config_chidenn["MODEL_PARAM"]['nmode'], dim,
                                        var, nnode), dtype=jnp.float64) # (nmode, dim, var, nnode)
        # x_grid = model.grid_dms[0] # (nnode,)
        # U_grid = v_u_fun(x_grid) # (nnode,)
        # params = params.at[0,0,0,:].set(U_grid) # set the initial condition
        params = params.at[:,:,:,[0,-1]].set(0.0) # set the Dirichlet BC

        print(f"------------ INN nmode: {config_chidenn['MODEL_PARAM']['nmode']}, nelem: {config_chidenn['MODEL_PARAM']['nelem']}, s={config_chidenn['MODEL_PARAM']['s_patch']}, P={config_chidenn['MODEL_PARAM']['p_order']} -------------")
        print(f"# of training parameters: {params.shape[0]*params.shape[1]*params.shape[2]*params.shape[3]}")

        # ### debug start ###
        # x_grid = model.grid_dms[0] # (nnode,)
        # U_grid = v_u_fun(x_grid) # (nnode,)
        # params = params.at[0,0,0,:].set(U_grid) # set the initial condition
        # #### debug end ###
    
    
    # Initialize the PINN
    pinn = PINN(model)

    # Train the PINN
    trained_params = pinn.train_pinn(
        params, cp_PDE, cp_IC, cp_BC_D, cp_BC_N, PDE_func, IC_func, BC_D_func, BC_N_func, forcing_func, lr=learning_rate, epochs=epochs, batch_size=batch_size
    )

    # pinn.params = params # save trained parameters


    ######################################################## Error analysis ###########################################################

    
    if PDE_type == "1D_static_heat_conduction":
        ## Mesh generation
        nodes_per_elem, elem_type, Gauss_Num_norm = 2, 'D1LN2N', 10 # 1D linear element
        # if model_type == "INN_nonlinear":
        #     nelem = config_chidenn["MODEL_PARAM"]['nelem'] # number of elements
        # elif model_type == "MLP":
        nelem = 100 # number of elements
        xmin, xmax = x_minmax[0][0], x_minmax[0][1]
        XY_host, Elem_nodes_host, nelem, nnode, dof_global = FEM.uniform_mesh(xmin, xmax, nelem, dim, nodes_per_elem, elem_type)
        XY = jnp.array(XY_host, dtype=jnp.float64) # (nnode, dim)
        Elem_nodes = jnp.array(Elem_nodes_host, dtype=jnp.int64) # (nelem, nodes_per_elem)

        # Compute errors
        L2_norm_FEM, H1_norm_FEM = FEM.get_FEM_norm(XY, Elem_nodes, pinn, vv_u_fun, vv_Grad_u_fun, Gauss_Num_norm, elem_type)





    ######################################################## Plot ###########################################################

    if PDE_type == "1D_static_heat_conduction":
        ## create mesh and data
        ### in normalized space, create prediction
        xmin, xmax = x_minmax[0][0], x_minmax[0][1]
        
        x_nds = jnp.linspace(xmin, xmax, 201, dtype=jnp.float64).reshape(-1,1) # (101,1)
        U_pred = model.v_forward(pinn.params, x_nds) # (101,L)
        U_exact = v_u_fun(x_nds.reshape(-1)) # (101,)
        if model_type == "INN_nonlinear":
            x_grid = model.grid_dms[0].reshape(-1,1) # (nnode,1)
            U_grid = model.v_forward(pinn.params, x_grid) # for grid points

        fig = plt.figure(figsize=(6,5))
        gs = gridspec.GridSpec(1, 1)
        ax1 = fig.add_subplot(gs[0])
        # plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

        # ax1.plot(x_nds, U_exact, '-', color='k', linewidth = 4,  label='Original function')
        ax1.plot(x_nds.reshape(-1), U_exact, '-', color='k', linewidth = 4,  label='Analytical\nsolution')
        ax1.plot(x_nds, U_pred, '--', color='g', linewidth = 4,  label='Prediction')
        if model_type == "INN_nonlinear":
            ax1.plot(x_grid, U_grid, 'o', color='r',  markersize=5, label='Grid points')
        ax1.set_xlabel(fr"Position, x", fontsize=16)
        ax1.set_ylabel(fr"Temperature, u", fontsize=16)
        ax1.tick_params(axis='both', labelsize=12)
        # ax1.set_title('INN prediction', fontsize=16)
        ax1.legend(shadow=True, borderpad=1, fontsize=14, loc='best')
        plt.tight_layout()

        parent_dir = os.path.abspath(os.getcwd())
        path_figure = os.path.join(parent_dir, 'plots')
        fig.savefig(os.path.join(path_figure, "Coll_" + PDE_type + "_" + model_type + ".png") , dpi=300)
        plt.close()


    elif PDE_type == "4D_SPT_heat_conduction":
        
        xmin, xmax = x_minmax[0][0], x_minmax[0][1]
        ymin, ymax = x_minmax[1][0], x_minmax[1][1]
        umin, umax = 0.0, 1.0
        x_nds = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64)
        y_nds = jnp.linspace(ymin, ymax, 101, dtype=jnp.float64)
        X,Y = jnp.meshgrid(x_nds, y_nds) # (101,101) each
        XY = jnp.dstack((X, Y)) # (101,101,2)
        k=2.0
        t=0.04
        XYkt = jnp.concatenate((XY, jnp.full((101,101,1), k), jnp.full((101,101,1), t)), axis=2) # (101,101,dim=4)
        U_pred = model.vv_forward(pinn.params, XYkt) # (101,101,L)

        U_exact = vv_u_fun(XYkt) # (101,)

        fig = plt.figure(figsize=(14,5))
        gs = gridspec.GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

        color_map="viridis"
        surf1 = ax1.pcolormesh(X, Y, U_pred[:,:,0], cmap=color_map, vmin=umin, vmax=umax)
        ax1.set_xlabel(f"x", fontsize=16)
        ax1.set_ylabel(f"y", fontsize=16)
        ax1.set_xticks([-2, -1, 0, 1, 2])
        ax1.set_yticks([-2, -1, 0, 1, 2])
        ax1.tick_params(axis='both', labelsize=12)
        cbar1 = fig.colorbar(surf1, shrink=0.8, aspect=20, pad=0.02)
        cbar1.set_label(f'u', fontsize=14)
        cbar1.ax.tick_params(labelsize=12)
        ax1.set_title('Prediction', fontsize=16)

        surf2 = ax2.pcolormesh(X, Y, U_exact[:,:,0], cmap=color_map, vmin=umin, vmax=umax)
        ax2.set_xlabel(f"x", fontsize=16)
        ax2.set_ylabel(f"y", fontsize=16)
        ax1.set_xticks([-2, -1, 0, 1, 2])
        ax1.set_yticks([-2, -1, 0, 1, 2])
        ax2.tick_params(axis='both', labelsize=12)
        cbar2 = fig.colorbar(surf2, shrink=0.8, aspect=20, pad=0.02)
        cbar2.set_label(f'u', fontsize=14)
        cbar2.ax.tick_params(labelsize=12)
        ax2.set_title('Original function', fontsize=16)

        parent_dir = os.path.abspath(os.getcwd())
        path_figure = os.path.join(parent_dir, 'plots')
        fig.savefig(os.path.join(path_figure, "Coll_" + PDE_type + "_" + model_type + ".png") , dpi=300)
        plt.close()
