## PINN3.py: parameterized PDE
import jax
import jax.numpy as jnp
from jax import random
from jax import jacfwd, jacrev
import numpy as np
import matplotlib.pyplot as plt
import optax
import os
import time
from functools import partial
from jax import config
from scipy.stats import qmc

config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # add this for memory pre-allocation
import importlib.util

# import tensorflow as tf
# import tensorflow_datasets as tfds
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt

if importlib.util.find_spec("GPUtil") is not None:  # for linux & GPU
    """If you are funning on GPU, please install the following libraries on your anaconda environment via
    $ conda install -c conda-forge humanize
    $ conda install -c conda-forge psutil
    $ conda install -c conda-forge gputil
    """
    import humanize, psutil, GPUtil

    # memory report
    def mem_report(num, gpu_idx):
        """This function reports memory usage for both CPU and GPU"""
        print(f"-{num}-CPU RAM Free: "+ humanize.naturalsize(psutil.virtual_memory().available))

        GPUs = GPUtil.getGPUs()
        gpu = GPUs[gpu_idx]
        # for i, gpu in enumerate(GPUs):
        print("---GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%\n".format(
                gpu_idx, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil * 100))


class CollocationPoints(Dataset):
    def __init__(self, cp_PDE, cp_IC, cp_BC_Dirichlet, cp_BC_Neumann, normal_vec_Neumann) -> None:
        self.cp_PDE = cp_PDE
        self.cp_IC = cp_IC
        self.cp_BC_Dirichlet = cp_BC_Dirichlet
        self.cp_BC_Neumann = cp_BC_Neumann
        self.normal_vec_Neumann = normal_vec_Neumann

    def __len__(self):
        return len(self.cp_PDE)

    def __getitem__(self, idx):
        return (self.cp_PDE[idx],self.cp_IC[idx],self.cp_BC_Dirichlet[idx],self.cp_BC_Neumann[idx],self.normal_vec_Neumann[idx])


class PINN:
    def __init__(self, layer_sizes, cls_cp, bc_ic_loss_weight, x_min, x_max, k, P, eta, d):
        """ """
        self.layer_sizes = layer_sizes
        self.cls_cp = cls_cp
        self.bc_ic_loss_weight = bc_ic_loss_weight
        self.x_min = x_min
        self.x_max = x_max
        self.k = k  # heat conduction coefficient (W/mm/K)
        self.P = P  # Laser power (W)
        self.eta = eta  # absorptivity, unitless ratio
        self.d = d  # laser penetration depth (mm)
        self.r_b = 0.05  # laser radius, (mm) fixed / 0.05 as default
        self.v = 250  # laser scan speed (mm/s)

        self.params = self.init_network_params(layer_sizes, random.PRNGKey(123))

        weights, biases = 0, 0
        for layer in self.params:
            w, b = layer[0], layer[1]
            weights += w.shape[0] * w.shape[1]
            biases += b.shape[0]
        print(f"FFNN parameters are {weights+biases}")

    # A helper function to randomly initialize weights and biases
    # for a dense neural network layer
    def random_layer_params(self, m, n, key, scale=1e-1):  # m input / n output neurons
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    # Initialize all layers for a fully-connected neural network with sizes "sizes"
    def init_network_params(self, sizes, key):
        keys = random.split(key, len(sizes))
        return [self.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

    @partial(jax.jit, static_argnames=["self"])
    def relu(self, x):
        return jnp.maximum(0, x)

    @partial(jax.jit, static_argnames=["self"])
    def sigmoid(self, x):
        return 1 / (1 + jnp.exp(-x))

    ## networkion functions
    @partial(jax.jit, static_argnames=["self"])
    def network(self, params, x):
        """
        --- inputs ---
        x: inputs, vector
        params
        """
        # normalize
        activations = (x - self.x_min) / (self.x_max - self.x_min)
        for w, b in params[:-1]:
            outputs = jnp.sum(w * activations[None, :], axis=1) + b  # (#, ), vector
            # activations = self.relu(outputs)
            activations = self.sigmoid(outputs)

        final_w, final_b = params[-1]
        outputs = (jnp.sum(final_w * activations[None, :], axis=1) + final_b)  # (#, ), vector
        return outputs.reshape()  # scalar

    v_network = jax.vmap(network, in_axes=(None, None, 0))
    vv_network = jax.vmap(v_network, in_axes=(None, None, 0))
    # dx_network = jax.jit(jax.grad(network, argnums=1), static_argnames=['self'])
    dx_network = jax.grad(network, argnums=2)
    hessian_network = jacfwd(jacrev(network, argnums=2), argnums=2)

    def rhs_function(self, x):
        "Source term"

        ## Call variables
        x, y, z, t = x[0], x[1], x[2], x[3]  ## 4 input variables
        P, eta, d = self.P, self.eta, self.d
        r_b, v = self.r_b, self.v
        S_h = (2 * P * eta / (jnp.pi * r_b**2 * d) * jnp.exp(-2 * ((x - v * t) ** 2 + (y - 2.5) ** 2) / r_b**2)) * jnp.heaviside(z-(2.5-d),0)
        # S_h = (2 * P * eta / (jnp.pi * r_b**2 * d) * jnp.exp(-2 * ((x - v * t) ** 2 + (y - 2.5) ** 2) / r_b**2))
        return S_h

    v_rhs_function = jax.vmap(rhs_function, in_axes=(None, 0))  # returns vector

    def pde_residuum(self, params, x):
        """
        --- iput ---
        x: input, vector (dim, )
        --- output ---
        PDE residual, scalar
        """
        dTdt = self.dx_network(params, x)[3]  # [dTdt]
        laplacian = jnp.diagonal(self.hessian_network(params, x))[:3]  # [d2udx2, d2udy2, d2udz2]

        k = self.k
        S_h = self.rhs_function(x)
        return dTdt - k * jnp.sum(laplacian) - S_h  # scalar

    v_pde_residuum = jax.vmap(pde_residuum, in_axes=(None, None, 0))  # returns vector

    def ic_residuum(self, params, x):
        """Boundary condition on left and right surfaces, Difichlet BC
        --- iput ---
        x: input, vector (dim, )
        --- output ---
        PDE residual, scalar
        """
        return self.network(params, x) - 0.0  # scalar

    v_ic_residuum = jax.vmap(ic_residuum, in_axes=(None, None, 0))  # returns vector

    def bc_residuum_Dirichlet(self, params, x):
        """Boundary condition on left and right surfaces, Difichlet BC
        --- iput ---
        x: input, vector (dim, )
        --- output ---
        PDE residual, scalar
        """
        return self.network(params, x) - 0.0  # scalar

    v_bc_residuum_Dirichlet = jax.vmap(bc_residuum_Dirichlet, in_axes=(None, None, 0))  # returns vector

    def bc_residuum_Neumann(self, params, x, n):
        """Boundary condition on left and right surfaces
        --- iput ---
        x: input, vector (spatial dim, )
        n: normal vector (spatial dim, )
        --- output ---
        PDE residual, scalar
        """
        dTdxyz = self.dx_network(params, x)[:3]  # [dTdx, dTdy, dTdz]

        return jnp.dot(dTdxyz, n) - 0.0  # scalar

    v_bc_residuum_Neumann = jax.vmap(bc_residuum_Neumann, in_axes=(None, None, 0, 0))  # returns vector

    def loss_fn(self,params,cp_PDE,cp_IC,cp_BC_Dirichlet,cp_BC_Neumann,normal_vec_Neumann,bc_ic_loss_weight):

        ## PDE residual
        pde_residuum_at_cp = self.v_pde_residuum(params, cp_PDE)
        pde_loss = 0.5 * jnp.mean(jnp.square(pde_residuum_at_cp))

        ## IC
        ic_residuum_at_cp = self.v_ic_residuum(params, cp_IC)
        ic_loss = 0.5 * jnp.mean(jnp.square(ic_residuum_at_cp))

        ## BC residual Dirichlet
        bc_residuum_Dirichlet_at_cp = self.v_bc_residuum_Dirichlet(params, cp_BC_Dirichlet)
        bc_loss_Dirichlet = 0.5 * jnp.mean(jnp.square(bc_residuum_Dirichlet_at_cp))

        ## BC residual Neumann
        bc_residuum_Neumann_at_cp = self.v_bc_residuum_Neumann(params, cp_BC_Neumann, normal_vec_Neumann)
        bc_loss_Neumann = 0.5 * jnp.mean(jnp.square(bc_residuum_Neumann_at_cp))

        return pde_loss + bc_ic_loss_weight * (ic_loss + bc_loss_Dirichlet + bc_loss_Neumann)

    Grad_loss_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=1), static_argnames=["self"])

    @partial(jax.jit, static_argnames=["self"])  # This will slower the function ######## I WAS EDITING THIS
    def update_optax(self,params,opt_state,cp_PDE,cp_IC,cp_BC_Dirichlet,cp_BC_Neumann,normal_vec_Neumann,bc_ic_loss_weight,):
        (loss, grads) = self.Grad_loss_fn(params,cp_PDE,cp_IC,cp_BC_Dirichlet,cp_BC_Neumann,normal_vec_Neumann,bc_ic_loss_weight)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self, num_epochs, batch_size, learning_rate, stopping_criteria):
        self.batch_size = batch_size
        params = self.params

        cp_dataloader = DataLoader(self.cls_cp, batch_size=self.batch_size, shuffle=True)

        ## Define optimizer
        self.optimizer = optax.adam(learning_rate)
        opt_state = self.optimizer.init(params)

        loss_train_list = []
        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_list_loss = []
            for batch in cp_dataloader:

                cp_PDE = jnp.array(batch[0])
                cp_IC = jnp.array(batch[1])
                cp_BC_Dirichlet = jnp.array(batch[2])
                cp_BC_Neumann = jnp.array(batch[3])
                normal_vec_Neumann = jnp.array(batch[4])

                params, opt_state, loss = self.update_optax(params,opt_state,cp_PDE,cp_IC,cp_BC_Dirichlet,cp_BC_Neumann,normal_vec_Neumann,self.bc_ic_loss_weight)  # ADAM
                epoch_list_loss.append(loss)

            batch_loss_train = np.mean(epoch_list_loss)
            loss_train_list.append(batch_loss_train)
            if batch_loss_train < stopping_criteria:
                print("Converged. Training stopped at:")
                print(f"Epoch: {epoch+1}, loss: {batch_loss_train:.4e}")
                break

            if epoch % 100 == 0:
                print(f"Epoch: {epoch+1}, loss: {batch_loss_train:.4e}")

        print(f"PINN solver took {time.time()-start_time:0.4f} sec")
        self.params = params
        self.loss_train_list = loss_train_list

        if importlib.util.find_spec("GPUtil") is not None:  # report GPU memory usage
            mem_report("After solving", gpu_idx)


# %% Set up

# --------------------- Global setup -----------------------------
gpu_idx = 1  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

method = "PINN"

# --------------------- Training setup ----------------------------
nlayers = 3
nneurons = 40  # dim * 10, the smaller the better training

num_epochs = 10_000
batch_size = 10_000
learning_rate = 2e-3
stopping_criteria = 2e-3
bc_ic_loss_weight = 1.0
dim = 4  # x,y,z,t
var = 1
k, P, eta, d = 1.03, 101, 0.102, 0.0304

n_cp = 1000_000  # number of collocation points
# n_cp_BC_IC = int(n_cp_PDE / 10)
# n_cp_BC_IC = n_cp_PDE


if method == "PINN":
    layer_sizes = [dim] + nlayers * [nneurons] + [var]

    ## Sample collocation points
    x_min = np.array([0, 0, 0, 0], dtype=np.double)  # x,y,z,t
    x_max = np.array([10, 5, 2.5, 0.04], dtype=np.double)  # x,y,z,t

    ### PDE collocation points
    # cp_PDE = np.random.rand(n_cp, dim)
    # cp_PDE = cp_PDE * (x_max[None, :] - x_min[None, :]) + x_min[None, :]
    cp_sampler = qmc.LatinHypercube(d=dim)
    cp_PDE = cp_sampler.random(n=n_cp)
    cp_PDE = qmc.scale(cp_PDE, x_min, x_max)


    ### IC collocation points
    cp_IC = cp_PDE.copy()
    cp_IC[:, 3] = x_min[3]  # t=0 - initial condition

    ### BC collocation points
    cp_xp = cp_PDE.copy()
    cp_xp[:, 0] = x_max[0]  # x - positive surface, right
    cp_xn = cp_PDE.copy()
    cp_xn[:, 0] = x_min[0]  # x - negative surface, left
    cp_yp = cp_PDE.copy()
    cp_yp[:, 1] = x_max[1]  # y - positive surface
    cp_yn = cp_PDE.copy()
    cp_yn[:, 1] = x_min[1]  # y - negative surface
    cp_zp = cp_PDE.copy()
    cp_zp[:, 2] = x_max[2]  # z - positive surface
    cp_zn = cp_PDE.copy()
    cp_zn[:, 2] = x_min[2]  # z - negative surface

    normal_vec_yp = np.tile(jnp.array([0, 1, 0], dtype=np.double), (n_cp, 1))
    normal_vec_yn = np.tile(jnp.array([0, -1, 0], dtype=np.double), (n_cp, 1))
    normal_vec_zp = np.tile(jnp.array([0, 0, 1], dtype=np.double), (n_cp, 1))
    normal_vec_zn = np.tile(jnp.array([0, 0, -1], dtype=np.double), (n_cp, 1))

    cp_BC_Dirichlet = np.concatenate((cp_xp, cp_xn), axis=0)
    random_idx = np.random.choice(range(2 * n_cp), size=n_cp, replace=False)
    cp_BC_Dirichlet = cp_BC_Dirichlet[random_idx]

    cp_BC_Neumann = np.concatenate((cp_yp, cp_yn, cp_zp, cp_zn), axis=0)
    normal_vec_Neumann = np.concatenate((normal_vec_yp, normal_vec_yn, normal_vec_zp, normal_vec_zn), axis=0)
    random_idx = np.random.choice(range(4 * n_cp), size=n_cp, replace=False)
    cp_BC_Neumann = cp_BC_Neumann[random_idx]
    normal_vec_Neumann = normal_vec_Neumann[random_idx]

    ## Train PINN
    cp = CollocationPoints(cp_PDE, cp_IC, cp_BC_Dirichlet, cp_BC_Neumann, normal_vec_Neumann)
    pinn = PINN(layer_sizes,cp,bc_ic_loss_weight,jnp.array(x_min),jnp.array(x_max),k,P,eta,d)

    ### debug
    # u = pinn.network(pinn.params, cp_PDE[0])
    pinn.train(num_epochs, batch_size, learning_rate, stopping_criteria)



## Plot results

x = np.linspace(0, 10, 100)
y = np.linspace(0, 5, 100)
z = 2.5
X, Y = np.meshgrid(x, y)
times = np.linspace(0, 0.04, 5)  # time range
inputs = np.zeros((100, 100, dim), dtype=np.double)  # input space
inputs[:, :, 0] = X  # (100,100)
inputs[:, :, 1] = Y  # (100,100)
inputs[:, :, 2] = z  # z

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

for timestep, t in enumerate(times):
    inputs[:, :, 3] = t  # t
    inputs_jnp = jnp.array(inputs)  # (100,100,4)
    T = pinn.vv_network(pinn.params, inputs_jnp)
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, T, levels=50, cmap="viridis")
    plt.colorbar()
    plt.title(f"Time = {t:.2f}")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")

    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, "plots")
    fig_name = f"STP-PINN_{dim}inputs_z{z}_timestep_{timestep}.jpg"
    # plt.tight_layout()
    plt.savefig(os.path.join(path_figure, fig_name), dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()
