import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, vmap, jit
from jax.example_libraries import optimizers
import os


# GPU settings
gpu_idx = 5
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing


# Define the forcing function f(x)
def f(x):
    # f = (jnp.pi ** 2) * jnp.sin(jnp.pi * x)
    f = 50.0 * (50.0 * x**2 - 1.0) * jnp.exp(-25.0 * x**2)
    return f

# Initialize neural network parameters manually
def init_params(layer_sizes, key):
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params = []
    for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
        weight_key, bias_key = jax.random.split(k)
        W = jax.random.normal(weight_key, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
        b = jnp.zeros(out_dim)
        params.append((W, b))
    return params

# Forward pass
def forward(params, x):
    activ = x
    for i, (W, b) in enumerate(params[:-1]):
        # activ = jnp.tanh(jnp.dot(activ, W) + b)
        # activ = jax.nn.sigmoid(jnp.dot(activ, W) + b)
        activ = jax.nn.relu(jnp.dot(activ, W) + b)
    W, b = params[-1]
    return jnp.dot(activ, W) + b

# Define u(x) as a scalar-output function
def u_fn(params, x):
    x = x.reshape(-1, 1)
    return forward(params, x).squeeze()

# Compute PDE residual: -u''(x) - f(x)
def pde_residual(params, x):
    u = lambda x_: u_fn(params, x_)
    du_dx = grad(u)
    d2u_dx2 = grad(du_dx)
    return d2u_dx2(x) - f(x)

# Loss function
def loss_fn(params, x_int, x_bnd):
    # Interior loss (PDE residual)
    residual_fn = lambda x: pde_residual(params, x)
    res = vmap(residual_fn)(x_int)
    loss_pde = jnp.mean(res ** 2)

    # Boundary loss: u(0) = u(1) = 0
    u_bnd_vals = vmap(lambda x: u_fn(params, x))(x_bnd)
    loss_bnd = jnp.mean(u_bnd_vals ** 2)

    return loss_pde + loss_bnd

# Training step
@jit
def train_step(i, opt_state, x_int, x_bnd):
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params, x_int, x_bnd)
    opt_state = opt_update(i, grads, opt_state)
    return opt_state, loss

# Set up neural network
layer_sizes = [1, 20, 20, 1]
key = jax.random.PRNGKey(0)
params = init_params(layer_sizes, key)


# Collocation and boundary points
# x_int = jnp.linspace(0, 1, 100)
# x_bnd = jnp.array([0.0, 1.0])
xmin, xmax = -2.0, 2.0
epochs = 5000
learning_rate = 1e-2
n_CP_PDE = 100

x_int = jnp.linspace(xmin, xmax, 100)
x_bnd = jnp.array([xmin, xmax])


# Set up optimizer
opt_init, opt_update, get_params = optimizers.adam(1e-2)
opt_state = opt_init(params)


# Training loop
for epoch in range(epochs):
    opt_state, loss = train_step(epoch, opt_state, x_int, x_bnd)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Final parameters
trained_params = get_params(opt_state)

# Prediction and plotting
x_plot = jnp.linspace(xmin, xmax, 200)
u_pred = vmap(lambda x: u_fn(trained_params, x))(x_plot)
# u_exact = jnp.sin(jnp.pi * x_plot)
u_exact = jnp.exp(-25.0 * x_plot**2)

plt.plot(x_plot, u_pred, label="Predicted")
plt.plot(x_plot, u_exact, '--', label="Exact")
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("1D Poisson Equation - PINN (Pure JAX)")
plt.grid(True)
# Save the plot
plt.savefig("pinn_poisson_1d.png", dpi=300)
