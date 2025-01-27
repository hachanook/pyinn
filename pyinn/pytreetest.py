import jax
import jax.numpy as jnp

# Mock data structure based on your description
# This should represent actual data, replace with your actual gradients
grads = {0: [
    (jnp.ones((10, 1)), jnp.ones((10,))),
    (jnp.ones((10, 10)), jnp.ones((10,))),
    (jnp.ones((10, 10)), jnp.ones((10,)))
]}

# Compute the norm for each sub-array in the gradient pytree.
def compute_norms(tree):
    # Assume leaf of tree is a tuple of arrays
    return [(jnp.linalg.norm(arr1), jnp.linalg.norm(arr2)) for arr1, arr2 in tree]

norms = jax.tree_map(compute_norms, grads)

# Print the norms of the arrays in the gradient tree.
print(f"Grad norms: {norms}")