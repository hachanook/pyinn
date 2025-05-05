import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax.experimental.sparse import BCOO
from jax.scipy.optimize import minimize
from functools import partial
from itertools import combinations
jax.config.update("jax_enable_x64", True)
import scipy
from scipy.sparse import csc_matrix
import scipy.sparse
from scipy.interpolate import griddata
import time, sys

import gauss


def uniform_mesh(xmin, xmax, nelem_x, dim, nodes_per_elem, elem_type, non_uniform_mesh_bool=False):
    """ Mesh generator
    --- Ijnputs ---
    xmin: minimum x-coordinate of the domain
    xmax: maximum x-coordinate of the domain
    nelem_x: number of elements in x-direction
    dim: problem dimension
    nodes_per_elem: number of nodes in one elements
    elem_type: element type
    --- Outputs ---
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    connectivity: elemental connectivity (nelem, node_per_elem*dim)
    nnode: number of nodes
    dof_global: global degrees of freedom
    """

    if elem_type == 'D1LN2N': # 1D 2-node linear element
        nelem = nelem_x
        nnode = nelem+1 # number of nodes
        dof_global = nnode*dim

        ## Nodes ##
        XY = np.ones([nnode, dim], dtype=np.double) * xmin
        dx = (xmax - xmin)/nelem_x # increment in the x direction

        n = 0 # This will allow us to go through rows in NL
        for i in range(1, nelem+2):
            if i == 1 or i == nelem+1: # boundary nodes
                XY[n,0] += (i-1)*dx
            else: # inside nodes
                XY[n,0] += (i-1)*dx
                if non_uniform_mesh_bool:
                    XY[n,0] += np.random.normal(0,0.2,1)*dx# for x values
            n += 1

        ## elements ##
        Elem_nodes = np.zeros([nelem, nodes_per_elem], dtype=np.int32)
        for j in range(1, nelem+1):
            Elem_nodes[j-1, 0] = j-1
            Elem_nodes[j-1, 1] = j

    return XY, Elem_nodes, nelem, nnode, dof_global

def get_shape_val_functions(elem_type):
    """ Shape function generator
    """
    ############ 1D ##################
    if elem_type == 'D1LN2N': # 1D linear element
        f1 = lambda x: 1./2.*(1 - x[0])
        f2 = lambda x: 1./2.*(1 + x[0])
        shape_fun = [f1, f2]

    return shape_fun

def get_shape_grad_functions(elem_type):
    """ Shape function gradient in the parent domain
    """
    shape_fns = get_shape_val_functions(elem_type)
    return [jax.grad(f) for f in shape_fns]

@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type']) # necessary
def get_shape_vals(Gauss_Num, dim, elem_type):
    """ Meature shape function values at quadrature points
    """
    shape_val_fns = get_shape_val_functions(elem_type)
    quad_points, quad_weights = gauss.get_quad_points(Gauss_Num, dim, elem_type)
    shape_vals = []
    for quad_point in quad_points:
        physical_shape_vals = []
        for shape_val_fn in shape_val_fns:
            physical_shape_val = shape_val_fn(quad_point)
            physical_shape_vals.append(physical_shape_val)

        shape_vals.append(physical_shape_vals)

    shape_vals = jnp.array(shape_vals) # (quad_num, nodes_per_elem)
    return shape_vals

@partial(jax.jit, static_argnames=['Gauss_Num', 'dim', 'elem_type']) # necessary
def get_shape_grads(Gauss_Num, dim, elem_type, XY, Elem_nodes):
    """ Meature shape function gradient values at quadrature points
    --- Outputs
    shape_grads_physical: shape function gradient in physcial coordinate (nelem, quad_num, nodes_per_elem, dim)
    JxW: Jacobian determinant times Gauss quadrature weights (nelem, quad_num)
    """
    shape_grad_fns = get_shape_grad_functions(elem_type)
    quad_points, quad_weights = gauss.get_quad_points(Gauss_Num, dim, elem_type)
    shape_grads = []
    for quad_point in quad_points:
        physical_shape_grads = []
        for shape_grad_fn in shape_grad_fns:
            physical_shape_grad = shape_grad_fn(quad_point)
            physical_shape_grads.append(physical_shape_grad)
        shape_grads.append(physical_shape_grads)

    shape_grads = jnp.array(shape_grads) # (quad_num, nodes_per_elem, dim)
    physical_coos = jnp.take(XY, Elem_nodes, axis=0) # (nelem, nodes_per_elem, dim)
    jacobian_dx_deta = jnp.sum(physical_coos[:, None, :, :, None] * shape_grads[None, :, :, None, :], axis=2, keepdims=True) # dx/deta
    # (nelem, quad_num, nodes_per_elem, dim, dim) -> (nelem, quad_num, 1, dim, dim)

    jacbian_det = jnp.squeeze(jnp.linalg.det(jacobian_dx_deta)) # det(J) (nelem, quad_num)
    jacobian_deta_dx = jnp.linalg.inv(jacobian_dx_deta) # deta/dx (nelem, quad_num, 1, dim, dim)
    shape_grads_physical = (shape_grads[None, :, :, None, :] @ jacobian_deta_dx)[:, :, :, 0, :]
    JxW = jacbian_det * quad_weights[None, :]
    return shape_grads_physical, JxW

def get_FEM_norm(XY, Elem_nodes, pinn, vv_u_fun, vv_Grad_u_fun, Gauss_Num_norm, elem_type):

    """ Variables
    XY: nodal coordinates (nnode, dim)
    Elem_nodes: elemental nodes (nelem, nodes_per_elem)
    L2_nom: (nelem, quad_num)
    L2_denom: (nelem, quad_num)
    H1_nom: (nelem, quad_num)
    H1_denom: (nelem, quad_num)
    XY_norm: (nelem, quad_num, dim)
    u_exact: (nelem, quad_num)
    uh: (nelem, quad_num)
    Grad_u_exact: (nelem, quad_num, dim)
    Grad_uh: (nelem, quad_num, dim)
    """

    dim = XY.shape[1]
    # quad_num_norm = Gauss_Num_norm**dim
    # L2_nom, L2_denom, H1_nom, H1_denom = 0,0,0,0
    shape_vals = get_shape_vals(Gauss_Num_norm, dim, elem_type) # (quad_num, nodes_per_elem)
    shape_grads_physical, JxW = get_shape_grads(Gauss_Num_norm, dim, elem_type, XY, Elem_nodes)

    physical_coos = jnp.take(XY, Elem_nodes, axis=0) # (nelem, nodes_per_elem, dim)
    xy_norm = jnp.sum(shape_vals[None, :, :, None] * physical_coos[:, None, :, :], axis=2) # (nelem, quad_num, dim)
    u_exact = vv_u_fun(xy_norm) # (nelem, quad_num, var)
    Grad_u_exact = vv_Grad_u_fun(xy_norm) # (nelem, quad_num, var, dim) # dim = 1

    uh = pinn.model.vv_forward(pinn.params, xy_norm) # (nelem, quad_num, var)
    Grad_uh = pinn.model.vv_g_forward(pinn.params, xy_norm) # (nelem, quad_num, var, dim) # dim = 1
    # uh = jnp.sum(shape_vals[None, :, :] * u_coos[:, None, :], axis=2)
    # Grad_uh = jnp.sum(shape_grads_physical[:, :, :, :] * u_coos[:, None, :, None], axis=2)

    L2_nom = jnp.sum((u_exact-uh)**2 * JxW[:,:,None])
    L2_denom = jnp.sum((u_exact)**2 * JxW[:,:,None])
    H1_nom = jnp.sum(((u_exact-uh)**2 + jnp.sum((Grad_u_exact-Grad_uh)**2, axis=3)) * JxW[:,:,None])
    H1_denom = jnp.sum(((u_exact)**2 + jnp.sum((Grad_u_exact)**2, axis=3)) * JxW[:,:,None])

    L2_norm = (L2_nom/L2_denom)**0.5
    H1_norm = (H1_nom/H1_denom)**0.5
    print(f'L2_norm: {L2_norm:.4e}')
    print(f'H1_norm: {H1_norm:.4e}')
    return L2_norm, H1_norm