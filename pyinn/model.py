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
from jax.scipy.interpolate import RegularGridInterpolator
import sys

# from .Interpolator import LinearInterpolator, NonlinearInterpolator ## when using pyinn package
from Interpolator import LinearInterpolator, NonlinearInterpolator ## when debugging
# from pyinn.Interpolator import LinearInterpolator, NonlinearInterpolator ## when debugging on streamlit

class INN_linear:
    def __init__(self, grid_dms, config): 
        """ 1D linear interpolation
        --- input --- 
        grid_dms: (dim,J) 1D vector of the grid and dimension or a "dim" componented list of (J,) arrays
        values: (J,) 1D vector of nodal values
        """
        self.grid_dms = grid_dms
        self.config = config
        
        if isinstance(grid_dms, jnp.ndarray): # same grids over dimension and normalized
            self.interpolate = LinearInterpolator(grid_dms) # we will use the same interpolator for all dimensions
            self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_dms_vars
        elif isinstance(grid_dms, list):
            self.interpolate = [LinearInterpolator(grid) for grid in grid_dms] # interpolate_dms
            
            all_same_length = all(len(grid) == len(grid_dms[0]) for grid in grid_dms) # check if all grids have the same length
            if all_same_length: # same grids over dimension but unnormalized
                self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_Vdms_vars
            else: # varying grids over dimension and normalized/unnormalized
                self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_Vdms_vars_varying_grid
        else:
            print("Error: check the grid type")
            exit()
    

    ## Case 1: the same grids over dimension and normalized
    @partial(jax.jit, static_argnames=['self', 'interpolate'])
    def get_Ju_idata_imd_idm_ivar(self, x_idata_idm, interpolate, u_imd_idm_ivar_nds):
        """ compute interpolation for a single mode, 1D function
        --- input ---
        x_idata_idm: scalar, jnp value / this can be any input
        interpolate: a 1D interpolator
        u_imd_idm_ivar_nds: (J,) jnp 1D array
        --- output ---
        Ju_idata_imd_idm_ivar: scalar, 1D interpolated value
        """
        Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        # Ju_idata_imd_idm_ivar = interpolate(x_idata_idm, u_imd_idm_ivar_nds)
        return Ju_idata_imd_idm_ivar
    get_Ju_idata_imd_idm_vars = jax.vmap(get_Ju_idata_imd_idm_ivar, in_axes = (None,None,None,0)) # input: scalar, fun, (var,J) / output: (var,)
    get_Ju_idata_imd_dms_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,0,None,0)) # input: (dim,), fun, (dim,var,J) / output: (dim,var)
    get_Ju_idata_mds_dms_vars = jax.vmap(get_Ju_idata_imd_dms_vars, in_axes = (None,None,None,0)) # input: (dim,), fun, (M,dim,var,J) / output: (M,dim,var)
    get_Ju_idata_mds_idm_vars = jax.vmap(get_Ju_idata_imd_idm_vars, in_axes = (None,None,None,0)) # input: scalar, (M,var,J) / output: (M,var)
    
    ## Case 2: the same grids over dimension but unnormalized
    def get_Ju_idata_mds_Vdms_vars(self, x_idata, interpolate_dms, params): # this function cannot be jitted because interpolate_dms is a list of interpolators
        """ Prediction function
            --- input ---
            x_idata: x_idata_dms (dim,)
            interpolate_dms: a "dim" componented list of 1D interpolators
            params: (M, dim, var, J)
            --- output ---
            predicted output (M,dim,var)
        """
        Ju_idata_mds_Vdms_vars = jnp.zeros((params.shape[0], params.shape[1], params.shape[2]), dtype=jnp.float64) # (M,dim,var)
        for idm, (x_idm, interpolate) in enumerate(zip(x_idata, interpolate_dms)):
            u_idata_imd_idm_vars = self.get_Ju_idata_mds_idm_vars(x_idm, interpolate, params[:,idm,:,:]) # (M,var)
            Ju_idata_mds_Vdms_vars = Ju_idata_mds_Vdms_vars.at[:,idm,:].set(u_idata_imd_idm_vars)
        return Ju_idata_mds_Vdms_vars # (M,dim,var)


    ## Case 3: varying grids over dimension 
    def get_Ju_idata_mds_Vdms_vars_varying_grid(self, x_idata, interpolate_dms, params): # this function cannot be jitted because interpolate_dms is a list of interpolators
        """ Prediction function
            --- input ---
            x_idata: x_idata_dms (dim,)
            interpolate_dms: a "dim" componented list of 1D interpolators
            params: dim-componented list of (M, var, J)
            --- output ---
            predicted output (M,dim,var)
        """
        Ju_idata_mds_Vdms_vars = jnp.zeros((params[0].shape[0], len(params), params[0].shape[1]), dtype=jnp.float64) # (M,dim,var)
        for idm, (x_idm, interpolate, param) in enumerate(zip(x_idata, interpolate_dms, params)):
            u_idata_imd_idm_vars = self.get_Ju_idata_mds_idm_vars(x_idm, interpolate, param) # (M,var)
            Ju_idata_mds_Vdms_vars = Ju_idata_mds_Vdms_vars.at[:,idm,:].set(u_idata_imd_idm_vars)
        return Ju_idata_mds_Vdms_vars # (M,dim,var)


    @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """ Prediction function
            run one forward pass on given input data
            --- input ---
            params: (M, dim, var, J)  or a "dim" componented list of (M, var, J)
            x_idata: x_idata_dms (dim,)
            --- output ---
            predicted output (var,)
        """
        pred = self.interpolate_mds_dms_vars(x_idata, self.interpolate, params) # input: (dim,), (dim,J), (M,dim,var,J) / output: (M,dim,var)
        pred = jnp.prod(pred, axis=1) # output: (M,var)
        pred = jnp.sum(pred, axis=0) # output: (var,)
        return pred 

    v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata,)
    vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata,)
    g_forward = jax.jacrev(forward, argnums=2) # returns (var, dim)
    gg_forward = jax.jacfwd(g_forward, argnums=2) # returns (var, dim, dim)
    v_g_forward = jax.vmap(g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)
    vv_g_forward = jax.vmap(v_g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)

        

class INN_nonlinear(INN_linear):
    def __init__(self, grid_dms, config):
        super().__init__(grid_dms, config) # prob being dropout probability

        self.s_patch = config['MODEL_PARAM']['s_patch']
        self.alpha_dil = config['MODEL_PARAM']['alpha_dil'] 
        self.p_order = config['MODEL_PARAM']['p_order']
        p_dict = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6} 
        self.mbasis = p_dict[self.p_order] 
        self.radial_basis = config['MODEL_PARAM']['radial_basis']
        self.activation = config['MODEL_PARAM']['INNactivation']

                    
        if isinstance(grid_dms, jnp.ndarray): # same grids over dimension and normalized
            self.interpolate = NonlinearInterpolator(grid_dms, 
                                                        self.s_patch, self.alpha_dil, self.p_order, 
                                                    self.mbasis, self.radial_basis, self.activation) # we will use the same interpolator for all dimensions
            self.interpolate_mds_dms_vars = self.get_Ju_idata_mds_dms_vars
        elif isinstance(grid_dms, list):
            self.interpolate = [NonlinearInterpolator(grid, 
                                                        self.s_patch, self.alpha_dil, self.p_order, 
                                                    self.mbasis, self.radial_basis, self.activation) 
                                                    for grid in grid_dms] # interpolate_dms
        else:
            print("Error: check the grid type")
            exit()

## MLP

class MLP:
    def __init__(self, activation): 
        self.activation = activation

    # def relu(x):
    #     return jnp.maximum(0, x)

    @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """ Prediction function
            run one forward pass on given input data
            --- input ---
            params: (nlayer, nnode) or a "dim" componented list of (nmode, var, nnode)
            x_idata: x_idata_dms (dim,)
            --- return ---
            predicted output (var,)
        """
        
        # activations = x_idata
        # for w, b in params[:-1]:
        #     outputs = jnp.dot(w, activations) + b
        #     if self.activation == 'relu':
        #         activations = jax.nn.relu(outputs)
        #     elif self.activation == 'sigmoid':
        #         activations = jax.nn.sigmoid(outputs)
        #     elif self.activation == 'tanh':
        #         activations = jax.nn.tanh(outputs)
        #     elif self.activation == 'softplus':
        #         activations = jax.nn.softplus(outputs)
        #     elif self.activation == 'leaky_relu':
        #         activations = jax.nn.leaky_relu(outputs)
        #     else:
        #         raise ValueError(f"Unsupported activation function: {self.activation}")
        # final_w, final_b = params[-1]
        # return jnp.dot(final_w, activations) + final_b

        activations = x_idata
        for w, b in params[:-1]:
            outputs = jnp.dot(activations, w) + b
            if self.activation == 'relu':
                activations = jax.nn.relu(outputs)
            elif self.activation == 'sigmoid':
                activations = jax.nn.sigmoid(outputs)
            elif self.activation == 'tanh':
                activations = jax.nn.tanh(outputs)
            elif self.activation == 'softplus':
                activations = jax.nn.softplus(outputs)
            elif self.activation == 'leaky_relu':
                activations = jax.nn.leaky_relu(outputs)
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")
        final_w, final_b = params[-1]
        return jnp.dot(activations, final_w) + final_b

    v_forward = jax.vmap(forward, in_axes=(None,None, 0)) # returns (ndata, var)
    vv_forward = jax.vmap(v_forward, in_axes=(None,None, 0)) # returns (ndata, var)
    g_forward = jax.jacrev(forward, argnums=2) # returns (var, dim)
    gg_forward = jax.jacfwd(g_forward, argnums=2) # returns (var, dim, dim)
    v_g_forward = jax.vmap(g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)
    vv_g_forward = jax.vmap(v_g_forward, in_axes=(None,None, 0)) # returns (ndata, var, dim)


## Kolmogorov-Arnold Network (KAN)

class KAN:
    """
    Kolmogorov-Arnold Network implementation in pure JAX.
    Uses learnable basis functions (simplified implementation for efficiency).

    Architecture:
    - Each edge has learnable coefficients for basis functions
    - Uses radial basis functions (RBF) instead of B-splines for JAX compatibility
    - More efficient and numerically stable than recursive B-splines
    """

    def __init__(self, layer_sizes, grid_size=5, spline_order=3):
        """
        Initialize KAN architecture.

        Args:
            layer_sizes: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            grid_size: Number of basis centers (default: 5)
            spline_order: Controls basis width (default: 3)
        """
        self.layer_sizes = layer_sizes
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_layers = len(layer_sizes) - 1

        # Create basis function centers uniformly in [-3, 3]
        self.centers = jnp.linspace(-3, 3, grid_size, dtype=jnp.float64)
        # Basis width parameter
        self.width = 6.0 / (grid_size - 1) * spline_order

    @partial(jax.jit, static_argnames=['self'])
    def _compute_basis(self, x):
        """
        Compute RBF basis functions efficiently.

        Args:
            x: Input value (scalar)

        Returns:
            Basis function values (grid_size,)
        """
        # Gaussian RBF basis
        distances = (x - self.centers) / self.width
        basis_values = jnp.exp(-0.5 * distances**2)
        return basis_values

    @partial(jax.jit, static_argnames=['self'])
    def _compute_edge_function(self, x, edge_coeffs):
        """
        Compute learnable univariate function for one edge.

        Args:
            x: Input value
            edge_coeffs: Coefficients for basis functions (grid_size,)

        Returns:
            Function value
        """
        basis = self._compute_basis(x)
        return jnp.dot(basis, edge_coeffs)

    @partial(jax.jit, static_argnames=['self'])
    def _layer_forward(self, x, spline_params, base_weights):
        """
        Forward pass through one KAN layer (vectorized).

        Args:
            x: Input activations (in_features,)
            spline_params: Basis coefficients (in_features, out_features, grid_size)
            base_weights: Residual weights (in_features, out_features)

        Returns:
            Output activations (out_features,)
        """
        in_features, out_features, grid_size = spline_params.shape

        # Vectorized computation of all basis functions
        # basis_values: (in_features, grid_size)
        basis_values = jax.vmap(self._compute_basis)(x)

        # Compute edge functions: einsum over basis dimension
        # spline_params: (in_features, out_features, grid_size)
        # basis_values: (in_features, grid_size)
        # result: (out_features,)
        edge_outputs = jnp.einsum('ijk,ik->j', spline_params, basis_values)

        # Add residual connection
        residual = jnp.dot(x, base_weights)

        return edge_outputs + residual

    @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """
        Full forward pass through KAN.

        Args:
            params: List of (spline_params, base_weights) tuples for each layer
            x_idata: Input data (input_dim,)

        Returns:
            Output predictions (output_dim,)
        """
        activations = x_idata

        # Pass through all layers
        for spline_params, base_weights in params:
            activations = self._layer_forward(activations, spline_params, base_weights)

        return activations

    v_forward = jax.vmap(forward, in_axes=(None, None, 0))  # returns (ndata, var)
    vv_forward = jax.vmap(v_forward, in_axes=(None, None, 0))  # returns (ndata, var)
    g_forward = jax.jacrev(forward, argnums=2)  # returns (var, dim)
    gg_forward = jax.jacfwd(g_forward, argnums=2)  # returns (var, dim, dim)
    v_g_forward = jax.vmap(g_forward, in_axes=(None, None, 0))  # returns (ndata, var, dim)
    vv_g_forward = jax.vmap(v_g_forward, in_axes=(None, None, 0))  # returns (ndata, var, dim)


## Fourier Neural Operator (FNO)

class FNO:
    """
    Fourier Neural Operator implementation in pure JAX.
    Learns operators in Fourier space for efficient function approximation.

    Architecture:
    - Lifting layer: maps input to hidden dimension
    - Fourier layers: spectral convolution in frequency domain
    - Projection layer: maps back to output dimension
    - Purely data-driven, no PDEs required
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=4, modes=16):
        """
        Initialize FNO architecture.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            hidden_dim: Hidden channel dimension (default: 64)
            num_layers: Number of Fourier layers (default: 4)
            modes: Number of Fourier modes to keep (default: 16)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modes = modes

    @partial(jax.jit, static_argnames=['self'])
    def _spectral_conv(self, x, weights):
        """
        Spectral convolution in Fourier space - simplified for 1D inputs.

        Args:
            x: Input (hidden_dim,)
            weights: Fourier weights (modes, hidden_dim) - complex

        Returns:
            Output after spectral convolution (hidden_dim,)
        """
        # Transform to Fourier space
        x_ft = jnp.fft.rfft(x)  # (hidden_dim//2 + 1,)

        # Truncate to relevant modes
        x_ft_truncated = x_ft[:self.modes]  # (modes,)

        # Element-wise multiplication in frequency domain
        # weights: (modes, hidden_dim)
        # x_ft_truncated: (modes,) -> expand for matrix multiply
        out_ft = jnp.dot(jnp.expand_dims(x_ft_truncated, axis=0), weights)  # (1, hidden_dim)
        out_ft = jnp.squeeze(out_ft, axis=0)  # (hidden_dim,)

        # For inverse FFT, we need to pad/truncate to match input size
        # The output should have size (hidden_dim//2 + 1) for rfft inverse
        n_fft = x_ft.shape[0]
        if out_ft.shape[0] < n_fft:
            # Pad with zeros
            out_ft_padded = jnp.pad(out_ft, (0, n_fft - out_ft.shape[0]), mode='constant')
        else:
            # Truncate
            out_ft_padded = out_ft[:n_fft]

        # Inverse FFT back to physical space
        out = jnp.fft.irfft(out_ft_padded, n=len(x))

        return jnp.real(out)

    @partial(jax.jit, static_argnames=['self'])
    def _fourier_layer(self, x, spectral_weights, linear_weights, bias):
        """
        One Fourier layer: spectral convolution + linear transform + activation.

        Args:
            x: Input (hidden_dim,)
            spectral_weights: Fourier domain weights (modes, hidden_dim)
            linear_weights: Physical domain weights (hidden_dim, hidden_dim)
            bias: Bias term (hidden_dim,)

        Returns:
            Output (hidden_dim,)
        """
        # Spectral path
        x_spectral = self._spectral_conv(x, spectral_weights)

        # Linear path (local features)
        x_linear = jnp.dot(x, linear_weights)

        # Combine both paths
        out = x_spectral + x_linear + bias

        # Activation
        out = jax.nn.gelu(out)

        return out

    @partial(jax.jit, static_argnames=['self'])
    def forward(self, params, x_idata):
        """
        Full forward pass through FNO.

        Args:
            params: Dictionary with keys:
                - 'lift': (input_dim, hidden_dim) - lifting weights
                - 'lift_bias': (hidden_dim,) - lifting bias
                - 'fourier_layers': List of (spectral_weights, linear_weights, bias) tuples
                - 'project': (hidden_dim, output_dim) - projection weights
                - 'project_bias': (output_dim,) - projection bias
            x_idata: Input data (input_dim,)

        Returns:
            Output predictions (output_dim,)
        """
        # Lifting layer: input_dim -> hidden_dim
        x = jnp.dot(x_idata, params['lift']) + params['lift_bias']

        # Fourier layers
        for spectral_weights, linear_weights, bias in params['fourier_layers']:
            x = self._fourier_layer(x, spectral_weights, linear_weights, bias)

        # Projection layer: hidden_dim -> output_dim
        out = jnp.dot(x, params['project']) + params['project_bias']

        return out

    v_forward = jax.vmap(forward, in_axes=(None, None, 0))  # returns (ndata, var)
    vv_forward = jax.vmap(v_forward, in_axes=(None, None, 0))  # returns (ndata, var)
    g_forward = jax.jacrev(forward, argnums=2)  # returns (var, dim)
    gg_forward = jax.jacfwd(g_forward, argnums=2)  # returns (var, dim, dim)
    v_g_forward = jax.vmap(g_forward, in_axes=(None, None, 0))  # returns (ndata, var, dim)
    vv_g_forward = jax.vmap(v_g_forward, in_axes=(None, None, 0))  # returns (ndata, var, dim)