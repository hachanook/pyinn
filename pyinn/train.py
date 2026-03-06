"""
INN Trainer Module
----------------------------------------------------------------------------------
Efficient training implementations for INN and other neural network models.
Uses NumPy arrays with batch-wise JAX transfer for optimal performance.

Copyright (C) 2024  Chanwook Park
Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", False)  # Use float32 for better GPU performance
import optax
from functools import partial
import numpy as np
import time
import sys

# Local imports (development mode - no package installation)
from model import INN_linear, INN_nonlinear, INN_gaussian, MLP, KAN, FNO


def get_linspace(xmin, xmax, nnode):
    """Create linearly spaced grid points."""
    return jnp.linspace(xmin, xmax, nnode, dtype=jnp.float32)


class Regression_INN:
    """
    INN (Interpolating Neural Network) trainer for regression tasks.

    Supports both linear and nonlinear interpolation methods with CP tensor decomposition.
    Uses efficient mini-batch training with NumPy arrays and batch-wise JAX transfer.
    """

    def __init__(self, cls_data, config):
        """
        Initialize INN trainer.

        Args:
            cls_data: Data class instance with NumPy arrays (x_data_train, u_data_train, etc.)
            config: Configuration dictionary with model and training parameters
        """
        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = int(time.time())
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_INN'])
        self.scale = 1

        # Initialize trainable parameters for INN
        if 'linear' in self.interp_method or self.interp_method in ['nonlinear', 'gaussian']:
            self.nmode = int(config['MODEL_PARAM']['nmode'])

            if isinstance(config['MODEL_PARAM']['nseg'], int):
                # Same discretization across all dimensions
                self.nseg = int(config['MODEL_PARAM']['nseg'])
                self.nnode = self.nseg + 1

                # Grid initialization
                if cls_data.bool_normalize:
                    self.grid_dms = jnp.linspace(0, 1, self.nnode, dtype=jnp.float32)
                else:
                    self.grid_dms = [get_linspace(xmin, xmax, self.nnode)
                                     for (xmin, xmax) in zip(cls_data.x_data_minmax["min"],
                                                             cls_data.x_data_minmax["max"])]

                # Parameter initialization
                self.params = jax.random.uniform(
                    jax.random.PRNGKey(self.key),
                    (self.nmode, self.cls_data.dim, self.cls_data.var, self.nnode),
                    dtype=jnp.float32
                ) / self.scale
                numParam = self.nmode * self.cls_data.dim * self.cls_data.var * self.nnode

            elif isinstance(config['MODEL_PARAM']['nseg'], list):
                # Varying discretization across dimensions
                self.nseg = jnp.array(config['MODEL_PARAM']['nseg'], dtype=jnp.int32)
                self.nnode = self.nseg + 1

                if len(self.nseg) != cls_data.dim:
                    print(f"Error: length of nseg {len(self.nseg)} differs from input dim {cls_data.dim}")
                    sys.exit()

                self.grid_dms, self.params, numParam = [], [], 0
                for idm, nnode_idm in enumerate(self.nnode):
                    if cls_data.bool_normalize:
                        self.grid_dms.append(jnp.linspace(0, 1, nnode_idm, dtype=jnp.float32))
                    else:
                        self.grid_dms.append(get_linspace(
                            cls_data.x_data_minmax["min"][idm],
                            cls_data.x_data_minmax["max"][idm],
                            nnode_idm
                        ))
                    self.params.append(jax.random.uniform(
                        jax.random.PRNGKey(self.key),
                        (self.nmode, self.cls_data.var, nnode_idm),
                        dtype=jnp.float32
                    ) / self.scale)
                    numParam += self.nmode * self.cls_data.var * nnode_idm

        # Create model
        if self.interp_method == "linear":
            model = INN_linear(self.grid_dms, self.config)
        elif self.interp_method == "nonlinear":
            model = INN_nonlinear(self.grid_dms, self.config)
        elif self.interp_method == "gaussian":
            model = INN_gaussian(self.grid_dms, self.config)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interp_method}")

        self.forward = model.forward
        self.v_forward = model.v_forward
        self.vv_forward = model.vv_forward
        self.get_std_dim = model.get_std_dim

        # Print model info
        if self.interp_method == "linear":
            print(f"------------ INN {config['TD_type']} {self.interp_method}, "
                  f"nmode: {config['MODEL_PARAM']['nmode']}, nseg: {config['MODEL_PARAM']['nseg']} -------------")
        elif self.interp_method == "nonlinear":
            print(f"------------ INN {config['TD_type']} {self.interp_method}, "
                  f"nmode: {config['MODEL_PARAM']['nmode']}, nseg: {config['MODEL_PARAM']['nseg']}, "
                  f"s={config['MODEL_PARAM']['s_patch']}, P={config['MODEL_PARAM']['p_order']} -------------")
        elif self.interp_method == "gaussian":
            sigma_factor = config['MODEL_PARAM'].get('sigma_factor', 1.2)
            print(f"------------ INN {config['TD_type']} {self.interp_method}, "
                  f"nmode: {config['MODEL_PARAM']['nmode']}, nseg: {config['MODEL_PARAM']['nseg']}, "
                  f"sigma_factor={sigma_factor} -------------")
        print(f"# of training parameters: {numParam}")

    @partial(jax.jit, static_argnames=['self']) ## We need this!
    def _loss_fn(self, params, x_data, u_data):
        """Compute MSE loss (JIT-compiled)."""
        u_pred = self.v_forward(params, x_data)

        ## Case 1: with normalization of CPD
        # std_dim = self.get_std_dim(params)
        # loss = jnp.mean((u_pred - u_data) ** 2) + jnp.mean(std_dim)

        ## Case 2: without normalization of CPD
        loss = jnp.mean((u_pred - u_data) ** 2)

        return loss, u_pred


    @partial(jax.jit, static_argnames=['self'])
    def _update_step(self, params, opt_state, x_data, u_data):
        """Single optimization step (JIT-compiled)."""
        (loss, u_pred), grads = jax.value_and_grad(
            self._loss_fn, argnums=0, has_aux=True
        )(params, x_data, u_data)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self):
        """
        Train the model with efficient mini-batch gradient descent.

        Training loop optimized for performance:
        - NumPy arrays with batch-wise JAX transfer
        - Accumulated epoch_loss (no extra forward passes)
        - Single validation loss computation per validation period
        - Early stopping based on validation loss
        """
        # Training configuration
        self.learning_rate = float(self.config['TRAIN_PARAM']['learning_rate'])
        self.validation_period = int(self.config['TRAIN_PARAM']['validation_period'])
        self.patience = int(self.config['TRAIN_PARAM']['patience'])
        self.batch_size = int(self.cls_data.batch_size)

        # Get data as NumPy arrays
        X_train = self.cls_data.x_data_train
        Y_train = self.cls_data.u_data_train
        X_val = self.cls_data.x_data_val
        Y_val = self.cls_data.u_data_val
        n_train = len(X_train)

        # Convert validation data to JAX once (small dataset, keep on GPU)
        X_val_jax = jnp.array(X_val)
        Y_val_jax = jnp.array(Y_val)

        # Initialize optimizer
        params = self.params
        self.optimizer = optax.adam(self.learning_rate)
        opt_state = self.optimizer.init(params)

        # Training history
        self.loss_history = []
        self.errors_train = []
        self.errors_val = []
        self.errors_epoch = []

        # Early stopping state
        best_val_loss = float('inf')
        patience_counter = 0

        # Batch setup
        batch_size = min(self.batch_size, n_train)
        n_batches = (n_train + batch_size - 1) // batch_size
        rng = np.random.default_rng(42)

        start_time = time.time()
        data_transfer_time = 0.0
        update_step_time = 0.0
        validation_time = 0.0

        for epoch in range(self.num_epochs):
            # Shuffle training data (NumPy operation)
            start_time_data_transfer = time.time()
            perm = rng.permutation(n_train)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]
            data_transfer_time += time.time() - start_time_data_transfer

            epoch_loss = 0.0

            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_train)

                # Transfer batch from NumPy to JAX
                start_time_data_transfer = time.time()
                X_batch = jnp.array(X_shuffled[start_idx:end_idx])
                Y_batch = jnp.array(Y_shuffled[start_idx:end_idx])
                data_transfer_time += time.time() - start_time_data_transfer

                # Training step
                start_time_update_step = time.time()
                params, opt_state, batch_loss = self._update_step(params, opt_state, X_batch, Y_batch)
                epoch_loss += float(batch_loss) * (end_idx - start_idx)
                update_step_time += time.time() - start_time_update_step

                # ################### Debugging ###################
                # std_dim = self.get_std_dim(params)
                # print(jnp.mean(std_dim))
                # ##################################################
                

            # Average training loss for epoch
            epoch_loss /= n_train # Mean squared error
            self.loss_history.append(epoch_loss)

            # Validation at specified intervals
            start_time_validation = time.time()
            if (epoch + 1) % self.validation_period == 0:
                # Compute validation loss (single forward pass)
                val_loss, _ = self._loss_fn(params, X_val_jax, Y_val_jax)
                val_loss = float(val_loss)

                # Store RMSE for history
                train_rmse = float(np.sqrt(epoch_loss))
                val_rmse = float(np.sqrt(val_loss))
                self.errors_train.append(train_rmse)
                self.errors_val.append(val_rmse)
                self.errors_epoch.append(epoch + 1)

                print(f"Epoch {epoch + 1}: Train rmse={train_rmse:.4e}, Val rmse={val_rmse:.4e}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Optional: stopping on training loss threshold
            if "stopping_loss_train" in self.config["TRAIN_PARAM"]:
                if epoch_loss < float(self.config["TRAIN_PARAM"]["stopping_loss_train"]):
                    print(f"Reached target training loss at epoch {epoch + 1}")
                    break

            validation_time += time.time() - start_time_validation

        train_time = time.time() - start_time
        if self.errors_train:
            print(f"Train RMSE: {self.errors_train[-1]:.4e}")
        else:
            print(f"Train RMSE: {float(np.sqrt(self.loss_history[-1])):.4e}")
        self.params = params

        # Final test evaluation
        X_test_jax = jnp.array(self.cls_data.x_data_test)
        Y_test_jax = jnp.array(self.cls_data.u_data_test)
        test_loss, Y_pred = self._loss_fn(params, X_test_jax, Y_test_jax)
        test_rmse = float(np.sqrt(float(test_loss)))
        self.error_test = test_rmse
        print(f"Test rmse: {test_rmse:.4e}")

        print(f"\tTraining completed in {train_time:.2f} seconds")
        print(f"\tData transfer time: {data_transfer_time:.2f} seconds")
        print(f"\tUpdate step time: {update_step_time:.2f} seconds")
        print(f"\tValidation time: {validation_time:.2f} seconds")


        # Inference timing
        self._time_inference()

    def _time_inference(self):
        """Measure inference time."""
        x_sample = jnp.array(self.cls_data.x_data_test[0:1])

        # Warm-up runs
        for _ in range(3):
            _ = self.forward(self.params, x_sample[0])

        # Timed run
        start = time.time()
        _ = self.forward(self.params, x_sample[0])
        print(f"Inference time: {time.time() - start:.6f} seconds")


class Regression_INN_sequential(Regression_INN):
    """Sequential INN training for progressive mode addition."""
    """ CPD noirmalization has not been implemented yet """

    def __init__(self, cls_data, config, params_prev):
        super().__init__(cls_data, config)
        self.params_prev = params_prev

        # Scale initial params to avoid multiplicative explosion
        if isinstance(self.params, list):
            scale = 0.1 ** (1 / len(self.params))
            self.params = [param * scale for param in self.params]
        else:
            scale = 0.1 ** (1 / self.params.shape[1])
            self.params = self.params * scale

    @partial(jax.jit, static_argnames=['self'])
    def _loss_fn(self, params, x_data, u_data):
        """Compute MSE loss with augmented parameters."""
        params_full = jnp.concatenate([self.params_prev, params], axis=0)
        u_pred = self.v_forward(params_full, x_data)
        loss = jnp.mean((u_pred - u_data) ** 2)
        return loss, u_pred

    def _time_inference(self):
        """Measure inference time with full parameters."""
        params_full = jnp.concatenate([self.params_prev, self.params], axis=0)
        x_sample = jnp.array(self.cls_data.x_data_test[0:1])

        for _ in range(3):
            _ = self.forward(params_full, x_sample[0])

        start = time.time()
        _ = self.forward(params_full, x_sample[0])
        print(f"Inference time: {time.time() - start:.6f} seconds")


class Regression_MLP(Regression_INN):
    """MLP trainer for regression tasks."""

    def __init__(self, cls_data, config):
        # Custom initialization (don't call parent __init__ fully)
        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = int(time.time())
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_MLP'])

        # MLP-specific parameters
        activation = config['MODEL_PARAM']["activation"]
        self.nlayers = config['MODEL_PARAM']["nlayers"]
        self.nneurons = config['MODEL_PARAM']["nneurons"]

        # Create model
        model = MLP(activation)
        self.forward = model.forward
        self.v_forward = model.v_forward
        self.vv_forward = model.vv_forward

        # Initialize parameters
        layer_sizes = [cls_data.dim] + self.nlayers * [self.nneurons] + [cls_data.var]
        self.params = self._init_network_params(layer_sizes, jax.random.PRNGKey(self.key))

        # Count parameters
        num_params = sum(w.size + b.size for w, b in self.params)
        print(f"------------ MLP, {layer_sizes} -------------")
        print(f"# of training parameters: {num_params}")

    def _init_network_params(self, layer_sizes, key):
        """Initialize MLP parameters with He initialization."""
        keys = jax.random.split(key, len(layer_sizes) - 1)
        params = []
        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
            weight_key, _ = jax.random.split(k)
            W = jax.random.normal(weight_key, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            b = jnp.zeros(out_dim)
            params.append((W, b))
        return params

    @partial(jax.jit, static_argnames=['self']) ## We need this!
    def _loss_fn(self, params, x_data, u_data):
        """Compute MSE loss (JIT-compiled)."""
        u_pred = self.v_forward(params, x_data)
        loss = jnp.mean((u_pred - u_data) ** 2)
        return loss, u_pred


class Regression_KAN(Regression_INN):
    """KAN (Kolmogorov-Arnold Network) trainer for regression."""

    def __init__(self, cls_data, config):
        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = int(time.time())
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_KAN'])

        # KAN-specific parameters
        self.nlayers = config['MODEL_PARAM']['nlayers']
        self.hidden_dim = config['MODEL_PARAM']['hidden_dim']
        self.grid_size = config['MODEL_PARAM']['grid_size']
        self.spline_order = config['MODEL_PARAM']['spline_order']

        # Create model
        layer_sizes = [cls_data.dim] + self.nlayers * [self.hidden_dim] + [cls_data.var]
        model = KAN(layer_sizes, self.grid_size, self.spline_order)
        self.forward = model.forward
        self.v_forward = model.v_forward
        self.vv_forward = model.vv_forward

        # Initialize parameters
        self.params = self._init_kan_params(layer_sizes, jax.random.PRNGKey(self.key))

        # Count parameters
        num_params = sum(sp.size + bw.size for sp, bw in self.params)
        print(f"------------ KAN, {layer_sizes}, grid_size={self.grid_size}, "
              f"spline_order={self.spline_order} -------------")
        print(f"# of training parameters: {num_params}")

    def _init_kan_params(self, layer_sizes, key):
        """Initialize KAN parameters."""
        num_basis = self.grid_size
        keys = jax.random.split(key, len(layer_sizes) - 1)
        params = []

        for in_dim, out_dim, k in zip(layer_sizes[:-1], layer_sizes[1:], keys):
            spline_key, weight_key = jax.random.split(k)
            spline_params = jax.random.normal(
                spline_key, (in_dim, out_dim, num_basis), dtype=jnp.float32
            ) * 0.1
            base_weights = jax.random.normal(
                weight_key, (in_dim, out_dim), dtype=jnp.float32
            ) * jnp.sqrt(1 / in_dim)
            params.append((spline_params, base_weights))

        return params

    @partial(jax.jit, static_argnames=['self']) ## We need this!
    def _loss_fn(self, params, x_data, u_data):
        """Compute MSE loss (JIT-compiled)."""
        u_pred = self.v_forward(params, x_data)
        loss = jnp.mean((u_pred - u_data) ** 2)
        return loss, u_pred


class Regression_FNO(Regression_INN):
    """FNO (Fourier Neural Operator) trainer for regression."""

    def __init__(self, cls_data, config):
        self.interp_method = config['interp_method']
        self.cls_data = cls_data
        self.config = config
        self.key = int(time.time())
        self.num_epochs = int(self.config['TRAIN_PARAM']['num_epochs_FNO'])

        # FNO-specific parameters
        self.hidden_dim = config['MODEL_PARAM']['hidden_dim']
        self.num_layers = config['MODEL_PARAM']['num_layers']
        self.modes = config['MODEL_PARAM']['modes']

        # Create model
        model = FNO(cls_data.dim, cls_data.var, self.hidden_dim, self.num_layers, self.modes)
        self.forward = model.forward
        self.v_forward = model.v_forward
        self.vv_forward = model.vv_forward

        # Initialize parameters
        self.params = self._init_fno_params(jax.random.PRNGKey(self.key))

        # Count parameters
        num_params = self.params['lift'].size + self.params['lift_bias'].size
        for spectral_w, linear_w, b in self.params['fourier_layers']:
            num_params += spectral_w.size + linear_w.size + b.size
        num_params += self.params['project'].size + self.params['project_bias'].size

        print(f"------------ FNO, hidden_dim={self.hidden_dim}, "
              f"num_layers={self.num_layers}, modes={self.modes} -------------")
        print(f"# of training parameters: {num_params}")

    def _init_fno_params(self, key):
        """Initialize FNO parameters."""
        input_dim = self.cls_data.dim
        output_dim = self.cls_data.var
        keys = jax.random.split(key, 2 + self.num_layers * 3)
        key_idx = 0

        # Lifting layer
        lift = jax.random.normal(
            keys[key_idx], (input_dim, self.hidden_dim), dtype=jnp.float32
        ) * jnp.sqrt(2 / input_dim)
        key_idx += 1
        lift_bias = jnp.zeros(self.hidden_dim, dtype=jnp.float32)

        # Fourier layers
        fourier_layers = []
        for _ in range(self.num_layers):
            spectral_real = jax.random.normal(
                keys[key_idx], (self.modes, self.hidden_dim), dtype=jnp.float32
            ) * 0.02
            key_idx += 1
            spectral_imag = jax.random.normal(
                keys[key_idx], (self.modes, self.hidden_dim), dtype=jnp.float32
            ) * 0.02
            key_idx += 1
            spectral_weights = spectral_real + 1j * spectral_imag

            linear_weights = jax.random.normal(
                keys[key_idx], (self.hidden_dim, self.hidden_dim), dtype=jnp.float32
            ) * jnp.sqrt(2 / self.hidden_dim)
            key_idx += 1

            bias = jnp.zeros(self.hidden_dim, dtype=jnp.float32)
            fourier_layers.append((spectral_weights, linear_weights, bias))

        # Projection layer
        project = jax.random.normal(
            keys[key_idx], (self.hidden_dim, output_dim), dtype=jnp.float32
        ) * jnp.sqrt(2 / self.hidden_dim)
        project_bias = jnp.zeros(output_dim, dtype=jnp.float32)

        return {
            'lift': lift, 'lift_bias': lift_bias,
            'fourier_layers': fourier_layers,
            'project': project, 'project_bias': project_bias
        }
    
    @partial(jax.jit, static_argnames=['self']) ## We need this!
    def _loss_fn(self, params, x_data, u_data):
        """Compute MSE loss (JIT-compiled)."""
        u_pred = self.v_forward(params, x_data)
        loss = jnp.mean((u_pred - u_data) ** 2)
        return loss, u_pred


# =============================================================================
# CLASSIFICATION MODELS
# =============================================================================

class Classification_INN(Regression_INN):
    """INN trainer for classification tasks using cross-entropy loss."""

    def __init__(self, cls_data, config):
        super().__init__(cls_data, config)

        # Classification-specific parameter initialization
        self.params = jax.random.uniform(
            jax.random.PRNGKey(self.key),
            (self.nmode, self.cls_data.dim, self.cls_data.var, self.nnode),
            dtype=jnp.float32, minval=0.98, maxval=1.02
        )

    @partial(jax.jit, static_argnames=['self'])
    def _loss_fn(self, params, x_data, u_data):
        """Compute cross-entropy loss."""
        u_pred = self.v_forward(params, x_data)
        log_probs = u_pred - jax.scipy.special.logsumexp(u_pred, axis=1)[:, None]
        loss = -jnp.mean(jnp.sum(log_probs * u_data, axis=1))
        return loss, u_pred

    def train(self):
        """Train classification model with cross-entropy loss."""
        # Training configuration
        self.learning_rate = float(self.config['TRAIN_PARAM']['learning_rate'])
        self.validation_period = int(self.config['TRAIN_PARAM']['validation_period'])
        self.patience = int(self.config['TRAIN_PARAM']['patience'])
        self.batch_size = int(self.cls_data.batch_size)

        # Get data as NumPy arrays
        X_train = self.cls_data.x_data_train
        Y_train = self.cls_data.u_data_train
        X_val = self.cls_data.x_data_val
        Y_val = self.cls_data.u_data_val
        n_train = len(X_train)

        # Convert validation data to JAX once
        X_val_jax = jnp.array(X_val)
        Y_val_jax = jnp.array(Y_val)

        # Initialize optimizer
        params = self.params
        self.optimizer = optax.adam(self.learning_rate)
        opt_state = self.optimizer.init(params)

        # Training history
        self.loss_history = []
        self.errors_train = []
        self.errors_val = []
        self.errors_epoch = []

        # Early stopping state
        best_val_loss = float('inf')
        patience_counter = 0

        # Batch setup
        batch_size = min(self.batch_size, n_train)
        n_batches = (n_train + batch_size - 1) // batch_size
        rng = np.random.default_rng(42)

        start_time = time.time()
        data_transfer_time = 0.0
        update_step_time = 0.0
        validation_time = 0.0

        for epoch in range(self.num_epochs):
            # Shuffle training data
            perm = rng.permutation(n_train)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]

            epoch_loss = 0.0

            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_train)

                start_time_data_transfer = time.time()
                X_batch = jnp.array(X_shuffled[start_idx:end_idx])
                Y_batch = jnp.array(Y_shuffled[start_idx:end_idx])
                data_transfer_time += time.time() - start_time_data_transfer

                start_time_update_step = time.time()
                params, opt_state, batch_loss = self._update_step(params, opt_state, X_batch, Y_batch)
                update_step_time += time.time() - start_time_update_step
                epoch_loss += float(batch_loss) * (end_idx - start_idx)

            epoch_loss /= n_train
            self.loss_history.append(epoch_loss)

            # Validation
            start_time_validation = time.time()
            if (epoch + 1) % self.validation_period == 0:
                val_loss, val_pred = self._loss_fn(params, X_val_jax, Y_val_jax)
                val_loss = float(val_loss)
                # Compute accuracy
                val_acc = float(jnp.mean(jnp.argmax(val_pred, axis=1) == jnp.argmax(Y_val_jax, axis=1)))

                self.errors_train.append(epoch_loss)
                self.errors_val.append(val_loss)
                self.errors_epoch.append(epoch + 1)

                print(f"Epoch {epoch + 1}: Train loss={epoch_loss:.4e}, Val loss={val_loss:.4e}, Val acc={val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            validation_time += time.time() - start_time_validation
            

        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.2f} seconds")
        print(f"Data transfer time: {data_transfer_time:.2f} seconds")
        print(f"Update step time: {update_step_time:.2f} seconds")
        print(f"Validation time: {validation_time:.2f} seconds")

        self.params = params

        # Test evaluation
        X_test_jax = jnp.array(self.cls_data.x_data_test)
        Y_test_jax = jnp.array(self.cls_data.u_data_test)
        test_loss, test_pred = self._loss_fn(params, X_test_jax, Y_test_jax)
        test_acc = float(jnp.mean(jnp.argmax(test_pred, axis=1) == jnp.argmax(Y_test_jax, axis=1)))
        self.error_test = 1.0 - test_acc
        print(f"Test accuracy: {test_acc:.4f}")

        self._time_inference()


class Classification_MLP(Regression_MLP):
    """MLP trainer for classification tasks."""

    @partial(jax.jit, static_argnames=['self'])
    def _loss_fn(self, params, x_data, u_data):
        """Compute cross-entropy loss."""
        u_pred = self.v_forward(params, x_data)
        log_probs = u_pred - jax.scipy.special.logsumexp(u_pred, axis=1)[:, None]
        loss = -jnp.mean(jnp.sum(log_probs * u_data, axis=1))
        return loss, u_pred

    def train(self):
        """Train MLP classification model."""
        # Use same efficient training pattern as Classification_INN
        self.learning_rate = float(self.config['TRAIN_PARAM']['learning_rate'])
        self.validation_period = int(self.config['TRAIN_PARAM']['validation_period'])
        self.patience = int(self.config['TRAIN_PARAM']['patience'])
        self.batch_size = int(self.cls_data.batch_size)

        X_train = self.cls_data.x_data_train
        Y_train = self.cls_data.u_data_train
        X_val_jax = jnp.array(self.cls_data.x_data_val)
        Y_val_jax = jnp.array(self.cls_data.u_data_val)
        n_train = len(X_train)

        params = self.params
        self.optimizer = optax.adam(self.learning_rate)
        opt_state = self.optimizer.init(params)

        self.loss_history = []
        self.errors_train = []
        self.errors_val = []
        self.errors_epoch = []

        best_val_loss = float('inf')
        patience_counter = 0
        batch_size = min(self.batch_size, n_train)
        n_batches = (n_train + batch_size - 1) // batch_size
        rng = np.random.default_rng(42)

        start_time = time.time()

        for epoch in range(self.num_epochs):
            perm = rng.permutation(n_train)
            X_shuffled = X_train[perm]
            Y_shuffled = Y_train[perm]
            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_train)
                X_batch = jnp.array(X_shuffled[start_idx:end_idx])
                Y_batch = jnp.array(Y_shuffled[start_idx:end_idx])
                params, opt_state, batch_loss = self._update_step(params, opt_state, X_batch, Y_batch)
                epoch_loss += float(batch_loss) * (end_idx - start_idx)

            epoch_loss /= n_train
            self.loss_history.append(epoch_loss)

            if (epoch + 1) % self.validation_period == 0:
                val_loss, val_pred = self._loss_fn(params, X_val_jax, Y_val_jax)
                val_loss = float(val_loss)
                val_acc = float(jnp.mean(jnp.argmax(val_pred, axis=1) == jnp.argmax(Y_val_jax, axis=1)))

                self.errors_train.append(epoch_loss)
                self.errors_val.append(val_loss)
                self.errors_epoch.append(epoch + 1)

                print(f"Epoch {epoch + 1}: Train loss={epoch_loss:.4e}, Val acc={val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        self.params = params

        X_test_jax = jnp.array(self.cls_data.x_data_test)
        Y_test_jax = jnp.array(self.cls_data.u_data_test)
        _, test_pred = self._loss_fn(params, X_test_jax, Y_test_jax)
        test_acc = float(jnp.mean(jnp.argmax(test_pred, axis=1) == jnp.argmax(Y_test_jax, axis=1)))
        self.error_test = 1.0 - test_acc
        print(f"Test accuracy: {test_acc:.4f}")
        self._time_inference()


class Classification_KAN(Regression_KAN):
    """KAN trainer for classification tasks."""

    @partial(jax.jit, static_argnames=['self'])
    def _loss_fn(self, params, x_data, u_data):
        """Compute cross-entropy loss."""
        u_pred = self.v_forward(params, x_data)
        log_probs = u_pred - jax.scipy.special.logsumexp(u_pred, axis=1)[:, None]
        loss = -jnp.mean(jnp.sum(log_probs * u_data, axis=1))
        return loss, u_pred


class Classification_FNO(Regression_FNO):
    """FNO trainer for classification tasks."""

    @partial(jax.jit, static_argnames=['self'])
    def _loss_fn(self, params, x_data, u_data):
        """Compute cross-entropy loss."""
        u_pred = self.v_forward(params, x_data)
        log_probs = u_pred - jax.scipy.special.logsumexp(u_pred, axis=1)[:, None]
        loss = -jnp.mean(jnp.sum(log_probs * u_data, axis=1))
        return loss, u_pred
