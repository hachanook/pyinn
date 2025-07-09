import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, Callable
import os
from model_utils import load_saved_model, create_model_from_saved_data

# GPU Configuration - Set to use only GPU index 0
gpu_idx = 2  # set which GPU to run on
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing  

def create_mesh_grid(nx: int = 50, ny: int = 20, xmin: float = 0, xmax: float = 10, ymin: float = 0, ymax: float = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create uniform mesh grid for the x-y domain.
    
    Args:
        nx: Number of points in x-direction
        ny: Number of points in y-direction
        
    Returns:
        Tuple of (X, Y) mesh grids
    """
    x = jnp.linspace(xmin, xmax, nx)
    y = jnp.linspace(ymin, ymax, ny)
    X, Y = jnp.meshgrid(x, y)
    return X, Y

def create_animation(use_normalized_concentration=True):
    """ 
    Create and save animation of the heat diffusion process.
    
    Args:
        use_normalized_concentration (bool): If True, plot normalized H2 concentration [0,1].
                                           If False, plot original H2 concentration values.
    """
    # # Create the heat diffusion function
    # heat_func = create_heat_diffusion_function()

    # Load the model
    data_name = "LAM"
    interp_method = "nonlinear"
    # interp_method = "MLP"
    run_type = "regression"
    
    # Set output path based on concentration type
    output_suffix = "normalized" if use_normalized_concentration else "original"
    output_path = f"./plots/{data_name}_{interp_method}_animation_{output_suffix}.gif"
    
    model_data = load_saved_model(data_name, interp_method)
    model = create_model_from_saved_data(model_data, run_type)

    forward = model.forward
    v_forward = model.v_forward

    
    # Vectorize the function for mesh grid computation (flatten and reshape approach)
    # vectorized_heat_func = jax.vmap(heat_func, in_axes=(0, 0, None))
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    tmin, tmax = 0.01, 1.0
    dt = tmin
    
    # Set concentration range based on the boolean flag
    if use_normalized_concentration:
        u_min, u_max = 0.0, 1.0
        cbar_label = 'Normalized Concentration H2'
        title_suffix = 'Normalized H2 Concentration Evolution'
    else:
        # Get original concentration range from model data
        u_min = float(model_data["u_data_minmax"]["min"].item())
        u_max = float(model_data["u_data_minmax"]["max"].item())
        cbar_label = 'Concentration H2'
        title_suffix = 'H2 Concentration Evolution'
    
    # Create mesh grid
    X, Y = create_mesh_grid(nx=200, ny=200, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    
    # Time parameters
    time_steps = jnp.arange(tmin, tmax + dt, dt) # in the normalized time domain
    
    # Set up the figure and axis with more height to accommodate larger labels
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Initialize the plot
    im = ax.imshow(np.zeros_like(X), extent=(xmin, xmax, ymin, ymax), origin='lower', 
                   cmap='jet', vmin=u_min, vmax=u_max, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=20)
    
    # Set labels and title
    ax.set_xlabel('Normalized z', fontsize=20)
    ax.set_ylabel('Normalized y', fontsize=20)
    # ax.set_title(f'LAM Data: {title_suffix}')
    
    # Text to show current time
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=18, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate(frame):
        """Animation function for matplotlib FuncAnimation."""
        t = time_steps[frame] # in the normalized time domain
        
        # Compute temperature field at current time
        # Flatten X and Y, compute temperature, then reshape back
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        XY_flat = jnp.stack([X_flat, Y_flat], axis=1)
        XY_flat = jnp.concatenate([XY_flat, jnp.ones((XY_flat.shape[0], 1)) * t], axis=1) # (ndata, dim=3)

        temperature_flat = v_forward(model_data['params'], XY_flat) # (ndata, var)
        
        # Denormalize if needed
        if not use_normalized_concentration:
            # Denormalize using the min/max values from model data
            u_min_val = model_data["u_data_minmax"]["min"].item()
            u_max_val = model_data["u_data_minmax"]["max"].item()
            temperature_flat = temperature_flat * (u_max_val - u_min_val) + u_min_val
        
        temperature = temperature_flat.reshape(X.shape) # note: this only works for var=1 case.
        
        # Update the image
        im.set_array(np.array(temperature))
        
        # Update time text
        time_text.set_text(f'Time: {t*0.1:.3f} s') # in the original time domain
        
        return [im, time_text]
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(time_steps), 
                                 interval=50, blit=True, repeat=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10)
    
    print(f"Animation saved successfully!")
    print(f"Concentration type: {'Normalized' if use_normalized_concentration else 'Original'}")
    if not use_normalized_concentration:
        print(f"Original concentration range: [{u_min:.2e}, {u_max:.2e}]")
    
    # Also show the plot
    plt.tight_layout()
    plt.show()
    
    return anim


if __name__ == "__main__":
    # # Demonstrate vectorization
    # demonstrate_vectorization()
    
    # Create both versions of the animation
    # print("Creating animation with normalized H2 concentration...")
    # animation_obj_normalized = create_animation(use_normalized_concentration=True)
    
    print("\nCreating animation with original H2 concentration...")
    animation_obj_original = create_animation(use_normalized_concentration=False)