import numpy as np
import pickle
import os
import tensorly as tl
from debug_LAM import denormalize_tensor, load_tensors_from_files
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# Read the saved model and extract components
print("="*50)
print("LOADING SAVED MODEL")
print("="*50)

# Load the saved model
# train_filename = 'all_concentration_data_train10'
train_filename = 'all_concentration_data_train20'
model_path = f'./pyinn/model_saved/LAM_{train_filename}.pkl'

## debugging
# test_filename = 'all_concentration_data_test'
# train_tensor, train_nodes, test_tensor, test_nodes = load_tensors_from_files(train_filename, test_filename) # (nnode, ntime, nflow), (nnode, ntime)
    
def scattered_to_grid(scattered_data, coordinates, grid_size=50):
    """
    Convert scattered data to uniform grid data.
    
    Args:
        scattered_data (np.ndarray): Shape (n_nodes, n_time) - data values at scattered points
        coordinates (np.ndarray): Shape (n_nodes, 2) - x,y coordinates of scattered points
        grid_size (int): Size of the output grid (grid_size x grid_size)
        
    Returns:
        np.ndarray: Shape (grid_size, grid_size, n_time) - gridded data
    """
    n_nodes, n_time = scattered_data.shape
    grid_data = np.zeros((grid_size, grid_size, n_time))
    
    # Create uniform grid in [0,1]^2 domain
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    # For each time step
    for t in range(n_time):
        # Use scipy's griddata for interpolation
        from scipy.interpolate import griddata
        grid_data[:, :, t] = griddata(
            coordinates, 
            scattered_data[:, t], 
            (X_grid, Y_grid), 
            method='linear',
            fill_value=0.0
        )
    
    return grid_data, X_grid, Y_grid
    
if os.path.exists(model_path):

    use_normalized_concentration = False


    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Extract components into separate local variables
    train_nodes = loaded_model['train_nodes']
    test_nodes = loaded_model['test_nodes']
    weights = loaded_model['weights']
    factors_test = loaded_model['factors']
    norm_params = loaded_model['norm_params']
    
    time_steps = test_nodes['timesteps']
    node_coords = np.flip(train_nodes['node_coords'], axis=1)
    # Normalize coordinates along axis=0 between 0 and 1
    node_coords = (node_coords - node_coords.min(axis=0)) / (node_coords.max(axis=0) - node_coords.min(axis=0))
    

    reconstructed_test = tl.cp_to_tensor((weights, factors_test)).squeeze()

    # Denormalize the reconstructed test tensor
    reconstructed_test_denorm = denormalize_tensor(reconstructed_test, norm_params)
    # rmse = np.sqrt(np.mean((test_tensor - reconstructed_test_denorm) ** 2))

    print(f"Reconstructed test tensor shape: {reconstructed_test.shape}")
    # print(f"Reconstructed test tensor shape: {reconstructed_test_denorm.shape}")


    # Convert scattered data to grid
    grid_data, X_grid, Y_grid = scattered_to_grid(reconstructed_test_denorm, node_coords, grid_size=200)
    print(f"Grid data shape: {grid_data.shape}")

    # Set up the figure and axis with more height to accommodate larger labels
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Initialize the plot
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 0.0, 1.0
    im = ax.imshow(np.zeros_like(X_grid), extent=(xmin, xmax, ymin, ymax), origin='lower', 
                   cmap='jet', vmin=norm_params['train_min'], vmax=norm_params['train_max'], aspect='auto')
    
    # Set concentration range based on the boolean flag
    if use_normalized_concentration:
        u_min, u_max = 0.0, 1.0
        cbar_label = 'Normalized Concentration H2'
        title_suffix = 'Normalized H2 Concentration Evolution'
    else:
        # Get original concentration range from model data
        u_min = norm_params['train_min']
        u_max = norm_params['train_max']
        cbar_label = 'Concentration H2'
        title_suffix = 'H2 Concentration Evolution'
        
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
        t = test_nodes['timesteps'][frame] # in the normalized time domain
        
        # Compute concentration field at current time
        # Flatten X and Y, compute concentration, then reshape back
        X_flat = X_grid.flatten()
        Y_flat = Y_grid.flatten()
        
        concentration = grid_data[:, :, frame] # note: this only works for var=1 case.
        
        # Update the image
        im.set_array(np.array(concentration))
        
        # Update time text
        time_text.set_text(f'Time: {t*0.001:.3f} s') # in the original time domain
        
        return [im, time_text]
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(time_steps), 
                                 interval=50, blit=True, repeat=True)
    
    # Ensure output directory exists
    output_suffix = "normalized" if use_normalized_concentration else "original"
    output_path = f"./plots/LAM_{train_filename}_{output_suffix}.gif"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save animation
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    
    print(f"Animation saved successfully!")
    print(f"Animation shows {len(time_steps)} time steps from {min(time_steps)} to {max(time_steps)}")
    # print(f"Data points per frame: ~{len(df) // len(time_steps)}")
    print(f"Concentration type: {'Normalized' if use_normalized_concentration else 'Original'}")
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()


    # # Create animation
    # use_normalized_concentration = False
    # time_steps = test_nodes['timesteps']

    # # Set up the figure with more height to accommodate larger labels
    # fig, ax = plt.subplots(figsize=(12, 5))
    
    # # Initialize scatter plot
    # marker_size = 100
    # scatter = ax.scatter([], [], c=[], cmap='jet', s=marker_size, alpha=0.7, edgecolors='none')
    
    # # Set up colorbar based on concentration type
    # if use_normalized_concentration:
    #     norm = Normalize(vmin=0, vmax=1)
    #     cbar_label = 'Normalized Concentration H2'
    #     title_suffix = 'Normalized H2 Concentration Evolution'
    # else:
    #     norm = Normalize(vmin=norm_params['train_min'], vmax=norm_params['train_max'])
    #     cbar_label = 'Concentration H2'
    #     title_suffix = 'H2 Concentration Evolution'
    
    # scatter.set_norm(norm)
    # cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    # cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=20)
    
    # # Set labels and title
    # # ax.set_xlabel('z', fontsize=20)
    # # ax.set_ylabel('y', fontsize=20)
    # ax.set_xlabel('Normalized z', fontsize=20)
    # ax.set_ylabel('Normalized y', fontsize=20)
    # # ax.set_title(f'LAM Data: {title_suffix}', fontsize=14)
    
    # # Set axis limits for normalized coordinates [0,1]
    # ax.set_xlim(-0.05, 1.05)  # Small padding around [0,1]
    # ax.set_ylim(-0.05, 1.05)  # Small padding around [0,1]
    
    # # Text to show current time step
    # time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=18,
    #                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # def animate(frame):
    #     """Animation function for matplotlib FuncAnimation."""
    #     t = time_steps[frame]
        
    #     # Filter data for current time step
    #     # time_data = t / 1000
        
    #     # Update scatter plot data with normalized coordinates
    #     scatter.set_offsets(node_coords)
        
    #     # Use appropriate concentration values based on the boolean flag
    #     if use_normalized_concentration:
    #         scatter.set_array(reconstructed_test[:,frame])
    #     else:
    #         scatter.set_array(reconstructed_test_denorm[:,frame])
    #     # scatter.set_array(test_tensor[:,frame])
        
    #     # Ensure marker properties are maintained
    #     scatter.set_sizes([marker_size] * len(node_coords))  # Set marker size for all points
    #     scatter.set_edgecolor('none')  # Remove marker edges
        
    #     # Update time text
    #     # time_text.set_text(f'Time Step: {int(t)}')
    #     time_text.set_text(f'Time: {float(t)*0.001:.3f} s') # in the original time domain
        
    #     return [scatter, time_text]
    
    # # Create animation
    # print("Creating animation...")
    # anim = animation.FuncAnimation(fig, animate, frames=len(time_steps),
    #                              interval=100, blit=True, repeat=True)
    
    # # Show progress during saving
    # print("Saving animation (this may take a while)...")
    
    # # Ensure output directory exists
    # output_suffix = "normalized" if use_normalized_concentration else "original"
    # output_path = f"./plots/LAM_{train_filename}_{output_suffix}.gif"
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # # Save animation
    # print(f"Saving animation to {output_path}...")
    # anim.save(output_path, writer='pillow', fps=10, dpi=100)
    
    # print(f"Animation saved successfully!")
    # print(f"Animation shows {len(time_steps)} time steps from {min(time_steps)} to {max(time_steps)}")
    # # print(f"Data points per frame: ~{len(df) // len(time_steps)}")
    # print(f"Concentration type: {'Normalized' if use_normalized_concentration else 'Original'}")
    
    # # Adjust layout to prevent label cutoff
    # plt.tight_layout()


    
    