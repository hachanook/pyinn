import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.colors import Normalize
from tqdm import tqdm

# GPU Configuration - Set to use only GPU index 0
gpu_idx = 1  # set which GPU to run on
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing  

def create_LAM_data_animation(use_normalized_concentration=True):
    """
    Create an animation of the LAM concentration data showing time evolution.
    
    Reads data from ../data/all_concentration_data.csv and creates a scatter plot
    animation showing concentration_h2 evolution over time steps 1-100.
    
    Args:
        use_normalized_concentration (bool): If True, plot normalized H2 concentration [0,1].
                                           If False, plot original H2 concentration values.
    
    The animation shows:
    - Scatter plot with z-coordinate on x-axis and y-coordinate on y-axis
    - Color-coded concentration_h2 values using 'jet' colormap
    - Time evolution from t=1 to t=100
    - Consistent colorbar and axis limits across all frames
    
    Returns:
        matplotlib.animation.FuncAnimation: The animation object
    """
    
    # Read the data
    print("Reading LAM concentration data...")
    # data_path = "./data/all_concentration_data.csv"
    data_path = "./data/all_concentration_data_test.csv"   
    # data_path = "./data/all_concentration_data_Case_6.csv"
    df = pd.read_csv(data_path)
    
    # Verify column names
    # expected_columns = ['y', 'z', 't', 'concentration_h2']
    expected_columns = ['y', 'z', 't', 'sample_value', 'concentration_h2']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Expected columns {expected_columns}, found {list(df.columns)}")
    
    # Get time range
    time_steps = sorted(df['t'].unique())
    print(f"Found {len(time_steps)} time steps: {min(time_steps)} to {max(time_steps)}")
    
    # Normalize data to [0,1] range
    print("Normalizing data to [0,1] range...")
    
    # Get original ranges for normalization
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()
    # concentration_min = float(df['concentration_h2'].min())
    # concentration_max = float(df['concentration_h2'].max())
    concentration_min = 0.0
    concentration_max = 3.99e-2
    
    # Create normalized columns
    df['y_norm'] = (df['y'] - y_min) / (y_max - y_min)
    df['z_norm'] = (df['z'] - z_min) / (z_max - z_min)
    df['concentration_h2_norm'] = (df['concentration_h2'] - concentration_min) / (concentration_max - concentration_min)
    
    
    # Print original and normalized ranges
    print(f"Flow rate of inlet 2: {df['sample_value'].min()}")
    print(f"Original coordinate ranges:")
    print(f"  y: [{y_min:.6f}, {y_max:.6f}]")
    print(f"  z: [{z_min:.6f}, {z_max:.6f}]")
    print(f"  concentration_h2: [{concentration_min:.2e}, {concentration_max:.2e}]")
    print(f"Normalized ranges (all [0,1]):")
    print(f"  y_norm: [{df['y_norm'].min():.6f}, {df['y_norm'].max():.6f}]")
    print(f"  z_norm: [{df['z_norm'].min():.6f}, {df['z_norm'].max():.6f}]")
    print(f"  concentration_h2_norm: [{df['concentration_h2_norm'].min():.6f}, {df['concentration_h2_norm'].max():.6f}]")
    
    # Print data statistics
    print(f"Total data points: {len(df):,}")
    print(f"Time range: [{df['t'].min():.0f}, {df['t'].max():.0f}]")
    
    # Set up the figure with more height to accommodate larger labels
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Initialize scatter plot
    marker_size = 100
    scatter = ax.scatter([], [], c=[], cmap='jet', s=marker_size, alpha=0.7, edgecolors='none')
    
    # Set up colorbar based on concentration type
    if use_normalized_concentration:
        norm = Normalize(vmin=0, vmax=1)
        cbar_label = 'Normalized Concentration H2'
        title_suffix = 'Normalized H2 Concentration Evolution'
    else:
        norm = Normalize(vmin=concentration_min, vmax=concentration_max)
        cbar_label = 'Concentration H2'
        title_suffix = 'H2 Concentration Evolution'
    
    scatter.set_norm(norm)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=20)
    
    # Set labels and title
    ax.set_xlabel('Normalized z', fontsize=20)
    ax.set_ylabel('Normalized y', fontsize=20)
    # ax.set_title(f'LAM Data: {title_suffix}', fontsize=14)
    
    # Set axis limits for normalized coordinates [0,1]
    ax.set_xlim(-0.05, 1.05)  # Small padding around [0,1]
    ax.set_ylim(-0.05, 1.05)  # Small padding around [0,1]
    
    # Text to show current time step
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=18,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate(frame):
        """Animation function for matplotlib FuncAnimation."""
        t = time_steps[frame]
        
        # Filter data for current time step
        time_data = df[df['t'] == t]
        
        # Update scatter plot data with normalized coordinates
        scatter.set_offsets(np.column_stack([time_data['z_norm'], time_data['y_norm']]))
        
        # Use appropriate concentration values based on the boolean flag
        if use_normalized_concentration:
            scatter.set_array(time_data['concentration_h2_norm'])
        else:
            scatter.set_array(time_data['concentration_h2'])
        
        # Ensure marker properties are maintained
        scatter.set_sizes([marker_size] * len(time_data))  # Set marker size for all points
        scatter.set_edgecolor('none')  # Remove marker edges
        
        # Update time text
        # time_text.set_text(f'Time Step: {int(t)}')
        time_text.set_text(f'Time: {float(t)*0.001:.3f} s') # in the original time domain
        
        return [scatter, time_text]
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(time_steps),
                                 interval=100, blit=True, repeat=True)
    
    # Show progress during saving
    print("Saving animation (this may take a while)...")
    
    # Ensure output directory exists
    output_suffix = "normalized" if use_normalized_concentration else "original"
    output_path = f"./plots/LAM_data_animation_{output_suffix}.gif"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    
    print(f"Animation saved successfully!")
    print(f"Animation shows {len(time_steps)} time steps from {min(time_steps)} to {max(time_steps)}")
    print(f"Data points per frame: ~{len(df) // len(time_steps)}")
    print(f"Concentration type: {'Normalized' if use_normalized_concentration else 'Original'}")
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return anim

if __name__ == "__main__":
    # Create both versions of the animation
    # print("Creating animation with normalized H2 concentration...")
    # animation_obj_normalized = create_LAM_data_animation(use_normalized_concentration=True)
    
    print("\nCreating animation with original H2 concentration...")
    animation_obj_original = create_LAM_data_animation(use_normalized_concentration=False)
    
    plt.show() 