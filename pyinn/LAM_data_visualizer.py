import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib.colors import Normalize
from tqdm import tqdm

def create_LAM_data_animation():
    """
    Create an animation of the LAM concentration data showing time evolution.
    
    Reads data from ../data/all_concentration_data.csv and creates a scatter plot
    animation showing concentration_h2 evolution over time steps 1-100.
    
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
    data_path = "./data/all_concentration_data.csv"
    df = pd.read_csv(data_path)
    
    # Verify column names
    expected_columns = ['y', 'z', 't', 'concentration_h2']
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
    concentration_min = float(df['concentration_h2'].min())
    concentration_max = float(df['concentration_h2'].max())
    
    # Create normalized columns
    df['y_norm'] = (df['y'] - y_min) / (y_max - y_min)
    df['z_norm'] = (df['z'] - z_min) / (z_max - z_min)
    df['concentration_h2_norm'] = (df['concentration_h2'] - concentration_min) / (concentration_max - concentration_min)
    
    # Print original and normalized ranges
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
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Initialize scatter plot
    scatter = ax.scatter([], [], c=[], cmap='jet', s=20, alpha=0.7)
    
    # Set up colorbar for normalized values [0,1]
    norm = Normalize(vmin=0, vmax=1)
    scatter.set_norm(norm)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Concentration H2', rotation=270, labelpad=20, fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Normalized z', fontsize=12)
    ax.set_ylabel('Normalized y', fontsize=12)
    ax.set_title('LAM Data: Normalized H2 Concentration Evolution', fontsize=14)
    
    # Set axis limits for normalized coordinates [0,1]
    ax.set_xlim(-0.05, 1.05)  # Small padding around [0,1]
    ax.set_ylim(-0.05, 1.05)  # Small padding around [0,1]
    
    # Text to show current time step
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate(frame):
        """Animation function for matplotlib FuncAnimation."""
        t = time_steps[frame]
        
        # Filter data for current time step
        time_data = df[df['t'] == t]
        
        # Update scatter plot data with normalized values
        scatter.set_offsets(np.column_stack([time_data['z_norm'], time_data['y_norm']]))
        scatter.set_array(time_data['concentration_h2_norm'])
        
        # Update time text
        time_text.set_text(f'Time Step: {int(t)}')
        
        return [scatter, time_text]
    
    # Create animation
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=len(time_steps),
                                 interval=100, blit=True, repeat=True)
    
    # Show progress during saving
    print("Saving animation (this may take a while)...")
    
    # Ensure output directory exists
    output_path = "./plots/LAM_data_animation.gif"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save animation
    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    
    print(f"Animation saved successfully!")
    print(f"Animation shows {len(time_steps)} time steps from {min(time_steps)} to {max(time_steps)}")
    print(f"Data points per frame: ~{len(df) // len(time_steps)}")
    
    return anim

if __name__ == "__main__":
    # Create the animation
    animation_obj = create_LAM_data_animation()
    plt.show() 