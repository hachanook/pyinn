import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os

# Create GIF with normalized concentration values (0 to 1)
# Load your CSV
df_norm = pd.read_csv("all_concentration_data_Case_7.csv")

# Get unique time steps
time_steps_norm = sorted(df_norm['t'].unique())

# Calculate global min and max concentration for normalization
global_vmin_orig = df_norm['concentration_h2'].min()
global_vmax_orig = df_norm['concentration_h2'].max()

# Normalize concentration values to 0-1 range
df_norm['concentration_h2_normalized'] = (df_norm['concentration_h2'] - global_vmin_orig) / (global_vmax_orig - global_vmin_orig)

print(f"Original concentration range: Min = {global_vmin_orig:.2e}, Max = {global_vmax_orig:.2e}")
print(f"Normalized concentration range: Min = {df_norm['concentration_h2_normalized'].min():.3f}, Max = {df_norm['concentration_h2_normalized'].max():.3f}")

# Create a folder to store normalized frames
os.makedirs("frames_normalized", exist_ok=True)

filenames_norm = []

for i, t in enumerate(time_steps_norm):
    # Filter data for the current time step
    df_t_norm = df_norm[df_norm['t'] == t]
    
    if len(df_t_norm) == 0:
        print(f"No data for timestep {t}")
        continue
    
    # Use 0-1 range for normalized color scaling
    vmin = 0.0
    vmax = 1.0

    plt.figure(figsize=(10, 6))
    
    # Create scatter plot with color mapping using normalized values
    scatter = plt.scatter(df_t_norm['z'], df_t_norm['y'], c=df_t_norm['concentration_h2_normalized'], 
                         cmap='jet', vmin=vmin, vmax=vmax, s=20, alpha=0.8)
    
    plt.title(f"H2 Concentration (Normalized) at t = {t}")
    plt.xlabel("z-coordinate")
    plt.ylabel("y-coordinate")
    
    # Add colorbar with normalized range
    cbar = plt.colorbar(scatter, label="Normalized Concentration")
    cbar.set_label(f"H2 Concentration (Normalized Range: 0.0 - 1.0)")
    
    # Set axis limits based on data with small padding
    z_range = df_t_norm['z'].max() - df_t_norm['z'].min()
    y_range = df_t_norm['y'].max() - df_t_norm['y'].min()
    
    plt.xlim(df_t_norm['z'].min() - 0.05 * z_range, df_t_norm['z'].max() + 0.05 * z_range)
    plt.ylim(df_t_norm['y'].min() - 0.05 * y_range, df_t_norm['y'].max() + 0.05 * y_range)
    
    # Set equal aspect ratio for natural scaling
    plt.axis('equal')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    fname = f"frames_normalized/frame_{i:03d}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    filenames_norm.append(fname)
    
    # Show current timestep range for comparison
    timestep_min = df_t_norm['concentration_h2_normalized'].min()
    timestep_max = df_t_norm['concentration_h2_normalized'].max()
    print(f"Frame {i+1}/{len(time_steps_norm)} saved for t={t} (Normalized range: {timestep_min:.3f} - {timestep_max:.3f})")

# Create GIF with normalized values
if filenames_norm:
    with imageio.get_writer('concentration_evolution_normalized.gif', mode='I', duration=0.5) as writer:
        for fname in filenames_norm:
            image = imageio.imread(fname)
            writer.append_data(image)
    
    print(f"Normalized GIF saved as concentration_evolution_normalized.gif with {len(filenames_norm)} frames")
    print(f"All frames use normalized color scale: 0.0 (blue) to 1.0 (red)")
    print(f"Original data range was: {global_vmin_orig:.2e} to {global_vmax_orig:.2e}")
else:
    print("No normalized frames were created!")