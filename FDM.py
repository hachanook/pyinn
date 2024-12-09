import numpy as np
# import tensorly as tl
# from tensorly.decomposition import parafac, tucker
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=4, suppress=True)



### Finite Difference Method for heat conduction

# Parameters
dx = 0.02  # Spatial step size in x direction
dy = 0.02  # Spatial step size in y direction
dt = 0.0001  # Time step size
total_time = 1.0  # Total simulation time
alpha = 0.01  # Thermal diffusivity (material property)

# Grid setup
nx = int(1 / dx) + 1  # Number of points in x
ny = int(1 / dy) + 1  # Number of points in y
time_steps = int(total_time / dt)  # Number of time steps

# Initialize temperature field (2D array)
T = np.zeros((nx, ny))  # Current temperature
T_new = np.zeros_like(T)  # Temperature for the next time step
T_tensor = np.zeros((nx,ny,time_steps)) # Final tensor

# Function to define the source term
# Moving source: a Gaussian peak that moves from x=0 to x=1 at y=0.5
def moving_source(x, t):
    source_position = t  # Linear motion from x=0 to x=1
    sigma = 0.05  # Spread of the Gaussian
    return np.exp(-((x - source_position)**2) / (2 * sigma**2))

# Time stepping loop
for n in range(time_steps):
    if n % 10 == 0: 
        print(f"{n}/{time_steps}")
    t = n * dt

    # Update the temperature field using finite difference method
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            T_new[i, j] = T[i, j] + alpha * dt * (
                (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dx**2 +
                (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dy**2
            )

    # Add the moving source term
    source_x = int(t / dx)
    source_y = int(0.5 / dy)
    if 0 < source_x < nx:
        T_new[source_x, source_y] += moving_source(source_x * dx, t) * dt

    # Swap temperature arrays
    T_tensor[:,:,n] = T_new
    T, T_new = T_new, T

    # # Visualization during simulation (optional)
    # if n % 50 == 0:  # Plot every 500 time steps
    #     plt.clf()
    #     plt.imshow(T, extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
    #     plt.colorbar(label='Temperature')
    #     plt.title(f"Time = {t:.3f} s")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.pause(0.01)

# Save T_tensor
# parent_dir = os.path.abspath(os.getcwd())
#     path_data = os.path.join(parent_dir, 'data')
parent_dir = os.getcwd()
save_dir = os.path.join(parent_dir, 'data')
file_name = f"FDM_{nx}_{ny}_{time_steps}.npy"
file_path = os.path.join(save_dir, file_name)
np.save(file_path, T_tensor)

# Final plot
plt.figure()
plt.imshow(T_tensor[:,:,int(0.2*time_steps)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
plt.colorbar(label='Temperature')
plt.title("Final Temperature Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()

plt.figure()
plt.imshow(T_tensor[:,:,int(0.5*time_steps)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
plt.colorbar(label='Temperature')
plt.title("Final Temperature Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()


plt.figure()
plt.imshow(T_tensor[:,:,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
plt.colorbar(label='Temperature')
plt.title("Final Temperature Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()
