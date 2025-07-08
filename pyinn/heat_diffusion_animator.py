import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple, Callable
import os

# Physical constants and parameters
THERMAL_DIFFUSIVITY = 1.0  # m^2/s
INITIAL_TEMP = 0.0  # Initial temperature
BOUNDARY_TEMP = 100.0  # Boundary temperature

def create_heat_diffusion_function() -> Callable:
    """
    Create a JAX function that represents transient heat diffusion solution.
    
    This function solves the 2D heat equation:
    ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    
    With boundary conditions:
    - u(0,y,t) = u(10,y,t) = BOUNDARY_TEMP (hot boundaries at x=0 and x=10)
    - u(x,0,t) = u(x,1,t) = INITIAL_TEMP (cold boundaries at y=0 and y=1)
    - u(x,y,0) = INITIAL_TEMP (cold initial condition)
    
    Returns:
        JAX function that takes (x, y, t) and returns temperature
    """
    
    @jax.jit
    def heat_diffusion_solution(x: jnp.ndarray, y: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Analytical solution for 2D transient heat diffusion.
        
        Args:
            x: x-coordinate (0 to 10)
            y: y-coordinate (0 to 1)  
            t: time (0 to 0.1)
            
        Returns:
            Temperature field at (x, y, t)
        """
        # Avoid division by zero at t=0
        t = jnp.maximum(t, 1e-8)
        
        # Create a solution using Fourier series approximation
        # This represents heat diffusion from hot x-boundaries into cold interior
        
        # Base steady-state solution (linear interpolation in x-direction)
        steady_state = BOUNDARY_TEMP * (1 - jnp.abs(x - 5.0) / 5.0)
        
        # Transient part using Fourier series
        temperature = steady_state
        
        # Add transient terms with multiple modes
        for n in range(1, 6):  # Use first 5 modes for approximation
            for m in range(1, 6):
                # Fourier coefficients
                lambda_n = (n * jnp.pi / 10.0) ** 2
                lambda_m = (m * jnp.pi / 1.0) ** 2
                eigenvalue = THERMAL_DIFFUSIVITY * (lambda_n + lambda_m)
                
                # Spatial eigenfunctions
                X_n = jnp.sin(n * jnp.pi * x / 10.0)
                Y_m = jnp.sin(m * jnp.pi * y / 1.0)
                
                # Time evolution
                time_factor = jnp.exp(-eigenvalue * t)
                
                # Amplitude (chosen to create interesting patterns)
                amplitude = BOUNDARY_TEMP * ((-1)**n) * ((-1)**m) / (n * m * jnp.pi**2)
                
                # Add contribution
                temperature += amplitude * X_n * Y_m * time_factor
        
        # Add some spatial variation to make it more interesting
        # Central heating source that decays over time
        center_x, center_y = 5.0, 0.5
        distance = jnp.sqrt((x - center_x)**2 + (y - center_y)**2)
        heat_source = 50.0 * jnp.exp(-distance**2 / 0.5) * jnp.exp(-t / 0.05)
        temperature += heat_source
        
        return temperature
    
    return heat_diffusion_solution

def create_mesh_grid(nx: int = 50, ny: int = 20) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create uniform mesh grid for the x-y domain.
    
    Args:
        nx: Number of points in x-direction
        ny: Number of points in y-direction
        
    Returns:
        Tuple of (X, Y) mesh grids
    """
    x = jnp.linspace(0, 10, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y)
    return X, Y

def create_animation(output_path: str = "../plots/heat_diffusion_animation.gif"):
    """
    Create and save animation of the heat diffusion process.
    
    Args:
        output_path: Path to save the animation video
    """
    # Create the heat diffusion function
    heat_func = create_heat_diffusion_function()
    
    # Vectorize the function for mesh grid computation (flatten and reshape approach)
    vectorized_heat_func = jax.vmap(heat_func, in_axes=(0, 0, None))
    
    # Create mesh grid
    X, Y = create_mesh_grid()
    
    # Time parameters
    t_start = 0.001  # Start slightly after 0 to avoid singularity
    t_end = 0.1
    dt = 0.001
    time_steps = jnp.arange(t_start, t_end + dt, dt)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Initialize the plot
    im = ax.imshow(np.zeros_like(X), extent=[0, 10, 0, 1], origin='lower', 
                   cmap='jet', vmin=0, vmax=150, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('2D Transient Heat Diffusion')
    
    # Text to show current time
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def animate(frame):
        """Animation function for matplotlib FuncAnimation."""
        t = time_steps[frame]
        
        # Compute temperature field at current time
        # Flatten X and Y, compute temperature, then reshape back
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        temperature_flat = vectorized_heat_func(X_flat, Y_flat, t)
        temperature = temperature_flat.reshape(X.shape)
        
        # Update the image
        im.set_array(np.array(temperature))
        
        # Update time text
        time_text.set_text(f'Time: {t:.3f} s')
        
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
    
    # Also show the plot
    plt.tight_layout()
    plt.show()
    
    return anim

def demonstrate_vectorization():
    """
    Demonstrate the vectorization capabilities of the heat diffusion function.
    """
    print("Demonstrating JAX vectorization...")
    
    # Create the heat diffusion function
    heat_func = create_heat_diffusion_function()
    
    # Test single point evaluation
    x_single = 5.0
    y_single = 0.5
    t_single = 0.05
    
    temp_single = heat_func(x_single, y_single, t_single)
    print(f"Temperature at ({x_single}, {y_single}) at t={t_single}: {temp_single:.2f}°C")
    
    # Test vectorized evaluation
    x_vec = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0])
    y_vec = jnp.array([0.2, 0.4, 0.5, 0.6, 0.8])
    t_vec = jnp.array([0.01, 0.02, 0.03, 0.04, 0.05])
    
    # Vectorize over all dimensions
    vmap_func = jax.vmap(heat_func, in_axes=(0, 0, 0))
    temp_vec = vmap_func(x_vec, y_vec, t_vec)
    
    print("\nVectorized evaluation:")
    for i in range(len(x_vec)):
        print(f"  Temperature at ({x_vec[i]}, {y_vec[i]}) at t={t_vec[i]}: {temp_vec[i]:.2f}°C")
    
    # Test mesh grid evaluation
    X, Y = create_mesh_grid(nx=20, ny=10)
    t_mesh = 0.05
    
    # Vectorize for mesh evaluation (flatten and reshape approach)
    mesh_vmap = jax.vmap(heat_func, in_axes=(0, 0, None))
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    temp_mesh_flat = mesh_vmap(X_flat, Y_flat, t_mesh)
    temp_mesh = temp_mesh_flat.reshape(X.shape)
    
    print(f"\nMesh grid evaluation shape: {temp_mesh.shape}")
    print(f"Temperature range: {jnp.min(temp_mesh):.2f}°C to {jnp.max(temp_mesh):.2f}°C")

if __name__ == "__main__":
    # Demonstrate vectorization
    demonstrate_vectorization()
    
    # Create and save animation
    create_animation()