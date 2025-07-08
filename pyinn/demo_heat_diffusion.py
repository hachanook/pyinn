#!/usr/bin/env python3
"""
Demonstration script for the JAX heat diffusion function.

This script shows how to use the heat diffusion function for:
1. Single point evaluation
2. Vectorized evaluation over multiple points
3. Time series analysis at a specific location
4. Creating temperature field snapshots
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from heat_diffusion_animator import create_heat_diffusion_function, create_mesh_grid

def demo_single_point():
    """Demonstrate single point evaluation."""
    print("=== Single Point Evaluation ===")
    heat_func = create_heat_diffusion_function()
    
    # Evaluate at center of domain at different times
    x, y = 5.0, 0.5
    times = [0.001, 0.01, 0.05, 0.1]
    
    print(f"Temperature at ({x}, {y}) at different times:")
    for t in times:
        temp = heat_func(x, y, t)
        print(f"  t = {t:6.3f}s: {temp:.2f}°C")

def demo_vectorized():
    """Demonstrate vectorized evaluation."""
    print("\n=== Vectorized Evaluation ===")
    heat_func = create_heat_diffusion_function()
    
    # Create arrays of points
    x_points = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0])
    y_points = jnp.array([0.2, 0.4, 0.5, 0.6, 0.8])
    t_points = jnp.array([0.01, 0.03, 0.05, 0.07, 0.09])
    
    # Vectorize the function
    vmap_func = jax.vmap(heat_func, in_axes=(0, 0, 0))
    temperatures = vmap_func(x_points, y_points, t_points)
    
    print("Vectorized evaluation at multiple points:")
    for i in range(len(x_points)):
        print(f"  Point ({x_points[i]:.1f}, {y_points[i]:.1f}) at t={t_points[i]:.2f}s: {temperatures[i]:.2f}°C")

def demo_time_series():
    """Demonstrate time series analysis at a specific location."""
    print("\n=== Time Series Analysis ===")
    heat_func = create_heat_diffusion_function()
    
    # Fixed location
    x, y = 3.0, 0.5
    
    # Time array
    time_array = jnp.linspace(0.001, 0.1, 50)
    
    # Vectorize over time
    vmap_time = jax.vmap(heat_func, in_axes=(None, None, 0))
    temperatures = vmap_time(x, y, time_array)
    
    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, temperatures, 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Evolution at Point ({x}, {y})')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Time series computed for point ({x}, {y})")
    print(f"Initial temperature: {temperatures[0]:.2f}°C")
    print(f"Final temperature: {temperatures[-1]:.2f}°C")

def demo_temperature_field():
    """Demonstrate temperature field computation."""
    print("\n=== Temperature Field Snapshots ===")
    heat_func = create_heat_diffusion_function()
    
    # Create mesh grid
    X, Y = create_mesh_grid(nx=30, ny=15)
    
    # Vectorize for spatial computation
    spatial_vmap = jax.vmap(heat_func, in_axes=(0, 0, None))
    
    # Time snapshots
    time_snapshots = [0.001, 0.025, 0.05, 0.1]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, t in enumerate(time_snapshots):
        # Compute temperature field
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        temp_flat = spatial_vmap(X_flat, Y_flat, t)
        temp_field = temp_flat.reshape(X.shape)
        
        # Plot
        im = axes[i].imshow(temp_field, extent=[0, 10, 0, 1], origin='lower', 
                           cmap='jet', vmin=0, vmax=150, aspect='auto')
        axes[i].set_title(f'Time = {t:.3f}s')
        axes[i].set_xlabel('x (m)')
        axes[i].set_ylabel('y (m)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.tight_layout()
    plt.show()
    
    print("Temperature field snapshots computed and displayed")

def demo_boundary_analysis():
    """Demonstrate boundary behavior analysis."""
    print("\n=== Boundary Analysis ===")
    heat_func = create_heat_diffusion_function()
    
    # Analyze temperature along boundaries
    t = 0.05
    
    # Bottom boundary (y=0)
    x_bottom = jnp.linspace(0, 10, 50)
    y_bottom = jnp.zeros_like(x_bottom)
    
    # Top boundary (y=1)
    x_top = jnp.linspace(0, 10, 50)
    y_top = jnp.ones_like(x_top)
    
    # Vectorize
    vmap_boundary = jax.vmap(heat_func, in_axes=(0, 0, None))
    
    temp_bottom = vmap_boundary(x_bottom, y_bottom, t)
    temp_top = vmap_boundary(x_top, y_top, t)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_bottom, temp_bottom, 'b-', linewidth=2, label='Bottom boundary (y=0)')
    plt.plot(x_top, temp_top, 'r-', linewidth=2, label='Top boundary (y=1)')
    plt.xlabel('x (m)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'Temperature Along Boundaries at t={t}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Boundary analysis at t={t}s:")
    print(f"Bottom boundary temperature range: {jnp.min(temp_bottom):.2f}°C to {jnp.max(temp_bottom):.2f}°C")
    print(f"Top boundary temperature range: {jnp.min(temp_top):.2f}°C to {jnp.max(temp_top):.2f}°C")

if __name__ == "__main__":
    print("JAX Heat Diffusion Function Demonstration")
    print("=" * 50)
    
    # Run all demonstrations
    demo_single_point()
    demo_vectorized()
    demo_time_series()
    demo_temperature_field()
    demo_boundary_analysis()
    
    print("\n" + "=" * 50)
    print("All demonstrations completed successfully!")
    print("The heat diffusion function is ready for use with JAX vectorization.") 