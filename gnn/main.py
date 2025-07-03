#!/usr/bin/env python3
"""
Main script for training Graph Neural Networks for Finite Element Analysis

This script demonstrates the complete pipeline:
1. Dataset creation
2. Model initialization
3. Training
4. Evaluation
5. Visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from real_fea_dataset import create_real_fea_dataset
from model import create_mpnn_model
from trainer import create_trainer
from utils import move_data_to_model_device, tensor_to_numpy
import os
import time


def main():
    """Main training function"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set CUDA device to 0 if available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print(f"CUDA device set to: {torch.cuda.get_device_name(0)}")
    
    print("=== GNN for Finite Element Analysis ===")
    
    # Configuration
    config = {
        # Model parameters
        'node_input_dim': 3,      # x, y, z coordinates
        'edge_input_dim': 4,      # 3D vector + length
        'hidden_dim': 64,         # Hidden dimension for MLPs
        'latent_dim': 32,         # Latent space dimension
        'num_message_passing': 3, # Number of message passing operations
        'output_dim': 4,          # u, v, w, von Mises stress
        
        # Training parameters
        'batch_size': 1,          # Batch size (single mesh)
        'learning_rate': 0.001,   # Learning rate
        'weight_decay': 1e-5,     # Weight decay
        'num_epochs': 100,        # Number of training epochs
        
        # Dataset parameters
        'nodes_csv_path': './gnn/data/nodes.csv',  # Path to nodes CSV file
        'elements_csv_path': './gnn/data/mesh_data/element_connectivity.csv',  # Path to elements CSV file
        'normalization': 'minmax',  # Normalization method: 'minmax' or 'zscore'
        
        # Output parameters
        'output_dir': 'gnn_outputs',  # This will be relative to the gnn directory
        'save_interval': 10
    }
    
    print(f"Configuration: {config}")
    
    # Create real FEA dataset
    print("\n1. Creating real FEA dataset...")
    dataset = create_real_fea_dataset(
        nodes_csv_path=config['nodes_csv_path'],
        elements_csv_path=config['elements_csv_path'],
        normalization=config['normalization']
    )
    
    # Use the same dataset for both training and validation (no split)
    train_dataset = dataset
    val_dataset = dataset
    
    print(f"Dataset created with single mesh")
    print(f"Training on entire dataset (no split)")
    
    # Create model
    print("\n2. Creating model...")
    model = create_mpnn_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    print("\n3. Creating trainer...")
    trainer = create_trainer(model, train_dataset, val_dataset, config)
    
    # Train the model
    print("\n4. Starting training...")
    start_time = time.time()
    trainer.train(config['num_epochs'])
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate the model
    print("\n5. Evaluating model...")
    results, predictions, targets = trainer.evaluate()
    
    # Calculate RMSE
    rmse = np.sqrt(results['mse_loss'])
    print(f"\n=== Final Results ===")
    print(f"Training RMSE: {rmse:.6f}")
    print(f"Training MAE: {results['mae_loss']:.6f}")
    print(f"Relative Errors:")
    for name, error in results['relative_errors'].items():
        print(f"  {name}: {error:.4f}")
    
    # Denormalize predictions for comparison
    mesh_data = dataset[0]
    denorm_predictions = dataset.denormalize_predictions(predictions.numpy())
    denorm_targets = dataset.denormalize_predictions(targets.numpy())
    
    print(f"\n=== Denormalized Results ===")
    print(f"Original scale RMSE: {np.sqrt(np.mean((denorm_predictions - denorm_targets)**2)):.6f}")
    
    # Measure inference time
    print("\n6. Measuring inference time...")
    inference_time = measure_inference_time(model, mesh_data)
    print(f"Average inference time per forward pass: {inference_time:.6f} seconds")
    
    # Visualize results
    print("\n7. Creating visualizations...")
    visualize_results(predictions, targets, config['output_dir'])
    
    print("\n=== Training Complete ===")
    print(f"Results saved to: {config['output_dir']}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Inference time per forward pass: {inference_time:.6f} seconds")


def measure_inference_time(model, data, num_runs=100):
    """Measure average inference time for a single forward pass"""
    model.eval()
    
    # Move data to the same device as the model
    data = move_data_to_model_device(data, model)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(data)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time


def visualize_results(predictions, targets, output_dir):
    """Create visualization plots for the results"""
    
    # Create subplots for each output
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('GNN Predictions vs Targets', fontsize=18, fontweight='bold')
    
    output_names = ['u', 'v', 'w', 'von_mises']
    
    for i, (ax, name) in enumerate(zip(axes.flat, output_names)):
        pred = predictions[:, i].numpy()
        targ = targets[:, i].numpy()
        
        # Scatter plot
        ax.scatter(targ, pred, alpha=0.6, s=15)
        
        # Perfect prediction line
        min_val = min(pred.min(), targ.min())
        max_val = max(pred.max(), targ.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        # Calculate R² score
        correlation = np.corrcoef(targ, pred)[0, 1]
        ax.set_xlabel(f'Target {name}', fontsize=14)
        ax.set_ylabel(f'Predicted {name}', fontsize=14)
        ax.set_title(f'{name.upper()} (R² = {correlation:.3f})', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, output_dir)
    plt.savefig(os.path.join(output_dir, 'predictions_vs_targets.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create error distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Prediction Error Distributions', fontsize=18, fontweight='bold')
    
    for i, (ax, name) in enumerate(zip(axes.flat, output_names)):
        pred = predictions[:, i].numpy()
        targ = targets[:, i].numpy()
        errors = pred - targ
        
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.set_xlabel(f'Error ({name})', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title(f'{name.upper()} Error Distribution', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distributions.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


def demo_single_prediction():
    """Demonstrate prediction on a single mesh"""
    print("\n=== Single Mesh Prediction Demo ===")
    
    # Load the real FEA dataset
    from real_fea_dataset import create_real_fea_dataset
    dataset = create_real_fea_dataset(
        nodes_csv_path='./gnn/data/nodes.csv',
        elements_csv_path='./gnn/data/mesh_data/element_connectivity.csv',
        normalization='minmax'  # Use minmax normalization for demo
    )
    mesh_data = dataset[0]
    
    # Create model and load trained weights
    config = {
        'node_input_dim': 3,
        'edge_input_dim': 4,
        'hidden_dim': 64,
        'latent_dim': 32,
        'num_message_passing': 3,
        'output_dim': 4
    }
    
    model = create_mpnn_model(config)
    
    # Try to load trained model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'gnn_outputs/final_model.pt')
    if os.path.exists(model_path):
        # Load to CPU first, then move to appropriate device
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Loaded trained model")
    else:
        print("No trained model found, using untrained model for demo")
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        # Move mesh data to the same device as the model
        mesh_data = move_data_to_model_device(mesh_data, model)
        prediction = model(mesh_data)
    
    print(f"Mesh has {mesh_data.x.shape[0]} nodes")
    print(f"Predicted shape: {prediction.shape}")
    print(f"Target shape: {mesh_data.y.shape}")
    
    # Show some sample predictions (denormalized)
    print("\nSample predictions (first 5 nodes, denormalized):")
    print("Node | Predicted (u,v,w,σ) | Target (u,v,w,σ)")
    print("-" * 50)
    for i in range(min(5, mesh_data.x.shape[0])):
        pred = tensor_to_numpy(prediction[i])
        targ = tensor_to_numpy(mesh_data.y[i])
        
        # Denormalize
        denorm_pred = dataset.denormalize_predictions(pred.reshape(1, -1)).flatten()
        denorm_targ = dataset.denormalize_predictions(targ.reshape(1, -1)).flatten()
        
        print(f"{i:4d} | {denorm_pred[0]:8.6f} {denorm_pred[1]:8.6f} {denorm_pred[2]:8.6f} {denorm_pred[3]:8.1f} | "
              f"{denorm_targ[0]:8.6f} {denorm_targ[1]:8.6f} {denorm_targ[2]:8.6f} {denorm_targ[3]:8.1f}")


if __name__ == "__main__":
    # Run main training
    main()
    
    # Run single prediction demo
    demo_single_prediction() 