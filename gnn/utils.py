import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric.data import Data
import networkx as nx


def get_device():
    """Get the appropriate device (CUDA:0 or CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def ensure_device_consistency(data, device):
    """Ensure all tensors in a PyTorch Geometric Data object are on the same device"""
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x.to(device, non_blocking=True)
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        data.edge_index = data.edge_index.to(device, non_blocking=True)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr = data.edge_attr.to(device, non_blocking=True)
    if hasattr(data, 'y') and data.y is not None:
        data.y = data.y.to(device, non_blocking=True)
    return data


def check_device_consistency(data, device):
    """Check if all tensors in a PyTorch Geometric Data object are on the same device"""
    tensors = []
    if hasattr(data, 'x') and data.x is not None:
        tensors.append(('x', data.x.device))
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        tensors.append(('edge_index', data.edge_index.device))
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        tensors.append(('edge_attr', data.edge_attr.device))
    if hasattr(data, 'y') and data.y is not None:
        tensors.append(('y', data.y.device))
    
    target_device = torch.device(device)
    inconsistent = [(name, tensor_device) for name, tensor_device in tensors if tensor_device != target_device]
    
    if inconsistent:
        print(f"Warning: Found tensors on different devices:")
        for name, tensor_device in inconsistent:
            print(f"  {name}: {tensor_device} (expected: {target_device})")
        return False
    return True


def move_data_to_model_device(data, model):
    """Move PyTorch Geometric Data to the same device as the model"""
    device = next(model.parameters()).device
    return ensure_device_consistency(data, device)


def tensor_to_numpy(tensor):
    """Safely convert tensor to numpy array, handling device placement"""
    if tensor.is_cuda:
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()


def visualize_mesh_3d(data, output_path=None, show_plot=True):
    """
    Visualize a 3D mesh with node positions and edge connectivity
    
    Args:
        data: PyTorch Geometric Data object
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract node coordinates (handle CUDA tensors)
    node_coords = tensor_to_numpy(data.x)
    edge_index = tensor_to_numpy(data.edge_index)
    
    # Plot nodes
    ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
              c='red', s=50, alpha=0.8, label='Nodes')
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        src_coord = node_coords[src]
        dst_coord = node_coords[dst]
        
        ax.plot([src_coord[0], dst_coord[0]], 
                [src_coord[1], dst_coord[1]], 
                [src_coord[2], dst_coord[2]], 
                'b-', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh Visualization')
    ax.legend()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_field_on_mesh(data, field_values, field_name, output_path=None, show_plot=True):
    """
    Visualize a scalar field on the mesh nodes
    
    Args:
        data: PyTorch Geometric Data object
        field_values: Scalar values for each node
        field_name: Name of the field for the plot title
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract node coordinates (handle CUDA tensors)
    node_coords = tensor_to_numpy(data.x)
    edge_index = tensor_to_numpy(data.edge_index)
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(node_coords[:, 0], node_coords[:, 1], node_coords[:, 2], 
                        c=field_values, cmap='viridis', s=50, alpha=0.8)
    
    # Plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        src_coord = node_coords[src]
        dst_coord = node_coords[dst]
        
        ax.plot([src_coord[0], dst_coord[0]], 
                [src_coord[1], dst_coord[1]], 
                [src_coord[2], dst_coord[2]], 
                'k-', alpha=0.2, linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label(field_name)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{field_name} Field on Mesh')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def analyze_mesh_properties(data):
    """
    Analyze basic properties of the mesh
    
    Args:
        data: PyTorch Geometric Data object
    
    Returns:
        dict: Dictionary containing mesh properties
    """
    node_coords = tensor_to_numpy(data.x)
    edge_index = tensor_to_numpy(data.edge_index)
    edge_attr = tensor_to_numpy(data.edge_attr)
    
    # Basic properties
    num_nodes = node_coords.shape[0]
    num_edges = edge_index.shape[1]
    
    # Node properties
    node_degrees = np.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        node_degrees[src] += 1
        node_degrees[dst] += 1
    
    # Edge properties
    edge_lengths = edge_attr[:, 3]  # Assuming length is the 4th feature
    
    properties = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'average_node_degree': np.mean(node_degrees),
        'min_node_degree': np.min(node_degrees),
        'max_node_degree': np.max(node_degrees),
        'average_edge_length': np.mean(edge_lengths),
        'min_edge_length': np.min(edge_lengths),
        'max_edge_length': np.max(edge_lengths),
        'mesh_density': num_edges / num_nodes,
        'node_coords_range': {
            'x': (np.min(node_coords[:, 0]), np.max(node_coords[:, 0])),
            'y': (np.min(node_coords[:, 1]), np.max(node_coords[:, 1])),
            'z': (np.min(node_coords[:, 2]), np.max(node_coords[:, 2]))
        }
    }
    
    return properties


def create_mesh_statistics_report(dataset, output_path=None):
    """
    Create a comprehensive statistics report for a dataset of meshes
    
    Args:
        dataset: PyTorch Geometric Dataset
        output_path: Path to save the report (optional)
    """
    print("=== Mesh Dataset Statistics ===")
    
    all_properties = []
    for i in range(min(100, len(dataset))):  # Analyze first 100 meshes
        data = dataset[i]
        properties = analyze_mesh_properties(data)
        all_properties.append(properties)
    
    # Aggregate statistics
    avg_properties = {}
    for key in all_properties[0].keys():
        if isinstance(all_properties[0][key], (int, float)):
            values = [p[key] for p in all_properties]
            avg_properties[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        else:
            avg_properties[key] = all_properties[0][key]  # Keep the same for non-numeric
    
    # Print report
    print(f"Dataset size: {len(dataset)} meshes")
    print(f"Analyzed: {len(all_properties)} meshes")
    print()
    
    for key, stats in avg_properties.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"{key}:")
            print(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        else:
            print(f"{key}: {stats}")
        print()
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write("=== Mesh Dataset Statistics ===\n")
            f.write(f"Dataset size: {len(dataset)} meshes\n")
            f.write(f"Analyzed: {len(all_properties)} meshes\n\n")
            
            for key, stats in avg_properties.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    f.write(f"{key}:\n")
                    f.write(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
                    f.write(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]\n")
                else:
                    f.write(f"{key}: {stats}\n")
                f.write("\n")


def compare_predictions_targets(predictions, targets, output_dir):
    """
    Create detailed comparison plots between predictions and targets
    
    Args:
        predictions: Predicted values [num_nodes, num_outputs]
        targets: Target values [num_nodes, num_outputs]
        output_dir: Directory to save plots
    """
    output_names = ['u', 'v', 'w', 'von_mises']
    
    # Create comparison plots for each output
    for i, name in enumerate(output_names):
        pred = predictions[:, i].numpy()
        targ = targets[:, i].numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax1.scatter(targ, pred, alpha=0.6, s=10)
        min_val = min(pred.min(), targ.min())
        max_val = max(pred.max(), targ.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax1.set_xlabel(f'Target {name}')
        ax1.set_ylabel(f'Predicted {name}')
        ax1.set_title(f'{name.upper()} - Predictions vs Targets')
        ax1.grid(True, alpha=0.3)
        
        # Error histogram
        errors = pred - targ
        ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel(f'Error ({name})')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{name.upper()} - Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create summary statistics
    print("\n=== Prediction vs Target Summary ===")
    for i, name in enumerate(output_names):
        pred = predictions[:, i].numpy()
        targ = targets[:, i].numpy()
        errors = pred - targ
        
        mse = np.mean(errors**2)
        mae = np.mean(np.abs(errors))
        rel_error = np.mean(np.abs(errors / (targ + 1e-8)))
        
        print(f"{name.upper()}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Relative Error: {rel_error:.4f}")
        print()


def save_mesh_to_file(data, filename):
    """
    Save mesh data to a simple text file format
    
    Args:
        data: PyTorch Geometric Data object
        filename: Output filename
    """
    node_coords = data.x.numpy()
    edge_index = data.edge_index.numpy()
    
    with open(filename, 'w') as f:
        # Write header
        f.write(f"# Mesh with {node_coords.shape[0]} nodes and {edge_index.shape[1]} edges\n")
        f.write("# Format: node_id x y z\n")
        
        # Write nodes
        f.write("\n# Nodes\n")
        for i, coord in enumerate(node_coords):
            f.write(f"{i} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
        
        # Write edges
        f.write("\n# Edges\n")
        f.write("# Format: node1 node2\n")
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            f.write(f"{src} {dst}\n")
    
    print(f"Mesh saved to {filename}") 