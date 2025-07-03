import torch
import numpy as np
from torch_geometric.data import Data, Dataset
import random


class FEDataset(Dataset):
    """
    Finite Element Analysis Dataset with tetrahedral meshes.
    Generates synthetic data with nodal coordinates, displacements, and von Mises stress.
    """
    
    def __init__(self, num_samples=1000, num_nodes=20, transform=None):
        super(FEDataset, self).__init__(transform)
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        
    def len(self):
        return self.num_samples
    
    def get(self, idx):
        """Generate a single FE mesh with synthetic data"""
        # Generate random nodal coordinates (3D)
        node_coords = torch.rand(self.num_nodes, 3) * 10.0  # Random coordinates in [0, 10]^3
        
        # Generate tetrahedral connectivity (simplified - just create some edges)
        # In a real scenario, this would be proper tetrahedral mesh generation
        edge_index = self._generate_tetrahedral_edges(node_coords)
        
        # Generate edge features: 3D vector and length
        edge_features = self._compute_edge_features(node_coords, edge_index)
        
        # Generate synthetic displacement and stress data
        # Using smooth functions based on coordinates
        displacements, von_mises = self._generate_synthetic_data(node_coords)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_coords,  # Node features: coordinates (3D)
            edge_index=edge_index,  # Edge connectivity
            edge_attr=edge_features,  # Edge features: vector + length (4D)
            y=torch.cat([displacements, von_mises.unsqueeze(1)], dim=1)  # Target: u,v,w + von Mises (4D)
        )
        
        return data
    
    def _generate_tetrahedral_edges(self, node_coords):
        """Generate edge connectivity for a tetrahedral mesh"""
        # Simplified approach: connect each node to its nearest neighbors
        edges = []
        for i in range(self.num_nodes):
            # Find nearest neighbors
            distances = torch.norm(node_coords - node_coords[i], dim=1)
            # Get indices of 4 nearest neighbors (excluding self)
            nearest_indices = torch.argsort(distances)[1:5]  # Skip first (self)
            
            for j in nearest_indices:
                if i < j:  # Avoid duplicate edges
                    edges.append([i, j])
                    edges.append([j, i])  # Bidirectional
        
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    def _compute_edge_features(self, node_coords, edge_index):
        """Compute edge features: 3D vector and length"""
        edge_features = []
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            edge_vector = node_coords[dst] - node_coords[src]
            edge_length = torch.norm(edge_vector)
            
            # Normalize edge vector and concatenate with length
            edge_vector_normalized = edge_vector / (edge_length + 1e-8)
            edge_feature = torch.cat([edge_vector_normalized, edge_length.unsqueeze(0)])
            edge_features.append(edge_feature)
        
        return torch.stack(edge_features)
    
    def _generate_synthetic_data(self, node_coords):
        """Generate synthetic displacement and von Mises stress data"""
        # Generate smooth displacement field based on coordinates
        x, y, z = node_coords[:, 0], node_coords[:, 1], node_coords[:, 2]
        
        # Displacement functions (smooth, physics-inspired)
        u = 0.1 * torch.sin(x) * torch.cos(y) * torch.exp(-z/5.0)
        v = 0.1 * torch.cos(x) * torch.sin(y) * torch.exp(-z/5.0)
        w = 0.05 * torch.sin(x + y) * torch.exp(-z/3.0)
        
        displacements = torch.stack([u, v, w], dim=1)
        
        # Von Mises stress (smooth function based on displacement gradients)
        von_mises = 100.0 + 50.0 * torch.sin(x/2.0) * torch.cos(y/2.0) * torch.exp(-z/4.0)
        
        return displacements, von_mises


def create_fe_dataset(num_samples=1000, num_nodes=20, train_ratio=0.8):
    """Create train/validation split of the FE dataset"""
    dataset = FEDataset(num_samples=num_samples, num_nodes=num_nodes)
    
    # Split into train and validation
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset 