import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
import os


class RealFEADataset(Dataset):
    """
    Real Finite Element Analysis Dataset using actual FEA results from CSV files.
    """
    
    def __init__(self, nodes_csv_path, elements_csv_path, transform=None, normalization='minmax'):
        super(RealFEADataset, self).__init__(transform)
        
        self.nodes_csv_path = nodes_csv_path
        self.elements_csv_path = elements_csv_path
        self.normalization = normalization  # 'minmax' or 'zscore'
        
        # Load and preprocess data
        self._load_data()
        
    def _load_data(self):
        """Load node and element data from CSV files"""
        print("Loading real FEA data...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load node data
        self.nodes_csv_path = os.path.join(script_dir, self.nodes_csv_path)
        nodes_df = pd.read_csv(self.nodes_csv_path)
        print(f"Loaded {len(nodes_df)} nodes from {self.nodes_csv_path}")
        
        # Extract node coordinates (columns 1, 2, 3: X, Y, Z)
        self.node_coords = nodes_df.iloc[:, [1, 2, 3]].values.astype(np.float32)
        
        # Extract target values (columns 7, 8, 9, 10: U-X, U-Y, U-Z, S-EQV)
        self.targets = nodes_df.iloc[:, [7, 8, 9, 10]].values.astype(np.float32)
        
        # Load element connectivity
        self.elements_csv_path = os.path.join(script_dir, self.elements_csv_path)
        elements_df = pd.read_csv(self.elements_csv_path)
        print(f"Loaded {len(elements_df)} elements from {self.elements_csv_path}")
        
        # Extract element connectivity (columns 1, 2, 3, 4: Node1, Node2, Node3, Node4)
        # Convert to 0-based indexing for PyTorch Geometric
        self.element_connectivity = elements_df.iloc[:, [1, 2, 3, 4]].values.astype(np.int64) - 1
        
        # Create edge connectivity from element connectivity
        self._create_edge_connectivity()
        
        # Normalize data
        self._normalize_data()
        
        print(f"Dataset created with {len(self.node_coords)} nodes and {len(self.element_connectivity)} elements")
        print(f"Using {self.normalization} normalization")
        
    def _create_edge_connectivity(self):
        """Create edge connectivity from element connectivity"""
        edges = set()
        
        # For each tetrahedral element, create edges between all pairs of nodes
        for element in self.element_connectivity:
            nodes = element.tolist()
            # Create all possible edges between the 4 nodes
            for i in range(4):
                for j in range(i+1, 4):
                    edge = tuple(sorted([nodes[i], nodes[j]]))
                    edges.add(edge)
        
        # Convert to edge index format for PyTorch Geometric
        edge_list = list(edges)
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        print(f"Created {len(edge_list)} unique edges")
        
    def _normalize_data(self):
        """Normalize node coordinates and targets using specified normalization method"""
        if self.normalization == 'minmax':
            self._normalize_minmax()
        elif self.normalization == 'zscore':
            self._normalize_zscore()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}. Use 'minmax' or 'zscore'")
    
    def _normalize_minmax(self):
        """Normalize node coordinates and targets between 0 and 1"""
        # Normalize node coordinates between 0 and 1
        coord_min = np.min(self.node_coords, axis=0)
        coord_max = np.max(self.node_coords, axis=0)
        self.node_coords = (self.node_coords - coord_min) / (coord_max - coord_min + 1e-8)
        
        # Normalize targets (displacements and von Mises stress) between 0 and 1
        target_min = np.min(self.targets, axis=0)
        target_max = np.max(self.targets, axis=0)
        self.targets = (self.targets - target_min) / (target_max - target_min + 1e-8)
        
        # Store normalization parameters for later use
        self.coord_min = coord_min
        self.coord_max = coord_max
        self.target_min = target_min
        self.target_max = target_max
        
        print("Data normalized between 0 and 1 (min-max normalization)")
    
    def _normalize_zscore(self):
        """Normalize node coordinates and targets using z-score normalization"""
        # Normalize node coordinates
        coord_mean = np.mean(self.node_coords, axis=0)
        coord_std = np.std(self.node_coords, axis=0)
        self.node_coords = (self.node_coords - coord_mean) / (coord_std + 1e-8)
        
        # Normalize targets (displacements and von Mises stress)
        target_mean = np.mean(self.targets, axis=0)
        target_std = np.std(self.targets, axis=0)
        self.targets = (self.targets - target_mean) / (target_std + 1e-8)
        
        # Store normalization parameters for later use
        self.coord_mean = coord_mean
        self.coord_std = coord_std
        self.target_mean = target_mean
        self.target_std = target_std
        
        print("Data normalized using z-score normalization")
        
    def len(self):
        return 1  # Single mesh dataset
    
    def get(self, idx):
        """Get the mesh data"""
        # Compute edge features: normalized edge vectors and lengths
        edge_features = self._compute_edge_features()
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(self.node_coords, dtype=torch.float32),  # Node features: normalized coordinates
            edge_index=self.edge_index,  # Edge connectivity
            edge_attr=torch.tensor(edge_features, dtype=torch.float32),  # Edge features: vector + length
            y=torch.tensor(self.targets, dtype=torch.float32)  # Target: normalized displacements + von Mises stress
        )
        
        return data
    
    def _compute_edge_features(self):
        """Compute edge features: normalized edge vectors and lengths"""
        edge_features = []
        
        for i in range(self.edge_index.size(1)):
            src, dst = self.edge_index[0, i], self.edge_index[1, i]
            edge_vector = self.node_coords[dst] - self.node_coords[src]
            edge_length = np.linalg.norm(edge_vector)
            
            # Normalize edge vector and concatenate with length
            if edge_length > 1e-8:
                edge_vector_normalized = edge_vector / edge_length
            else:
                edge_vector_normalized = edge_vector
                
            edge_feature = np.concatenate([edge_vector_normalized, [edge_length]])
            edge_features.append(edge_feature)
        
        return np.array(edge_features, dtype=np.float32)
    
    def get_normalization_params(self):
        """Get normalization parameters for denormalization"""
        if self.normalization == 'minmax':
            return {
                'coord_min': self.coord_min,
                'coord_max': self.coord_max,
                'target_min': self.target_min,
                'target_max': self.target_max
            }
        elif self.normalization == 'zscore':
            return {
                'coord_mean': self.coord_mean,
                'coord_std': self.coord_std,
                'target_mean': self.target_mean,
                'target_std': self.target_std
            }
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")
    
    def denormalize_predictions(self, predictions):
        """Denormalize predictions back to original scale"""
        if self.normalization == 'minmax':
            return predictions * (self.target_max - self.target_min) + self.target_min
        elif self.normalization == 'zscore':
            return predictions * self.target_std + self.target_mean
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")


def create_real_fea_dataset(nodes_csv_path, elements_csv_path, normalization='minmax'):
    """Create a real FEA dataset from CSV files
    
    Args:
        nodes_csv_path: Path to nodes CSV file
        elements_csv_path: Path to elements CSV file
        normalization: Normalization method ('minmax' or 'zscore'), default is 'minmax'
    """
    return RealFEADataset(nodes_csv_path, elements_csv_path, normalization=normalization) 