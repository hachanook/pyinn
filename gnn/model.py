import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class MLP(nn.Module):
    """Multi-layer perceptron with batch normalization"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MessagePassingLayer(MessagePassing):
    """Message passing layer for the GNN"""
    
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MessagePassingLayer, self).__init__(aggr='mean')  # Use mean aggregation
        
        # Message function: combines node and edge features
        self.message_mlp = MLP(
            input_dim=node_dim + edge_dim + node_dim,  # src_node + edge + dst_node
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2
        )
        
        # Update function: combines current node features with aggregated messages
        self.update_mlp = MLP(
            input_dim=node_dim + hidden_dim,  # current_node + aggregated_messages
            hidden_dim=hidden_dim,
            output_dim=node_dim,
            num_layers=2
        )
    
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to edge_index
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # x_i: source node features
        # x_j: target node features  
        # edge_attr: edge features
        
        # Concatenate source node, edge, and target node features
        inputs = torch.cat([x_i, edge_attr, x_j], dim=1)
        return self.message_mlp(inputs)
    
    def update(self, aggr_out, x):
        # aggr_out: aggregated messages
        # x: current node features
        
        # Concatenate current node features with aggregated messages
        inputs = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(inputs)


class MPNN(nn.Module):
    """
    Message Passing Neural Network for Finite Element Analysis
    
    Architecture:
    1. Encoder: Node features (coordinates) -> Latent space
    2. Message Passing: Multiple message passing operations in latent space
    3. Decoder: Latent space -> Output (displacements + von Mises stress)
    """
    
    def __init__(self, 
                 node_input_dim=3,      # Node features: x, y, z coordinates
                 edge_input_dim=4,      # Edge features: 3D vector + length
                 hidden_dim=64,         # Hidden dimension for all MLPs
                 latent_dim=32,         # Latent space dimension
                 num_message_passing=3, # Number of message passing operations
                 output_dim=4):         # Output: u, v, w, von Mises stress
        
        super(MPNN, self).__init__()
        
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_message_passing = num_message_passing
        self.output_dim = output_dim
        
        # Encoder: Node features -> Latent space
        self.node_encoder = MLP(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=3
        )
        
        # Edge encoder: Edge features -> Latent space
        self.edge_encoder = MLP(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_layers=2
        )
        
        # Message passing layers
        self.message_passing_layers = nn.ModuleList([
            MessagePassingLayer(
                node_dim=latent_dim,
                edge_dim=latent_dim,
                hidden_dim=hidden_dim
            ) for _ in range(num_message_passing)
        ])
        
        # Decoder: Latent space -> Output
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3
        )
    
    def forward(self, data):
        """
        Forward pass through the MPNN
        
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features (coordinates) [num_nodes, node_input_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_input_dim]
        
        Returns:
            output: Predicted displacements and von Mises stress [num_nodes, output_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Encoding phase
        node_latent = self.node_encoder(x)  # [num_nodes, latent_dim]
        edge_latent = self.edge_encoder(edge_attr)  # [num_edges, latent_dim]
        
        # Message passing phase
        current_node_latent = node_latent
        for message_layer in self.message_passing_layers:
            current_node_latent = message_layer(current_node_latent, edge_index, edge_latent)
            # Add residual connection
            current_node_latent = current_node_latent + node_latent
        
        # Decoding phase
        output = self.decoder(current_node_latent)  # [num_nodes, output_dim]
        
        return output
    
    def predict_displacements(self, data):
        """Predict only displacements (first 3 outputs)"""
        output = self.forward(data)
        return output[:, :3]  # u, v, w
    
    def predict_von_mises(self, data):
        """Predict only von Mises stress (last output)"""
        output = self.forward(data)
        return output[:, 3]  # von Mises stress


def create_mpnn_model(config):
    """Create MPNN model from configuration dictionary"""
    return MPNN(
        node_input_dim=config.get('node_input_dim', 3),
        edge_input_dim=config.get('edge_input_dim', 4),
        hidden_dim=config.get('hidden_dim', 64),
        latent_dim=config.get('latent_dim', 32),
        num_message_passing=config.get('num_message_passing', 3),
        output_dim=config.get('output_dim', 4)
    ) 