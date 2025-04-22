import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=4, hidden_dim=100, output_dim=1):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
        self.init_weights()
    
    # initialize weights
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input):
        return self.net(input)

if __name__ == '__main__':
    model = MLP(input_dim=5, hidden_layers=5, hidden_dim=100, output_dim=1)
    
    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the MLP model: {total_params}")
