import torch
from torch import nn

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, w0=30., is_first_layer=False):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        self.is_first_layer = is_first_layer

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first_layer:
                self.linear.weight.uniform_(-1. / self.in_features, 
                                            1. / self.in_features)
            else:
                self.linear.weight.uniform_(-torch.sqrt(torch.tensor(6.) / self.in_features) / self.w0, 
                                            torch.sqrt(torch.tensor(6.) / self.in_features) / self.w0)
            
            if self.linear.bias is not None:
                self.linear.bias.fill_(0)

    def forward(self, input):
        return torch.sin(self.w0 * self.linear(input))

class Siren(nn.Module):
    def __init__(self, input_dim, hidden_layers=4, hidden_dim=100, output_dim=1):
        super().__init__()

        self.first_layer = SirenLayer(input_dim, hidden_dim, is_first_layer=True)
        layers = []
        for _ in range(hidden_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, input):
        output = self.first_layer(input)
        output = self.net(output)
        return output


if __name__ == '__main__':
    import torch
    from torch import nn

    model = Siren(input_dim=5, hidden_layers=6, hidden_dim=50, output_dim=1)

    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the Siren model: {total_params}")
