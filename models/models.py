import torch.nn as nn
import torch
from .ast_models import ASTModel

#Adapted from https://github.com/AntixK/PyTorch-VAE
class ConvHead(nn.Module):
    def __init__(self, latent_dim = 128, hidden_dims = None):
        super().__init__()
        self.latent_dim = latent_dim
        in_channels = 1

        modules = []
        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128, 256, 512]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride= (2, 2), padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1]*2, latent_dim)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:

        result = self.encoder(input)

        result = torch.flatten(result, start_dim=1)

        output = self.fc(result)

        return output

class AST_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ASTModel(input_tdim=100, label_dim=2)
    def forward(self, X):
        output = self.model(X)
        return output