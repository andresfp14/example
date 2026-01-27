import torch
import torch.nn as nn
import torch.nn.functional as F


class NetBN(nn.Module):
    """
    A slightly more regularized CNN than simple_net.Net:
    - BatchNorm after each convolution to stabilize training.
    - Configurable intermediate fully connected stack for quick depth sweeps.
    """

    def __init__(self, num_layers: int = 2, latent_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        # Convolutional stem
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected stack
        self.fc1 = nn.Linear(9216, latent_dim)
        self.fc_intermediates = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(num_layers - 1)]
        )
        self.fc_out = nn.Linear(latent_dim, 10)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        for fc in self.fc_intermediates:
            x = F.relu(fc(x))
            x = self.dropout(x)

        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)
