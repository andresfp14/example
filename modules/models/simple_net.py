"""A small MNIST classifier returning unnormalized class logits."""

import torch
from torch import nn


class Net(nn.Module):
    """Map (batch, 1, 28, 28) images to (batch, 10) logits.

    num_layers counts hidden linear layers; BatchNorm is an optional ablation.
    """

    def __init__(
        self,
        num_layers: int = 2,
        latent_dim: int = 128,
        dropout: float = 0.25,
        batch_norm: bool = False,
    ):
        super().__init__()
        # 1. Extract image features, optionally adding BatchNorm.
        features = []
        for incoming, outgoing in ((1, 32), (32, 64)):
            features.append(nn.Conv2d(incoming, outgoing, 3))
            if batch_norm:
                features.append(nn.BatchNorm2d(outgoing))
            features.append(nn.ReLU())
        features.extend([nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(dropout)])
        self.features = nn.Sequential(*features)
        # 2. Vary the hidden linear depth while keeping the output classes fixed.
        hidden = []
        for layer in range(num_layers):
            hidden.extend(
                [
                    nn.Linear(9216 if layer == 0 else latent_dim, latent_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.classifier = nn.Sequential(*hidden, nn.Linear(latent_dim, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Map image features to ten unnormalized class scores.
        return self.classifier(self.features(x))
