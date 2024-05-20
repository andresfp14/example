import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_layers=1, latent_dim=128):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # First fully connected layer
        self.fc1 = nn.Linear(9216, latent_dim)
        
        # Intermediate fully connected layers
        self.fc_intermediates = nn.ModuleList(
            [nn.Linear(latent_dim, latent_dim) for _ in range(num_layers - 1)]
        )
        
        # Output layer
        self.fc2 = nn.Linear(latent_dim, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        
        # Apply the intermediate fully connected layers
        for fc in self.fc_intermediates:
            x = fc(x)
            x = F.relu(x)
        
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
