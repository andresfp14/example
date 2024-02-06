import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import fire

from modules.models.simple_net import Net
from modules.training.training import train, test
from modules.utils.parallelize import pex


def main(batch_size: int = 64, test_batch_size: int = 1000, epochs: int = 14, lr: float = 1.0, 
         gamma: float = 0.7, no_cuda: bool = False, no_mps: bool = False, 
         dry_run: bool = False, seed: int = 1, log_interval: int = 10, save_model: bool = False) -> None:
    """
    Main function for training and evaluating a neural network on the MNIST dataset.

    Args:
        batch_size (int): Input batch size for training. Default: 64.
        test_batch_size (int): Input batch size for testing. Default: 1000.
        epochs (int): Number of epochs to train. Default: 14.
        lr (float): Learning rate. Default: 1.0.
        gamma (float): Learning rate step gamma. Default: 0.7.
        no_cuda (bool): Flag to disable CUDA training. Default: False.
        no_mps (bool): Flag to disable macOS GPU training. Default: False.
        dry_run (bool): Flag for a quick single pass. Default: False.
        seed (int): Random seed. Default: 1.
        log_interval (int): Interval for logging training status. Default: 10.
        save_model (bool): Flag to save the current model. Default: False.

    Returns:
        None: This function does not return any value.
    """
    use_cuda = not no_cuda and torch.cuda.is_available()
    use_mps = not no_mps and torch.backends.mps.is_available()

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval, dry_run)
        test(model, device, test_loader)
        scheduler.step()

    if save_model:
        Path("./data/models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"./data/models/mnist_cnn_{seed}.pt")


if __name__ == '__main__':
    fire.Fire()