import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from pathlib import Path

from modules.models.simple_net import Net
from modules.training.training import train, test

# Registering the config path with Hydra
@hydra.main(config_path="./data/config", config_name="train_model", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main function for training and evaluating a neural network on the MNIST dataset.
    Utilizes Hydra for configuration management, separating model and training configurations.

    Args:
        cfg (DictConfig): Configuration object containing all parameters and sub-configurations.
            Structure and default values of cfg are as follows:
            ```
            model:
                num_layers: 2  # Default: 2, Number of layers in the neural network model.
            training:
                batch_size: 64  # Default: 64, Input batch size for training.
                test_batch_size: 1000  # Default: 1000, input batch size for testing.
                epochs: 14  # Default: 14, number of epochs to train.
                lr: 1.0  # Default: 1.0, learning rate.
                gamma: 0.7  # Default: 0.7, learning rate step gamma.
                no_cuda: False  # Default: False, flag to disable CUDA training.
                no_mps: False  # Default: False, flag to disable mps training.
                dry_run: False  # Default: False, flag for a quick single pass.
                seed: 1  # Default: 1, random seed for reproducibility.
                log_interval: 10  # Default: 10, interval for logging training status.
                save_model: True  # Default: True, flag to save the trained model.
                data_dir: "./data"  # Default: "./data", directory for storing dataset files.
                model_dir: "./models"  # Default: "./models", directory for saving trained model files.
            ```

    Returns:
        None: This function does not return any value.

    Examples:
        To run training with the default configuration specified in `./data/config/train_model.yaml`:
        ```bash
        $ python train.py
        ```

        To change the number of epochs to 20:
        ```bash
        $ python train.py training.epochs=20
        ```

        To override configuration with another file `alternative.yaml`:
        ```bash
        $ python train.py +config=alternative.yaml
        ```

        To perform multiple runs with different model sizes using Hydra's multirun feature:
        ```bash
        $ python train.py --multirun model.num_layers=1,2,3
        ```

        Using Hydra and Slurm for cluster job submissions:
        ```bash
        $ python train.py --multirun model.num_layers=1,2,3 hydra/launcher=slurm \
            hydra.launcher.partition=my_partition \
            hydra.launcher.comment='MNIST training runs' \
            hydra.launcher.nodes=1 \
            hydra.launcher.tasks_per_node=1 \
            hydra.launcher.mem_per_cpu=4G
        ```
        
        Note: For integrating Hydra with Slurm, additional configuration may be required and should be checked against Hydra's documentation and your Slurm setup.
    """

    # Determine if CUDA or MPS should be used based on configuration and availability
    use_cuda: bool = not cfg.training.no_cuda and torch.cuda.is_available()
    use_mps: bool = not cfg.training.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(cfg.training.seed)

    device: torch.device = torch.device("cuda") if use_cuda else torch.device("mps") if use_mps else torch.device("cpu")

    # Setup DataLoader arguments based on device availability
    train_kwargs: dict = {'batch_size': cfg.training.batch_size}
    test_kwargs: dict = {'batch_size': cfg.training.test_batch_size}
    if use_cuda:
        cuda_kwargs: dict = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Image transformation pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Dataset preparation
    dataset1: datasets.MNIST = datasets.MNIST(cfg.training.data_dir, train=True, download=True, transform=transform)
    dataset2: datasets.MNIST = datasets.MNIST(cfg.training.data_dir, train=False, transform=transform)
    
    # DataLoaders for training and testing
    train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Model initialization
    model: Net = Net(num_layers=cfg.model.num_layers).to(device)
    
    # Optimizer setup
    optimizer: optim.Optimizer = optim.Adadelta(model.parameters(), lr=cfg.training.lr)
    
    # Learning rate scheduler
    scheduler: StepLR = StepLR(optimizer, step_size=1, gamma=cfg.training.gamma)

    # Training loop
    for epoch in range(1, cfg.training.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, cfg.training.log_interval, cfg.training.dry_run)
        test(model, device, test_loader)
        scheduler.step()

    # Save the model checkpoint if configured to do so
    if cfg.training.save_model:
        Path(cfg.training.model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), f"{cfg.training.model_dir}/mnist_cnn_{cfg.training.seed}.pt")

if __name__ == '__main__':
    main()