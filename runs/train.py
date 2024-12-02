# Add the parent directory to the Python path (because we are executing hydra from within runs)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import json
import time

from modules.training.training import train_model
from modules.utils.seeds import seed_everything

# Registering the config path with Hydra
@hydra.main(config_path="../data/config", config_name="train_model", version_base="1.3")
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
                use_cuda: True  # Default: True, flag to enable CUDA training.
                use_mps: True  # Default: True, flag to enable macOS GPU training.
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
        To run training with the default configuration specified in `../data/config/train_model.yaml`:
        ```bash
        $ python runs/01_train_model.py
        ```

        To change the number of epochs to 2 and seed to 7:
        ```bash
        $ python runs/01_train_model.py training.epochs=2 training.seed=7
        ```

        To override configuration with another file `sweep_models_lr.yaml`:
        ```bash
        $ python runs/01_train_model.py +experiment=sweep_models_lr
        ```

        To perform multiple runs with different model sizes and training epochs using Hydra's multirun feature:
        ```bash
        $ python runs/01_train_model.py --multirun training.epochs=2 model.num_layers=1,2,3
        ```

        Using Hydra's launcher for multiple runs:
        ```bash
        $ python runs/01_train_model.py --multirun training.epochs=2 model.num_layers=1,2,3 +launcher=joblib
        ```

        Or using Slurm for cluster job submissions:
        ```bash
        $ python runs/01_train_model.py --multirun training.epochs=2 model.num_layers=1,2,3 +launcher=slurm
        ```

        Or using Slurm with GPU for multiple seeds:
        ```bash
        $ python runs/01_train_model.py --multirun training.epochs=2 training.seed=0,1,2,3,4 +launcher=slurmgpu
        ```

        Note: For integrating Hydra with Slurm, additional configuration may be required and should be checked against Hydra's documentation and your Slurm setup.
    """

    ##############################
    # Preliminaries
    ##############################

    # Create a directory for saving models and results
    model_save_dir = Path(cfg.path.save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    # Set the random seed for reproducibility
    seed_everything(cfg.training.seed)

    # Select the device for computation (CUDA, MPS, or CPU)
    device = "cuda" if (cfg.training.device=="cuda" and torch.cuda.is_available()) else "cpu"

    ##############################
    # Object Instantiation
    ##############################

    # Instantiate transforms for training and testing
    train_transform = hydra.utils.instantiate(cfg.transforms.train)
    test_transform = hydra.utils.instantiate(cfg.transforms.test)

    # Instantiate datasets with transforms
    train_dataset = hydra.utils.instantiate(cfg.dataset.train, transform=train_transform)
    test_dataset = hydra.utils.instantiate(cfg.dataset.test, transform=test_transform)

    # Instantiate data loaders
    train_loader = hydra.utils.instantiate(cfg.loader.train, dataset=train_dataset)
    test_loader = hydra.utils.instantiate(cfg.loader.test, dataset=test_dataset)

    # Instantiate model
    model = hydra.utils.instantiate(cfg.model.object).to(device)

    ##############################
    # Actual Task: Training Loop
    ##############################

    # Training loop    
    train_model(model, train_loader, test_loader, cfg.training)

    ##############################
    # Saving Results
    ##############################

    # Save the model checkpoint if configured to do so
    model_path = model_save_dir / f"checkpoint.ckpt"
    torch.save(model.state_dict(), model_path)

    # Save the configuration
    config_path = model_save_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

if __name__ == '__main__':
    main()