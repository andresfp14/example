import os
import random
import numpy as np
import torch

def seed_everything(seed=0):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
    - seed (int): Seed value. Default is 0.

    Note:
    This function sets seeds for the Python standard library, NumPy, and PyTorch.
    Additionally, it sets the environment variable for PL_GLOBAL_SEED.
    """
    
    # Set seed for the Python standard library's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If CUDA is available, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Set environment variable for PyTorch Lightning's global seed
    os.environ["PL_GLOBAL_SEED"] = str(seed)