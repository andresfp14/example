"""Explicit randomness and compute settings for an experiment."""

import os
import random

import numpy as np
import torch


def configure(seed: int, device: str, deterministic: bool, threads: int) -> torch.device:
    # 1. Set the CUDA workspace before any GPU operations.
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # 2. Seed the random generators used by the example.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # 3. Apply the requested execution settings; PyTorch reports unsupported devices.
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    return torch.device(device)
