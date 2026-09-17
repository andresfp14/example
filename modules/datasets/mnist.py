"""MNIST with a validation split independent of the training seed."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def dataset(root: str, *, train: bool, download: bool = False):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return datasets.MNIST(root, train=train, download=download, transform=transform)


def loaders(cfg, seed: int):
    """Return training/validation loaders and the exact split indices."""
    # 1. Fix the split independently of model initialization and training order.
    data = dataset(cfg.root, train=True)
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(cfg.split_seed))
    valid_ids, train_ids = indices[: cfg.validation_size], indices[cfg.validation_size :]
    # 2. Optionally shorten both splits for quick exploratory runs.
    train_ids = train_ids[: cfg.train_limit]
    valid_ids = valid_ids[: cfg.valid_limit]

    # 3. Shuffle training examples with a separate, seeded generator.
    kwargs = {"batch_size": cfg.batch_size, "num_workers": cfg.workers}
    train = DataLoader(
        Subset(data, train_ids.tolist()),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **kwargs,
    )
    valid = DataLoader(Subset(data, valid_ids.tolist()), shuffle=False, **kwargs)
    return train, valid, {"train": train_ids.tolist(), "validation": valid_ids.tolist()}
