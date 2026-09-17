"""Evaluate a chosen checkpoint on the held-out test set."""

from pathlib import Path

import hydra

from modules.utils.hydraqol import run_decorator, write_json


@hydra.main(config_path="../config", config_name="evaluate", version_base="1.3")
@run_decorator
def main(cfg) -> dict:
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from modules.datasets.mnist import dataset
    from modules.training.training import evaluate
    from modules.utils.seeds import configure

    # 1. Reconstruct the selected model from its saved configuration and weights.
    source = Path(cfg.run_dir)
    original = OmegaConf.load(source / "config.yaml")
    folder = Path(cfg.save_dir)
    device = configure(original.seed, cfg.device, original.training.deterministic, 1)
    model = hydra.utils.instantiate(original.model.object).to(device)
    model.load_state_dict(torch.load(source / "weights.pt", map_location=device, weights_only=True))

    # 2. Evaluate once on the full test split without updating the model.
    loader = DataLoader(
        dataset(original.data.root, train=False), batch_size=original.data.batch_size
    )
    write_json(folder / "metrics.json", evaluate(model, loader, device))

    return {"source_run": str(source.resolve()), "device": str(device)}


if __name__ == "__main__":
    main()
