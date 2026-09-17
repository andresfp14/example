"""Train a configured model and select its best validation checkpoint."""

from pathlib import Path

import hydra

from modules.utils.hydraqol import run_decorator


@hydra.main(config_path="../config", config_name="train", version_base="1.3")
@run_decorator
def main(cfg) -> dict:
    # 1. Import computation in each worker so parallel launchers can serialize the task.
    import torch

    from modules.datasets.mnist import loaders
    from modules.training.training import train_model
    from modules.utils.hydraqol import write_json
    from modules.utils.seeds import configure

    # 2. Fix randomness before constructing the data and model.
    folder = Path(cfg.save_dir)
    device = configure(
        cfg.seed, cfg.training.device, cfg.training.deterministic, cfg.training.threads
    )
    train, valid, split = loaders(cfg.data, cfg.seed)
    model = hydra.utils.instantiate(cfg.model.object).to(device)

    # 3. Save the exact split alongside the configuration.
    write_json(folder / "split.json", split)

    # 4. Train on the training split and save the selected validation metrics.
    metrics = train_model(model, train, valid, cfg.training, device, folder)
    write_json(folder / "metrics.json", metrics)

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }


if __name__ == "__main__":
    main()
