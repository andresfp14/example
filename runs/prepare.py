"""Download MNIST before local experiments or cluster submission."""

import hydra

from modules.utils.hydraqol import run_decorator


@hydra.main(config_path="../config", config_name="prepare", version_base="1.3")
@run_decorator
def main(cfg) -> None:
    from modules.datasets.mnist import dataset

    # 1. Download both official splits into the shared data directory.
    for train in (True, False):
        dataset(cfg.data.root, train=train, download=True)
    print(f"MNIST ready: {cfg.data.root}")


if __name__ == "__main__":
    main()
