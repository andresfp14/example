"""Compare validation results across seeds within one study."""

import hashlib
import json
from pathlib import Path

import hydra

from modules.utils.hydraqol import read_run_info, run_decorator


@hydra.main(config_path="../config", config_name="report", version_base="1.3")
@run_decorator
def main(cfg) -> None:
    import matplotlib
    import pandas as pd
    from omegaconf import OmegaConf

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = Path(cfg.save_dir)
    # 1. Collect completed runs and show which unfinished runs were skipped.
    rows = []
    source = Path(cfg.paths.outputs) / cfg.study / "train"
    for run in sorted(source.iterdir()):
        info = read_run_info(run)
        if info["state"] != "completed":
            print(f"Skipped {run.name}: {info['state']}")
            continue
        config = OmegaConf.to_container(OmegaConf.load(run / "config.yaml"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))

        # 2. Group identical scientific settings; the training seed varies within a group.
        settings = {key: config[key] for key in ("data", "model", "training")}
        group = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
        rows.append(
            {
                "run": str(run),
                "group": group,
                "seed": config["seed"],
                "model": config["model"]["name"],
                "layers": config["model"]["object"]["num_layers"],
                **metrics,
            }
        )

    # 3. Keep raw rows and report counts so missing or repeated seeds stay visible.
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "runs.csv", index=False)
    summary = (
        frame.groupby(["group", "model", "layers"])
        .agg(
            runs=("seed", "size"),
            seeds=("seed", "nunique"),
            loss_mean=("valid_loss", "mean"),
            loss_std=("valid_loss", "std"),
            accuracy_mean=("valid_accuracy", "mean"),
            accuracy_std=("valid_accuracy", "std"),
        )
        .reset_index()
    )
    summary.to_csv(folder / "summary.csv", index=False)
    print(f"Completed runs: {len(frame)}")
    print(summary.to_string(index=False))

    # 4. Show individual runs and their mean rather than only an aggregate score.
    fig, ax = plt.subplots(figsize=(7, 4))
    for position, row in summary.iterrows():
        values = frame.loc[frame.group == row.group, "valid_accuracy"]
        ax.scatter([position] * len(values), values, alpha=0.6)
        ax.plot(position, row.accuracy_mean, "k_")
    ax.set_xticks(
        range(len(summary)),
        [f"{r.model}/{r.layers} {r.group[:6]}" for r in summary.itertuples()],
    )
    ax.set_ylabel("Validation accuracy (points: runs; line: mean)")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
