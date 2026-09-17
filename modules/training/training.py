"""Supervised classification with sample-weighted metrics."""

import csv
from pathlib import Path

import hydra
import torch
from torch import nn


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict[str, float]:
    # 1. Disable training behavior and accumulate metrics over individual samples.
    model.eval()
    loss_sum, correct, samples = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        samples += len(y)
    # 2. Weight every sample equally, including the final partial batch.
    return {"loss": loss_sum / samples, "accuracy": correct / samples}


def train_model(model, train_loader, valid_loader, cfg, device, folder: Path):
    """Save the best validation checkpoint; return its validation metrics."""
    # 1. Build the optimizer and validation-driven learning-rate scheduler.
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer)
    best_loss, best = float("inf"), None
    # 2. Keep an epoch-by-epoch record while training.
    with (folder / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["epoch", "train_loss", "valid_loss", "valid_accuracy"]
        )
        writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            # 3. Accumulate summed gradients over the configured number of batches.
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss, total_samples, group_samples = 0.0, 0, 0
            for batch, (x, y) in enumerate(train_loader, 1):
                x, y = x.to(device), y.to(device)
                loss = nn.functional.cross_entropy(model(x), y, reduction="sum")
                loss.backward()
                group_samples += len(y)
                total_samples += len(y)
                total_loss += loss.item()
                # 4. Normalize by the actual group size, including its last partial batch.
                if batch % cfg.accumulation_steps == 0 or batch == len(train_loader):
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(group_samples)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    group_samples = 0
            # 5. Measure validation performance without touching the test set.
            valid = evaluate(model, valid_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": total_loss / total_samples,
                "valid_loss": valid["loss"],
                "valid_accuracy": valid["accuracy"],
            }
            writer.writerow(row)
            stream.flush()
            print(row, flush=True)
            # 6. Save the best validation model and update the learning rate.
            if valid["loss"] < best_loss:
                best_loss, best = valid["loss"], row
                torch.save(model.state_dict(), folder / "weights.pt")
            scheduler.step(valid["loss"])
    return best
