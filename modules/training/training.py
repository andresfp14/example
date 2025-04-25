import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import logging

def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, cfg: DictConfig):
    """
    Trains and evaluates a neural network model.

    Args:
        model (nn.Module): The neural network model to be trained and evaluated.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        cfg (DictConfig): Configuration object containing training parameters.

    Returns:
        None: The function does not return any value.
    """

    ##############################
    # Device Setup
    ##############################
    device = "cuda" if (cfg.device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device)

    ##############################
    # Object Instantiation
    ##############################
    # Instantiate loss function
    criterion = hydra.utils.instantiate(cfg.loss)

    # Instantiate optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    # Instantiate scheduler
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer)

    # Initialize loggers
    loggers = [hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.loggers.values()]
    logger = logging.getLogger("training")

    # scaler for automatic mixed precision
    scaler = torch.amp.GradScaler(device)

    ##############################
    # Epoch Loop
    ##############################
    for epoch in range(cfg.epochs):
        ##############################
        # Training Loop
        ##############################
        model.train()
        train_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device):
                out = model(x)
                logprob = F.log_softmax(out, dim=1)
                loss = criterion(logprob, y)
                loss_acc = loss / cfg.gradient_accumulation_steps

            # Accumulates scaled gradients.
            scaler.scale(loss_acc).backward()

            # Gradient accumulation
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ##############################
        # Validation Loop
        ##############################
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(valid_loader):
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device):
                    out = model(x)
                    logprob = F.log_softmax(out, dim=1)
                    loss = criterion(logprob, y)
                val_loss += loss.item()

        val_loss /= len(valid_loader)

        # Log the loss
        for logger_ in loggers:
            logger_.log_metrics({"train_loss": train_loss, "valid_loss": val_loss}, step=epoch)
        logger.info(f"Epoch {epoch+1} Train Loss: {train_loss:.4f} Valid Loss: {val_loss:.4f}")

        ##############################
        # End of epoch
        ##############################
        # Step the scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

    ##############################
    # Saving Results
    ##############################
    for logger_ in loggers:
        logger_.save()
        logger_.finalize("success")
    
    return {"train_loss": train_loss, "valid_loss": val_loss}
