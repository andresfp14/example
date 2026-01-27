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
    # Step 1: Device setup
    ##############################
    device = "cuda" if (cfg.device=="cuda" and torch.cuda.is_available()) else "cpu"
    model.to(device)

    ##############################
    # Step 2: Instantiate training objects
    ##############################
    # 1) Loss, 2) optimizer, 3) scheduler
    criterion = hydra.utils.instantiate(cfg.loss)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer)
    # 4) Loggers (e.g., tensorboard/CSV)
    loggers = [hydra.utils.instantiate(logger_cfg) for logger_cfg in cfg.loggers.values()]
    logger = logging.getLogger("training")
    # 5) AMP scaler for mixed precision
    scaler = torch.amp.GradScaler(device)

    ##############################
    # Step 3: Epoch loop
    ##############################
    for epoch in range(cfg.epochs):
        ##############################
        # 3.1 Training loop
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

            # Gradient accumulation step
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        ##############################
        # 3.2 Validation loop
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
        # 3.3 End-of-epoch updates
        ##############################
        # Step the scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

    ##############################
    # Step 4: Finalize & save logs
    ##############################
    for logger_ in loggers:
        logger_.save()
        logger_.finalize("success")
    
    return {"train_loss": train_loss, "valid_loss": val_loss}
