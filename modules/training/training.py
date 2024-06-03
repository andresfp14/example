import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from modules.utils import MetricAggregator, EarlyStoppingCustom
import logging

def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, cfg: DictConfig, device="cpu"):
    """
    Trains and evaluates a neural network model.

    Args:
        model (nn.Module): The neural network model to be trained and evaluated.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        cfg (DictConfig): Configuration object containing training parameters.
        device (str): The device to run the training on ("cpu" or "cuda").

    Returns:
        None: The function does not return any value.
    """

    ##############################
    # Device Setup
    ##############################
    device = torch.device(cfg.device)
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

    # Initialize MetricAggregator
    num_classes = len(train_loader.dataset.classes)
    metric_aggregator = MetricAggregator(
        num_classes=num_classes,
        metrics=cfg.metrics,
        device=device,
        loggers=loggers
    )

    # Early stopping and checkpointing
    early_stopping = EarlyStoppingCustom(**cfg.early_stopping_config)

    ##############################
    # Epoch Loop
    ##############################
    for epoch in range(cfg.max_epochs):
        ##############################
        # Training Loop
        ##############################
        model.train()
        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        for batch_idx, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            out = model(x)
            logprob = F.log_softmax(out, dim=1)
            y_hat_prob = torch.exp(logprob)
            loss = criterion(logprob, y)
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            # Update metrics
            metric_aggregator.step(y_hat_prob=y_hat_prob, y=y, loss=loss, epoch=torch.tensor(epoch+1), lr=lr, phase="train")

        # Compute and log metrics
        train_results = metric_aggregator.compute(phase="train")
        logger.info(f"Epoch {epoch+1} Train: {' '.join([f'{k}:{v:.3E}'.replace('_epoch','').replace('train_','') for k,v in train_results.items() if isinstance(v,float)])}")

        ##############################
        # Validation Loop
        ##############################
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)
                logprob = F.log_softmax(out, dim=1)
                y_hat_prob = torch.exp(logprob)
                val_loss = criterion(logprob, y)

                # Update metrics
                metric_aggregator.step(y_hat_prob=y_hat_prob, y=y, loss=loss, epoch=torch.tensor(epoch+1), lr=lr, phase="valid")

        # Compute and log metrics
        valid_results = metric_aggregator.compute(phase="valid")
        logger.info(f"Epoch {epoch+1} Valid: {' '.join([f'{k}:{v:.3E}'.replace('_epoch','').replace('valid_','') for k,v in valid_results.items() if isinstance(v,float)])}")

        ##############################
        # End of epoch
        ##############################
        metric_aggregator.reset(phase="train")
        metric_aggregator.reset(phase="valid")

        # Step the scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)

        # Check early stopping
        early_stopping(valid_results)
        if early_stopping.should_stop and cfg.min_epochs < epoch:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    ##############################
    # Saving Results
    ##############################
    for logger_ in loggers:
        logger_.save()
