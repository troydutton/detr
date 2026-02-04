from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import wandb
from torch import device
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import Criterion
from evaluators import Evaluator
from models import Model
from utils.misc import send_to_device


def train(
    model: Model,
    optimizer: Optimizer,
    scaler: GradScaler,
    scheduler: _LRScheduler,
    criterion: Criterion,
    evaluator: Evaluator,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int,
    device: device,
    output_dir: Union[str, Path],
    start_epoch: int = 0,
    save_period: int = 1,
    max_grad_norm: float = 0.1,
    amp: bool = False,
) -> None:
    """
    Train and evaluate a model for a number of epochs.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        scaler: Gradient scaler for AMP.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        evaluator: Evaluator to compute metrics.
        train_data: Training data.
        val_data: Validation data.
        num_epochs: Number of epochs to train for.
        device: Device to use.
        output_dir: Parent directory to save the weights to.
        start_epoch: Epoch to start training from, optional.
        save_period: Period (in epochs) to save the model weights, optional.
        max_grad_norm: Maximum gradient norm for clipping, optional.
        amp: Whether to use AMP, optional.
    """

    for epoch in range(start_epoch, num_epochs):
        # Train for a single epoch
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            scheduler=scheduler,
            data=train_data,
            epoch=epoch,
            device=device,
            max_grad_norm=max_grad_norm,
            amp=amp,
        )

        # Evaluate the model
        val_losses, val_metrics = evaluate(
            model=model,
            criterion=criterion,
            evaluator=evaluator,
            data=val_data,
            epoch=epoch,
            device=device,
            amp=amp,
        )

        # Log the validation losses and learning rates for this epoch
        learning_rates = {group.get("name", i): group["lr"] for i, group in enumerate(scheduler.optimizer.param_groups)}
        wandb.log({"val": {"loss": val_losses, "metric": val_metrics}, "lr": learning_rates}, step=wandb.run.step)

        # Save the model weights
        if (epoch + 1) % save_period == 0 or (epoch + 1) == num_epochs:
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            }
            torch.save(state, output_dir / f"{epoch + 1}.pt")


def train_one_epoch(
    model: Model,
    optimizer: Optimizer,
    scaler: GradScaler,
    criterion: Criterion,
    scheduler: _LRScheduler,
    data: DataLoader,
    epoch: int,
    device: device,
    *,
    max_grad_norm: float = 0.1,
    amp: bool = False,
) -> None:
    """
    Train a model for a single epoch.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        scaler: Gradient scaler for AMP.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        data: Training data.
        epoch: Current epoch.
        device: Device to train on.
        max_grad_norm: Maximum gradient norm for clipping, optional.
        amp: Whether to use AMP, optional.
    """

    # Set the model to training mode
    model.train()

    for images, targets in tqdm(data, desc=f"Training (Epoch {epoch})", dynamic_ncols=True):
        # Zero the gradients
        optimizer.zero_grad()

        # Send the batch to the training device
        images, targets = send_to_device(images, device), send_to_device(targets, device)

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            predictions = model(images)
            losses = criterion(predictions, targets)

        # Backward pass
        scaler.scale(losses["overall"]).backward()

        # Unscale for gradient clipping
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Step the learning rate scheduler every update
        scheduler.step()

        # Log the loss
        wandb.log({"train": {"loss": losses}}, step=wandb.run.step + len(images))


@torch.no_grad()
def evaluate(
    model: Model,
    criterion: Criterion,
    evaluator: Evaluator,
    data: DataLoader,
    epoch: int,
    device: device,
    amp: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate a model.

    Args:
        model: Model to evaluate.
        criterion: Loss function.
        evaluator: Evaluator to compute metrics.
        data: Validation data.
        epoch: Current epoch.
        device: Device to evaluate on.
        amp: Whether to use AMP, optional.

    Returns:
        losses: Dictionary of average losses.
        #### metrics
        Dictionary of evaluation metrics.
    """

    # Set the model to evaluation mode
    model.eval()
    evaluator.reset()

    # Keep track of the running loss
    losses = {}

    for images, targets in tqdm(data, desc=f"Validation (Epoch {epoch})", dynamic_ncols=True):
        # Send the batch to the evaluation device
        images, targets = send_to_device(images, device), send_to_device(targets, device)

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            predictions = model(images)
            batch_losses = criterion(predictions, targets)

        # Update the running loss
        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        # Update the evaluator
        evaluator.update(predictions, targets)

    # Calculate the average losses and metrics
    losses = {k: v / len(data) for k, v in losses.items()}
    metrics = evaluator.compute()

    return losses, metrics
