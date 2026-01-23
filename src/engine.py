import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import ModelType
from criterion import CriterionType
from evaluators import EvaluatorType
from utils.misc import send_to_device
from torch import device
from torch.optim.lr_scheduler import _LRScheduler


def train(
    model: ModelType,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    criterion: CriterionType,
    evaluator: EvaluatorType,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int,
    device: device,
    output_dir: Path,
    save_period: int = 1,
) -> None:
    """
    Train and evaluate a model for a number of epochs.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        evaluator: Evaluator to compute metrics.
        train_data: Training data.
        val_data: Validation data.
        num_epochs: Number of epochs to train for.
        device: Device to use.
        output_dir: Parent directory to save the weights to.
        save_period: Period (in epochs) to save the model weights.
    """

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Train for a single epoch
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            data=train_data,
            epoch=epoch,
            device=device,
        )

        # Evaluate the model
        val_losses, val_metrics = evaluate(
            model=model,
            criterion=criterion,
            evaluator=evaluator,
            data=val_data,
            epoch=epoch,
            device=device,
        )

        # Log the validation losses
        wandb.log({"val": {"loss": val_losses, "metric": val_metrics}}, step=wandb.run.step)

        # Save the model weights
        if (epoch + 1) % save_period == 0 or (epoch + 1) == num_epochs:
            torch.save(model.state_dict(), output_dir / f"{epoch}.pt")


def train_one_epoch(
    model: ModelType,
    optimizer: Optimizer,
    criterion: CriterionType,
    scheduler: _LRScheduler,
    data: DataLoader,
    epoch: int,
    device: device,
) -> None:
    """
    Train a model for a single epoch.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        data: Training data.
        epoch: Current epoch.
        device: Device to train on.
    """

    # Set the model to training mode
    model.train()

    for images, targets in tqdm(data, desc=f"Training (Epoch {epoch})", dynamic_ncols=True):
        # Zero the gradients
        optimizer.zero_grad()

        # Send the batch to the training device
        images, targets = send_to_device(images, device), send_to_device(targets, device)

        # Forward pass
        predictions = model(images)

        # Backward pass and optimization step
        losses = criterion(predictions, targets)
        losses["overall"].backward()
        optimizer.step()

        # Step the learning rate scheduler every update
        scheduler.step()

        # Log the loss
        wandb.log({"train": {"loss": losses}}, step=wandb.run.step + len(images))



@torch.no_grad()
def evaluate(
    model: ModelType,
    criterion: CriterionType,
    evaluator: EvaluatorType,
    data: DataLoader,
    epoch: int,
    device: device,
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
        predictions = model(images)

        # Compute the loss & update the running loss
        batch_losses = criterion(predictions, targets)
        losses = {k: losses.get(k, 0) + v.item() for k, v in batch_losses.items()}

        # Update the evaluator
        evaluator.update(predictions, targets)

    # Calculate the average losses and metrics
    losses = {k: v / len(data) for k, v in losses.items()}
    metrics = evaluator.evaluate()

    return losses, metrics
