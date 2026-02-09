from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import wandb
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import Criterion
from evaluators import Evaluator
from models import DETR


def train(
    model: DETR,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    criterion: Criterion,
    evaluator: Evaluator,
    train_data: DataLoader,
    val_data: DataLoader,
    num_epochs: int,
    accelerator: Accelerator,
    output_dir: Union[str, Path],
    save_period: int = 1,
    max_grad_norm: float = 0.1,
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
        accelerator: Accelerator object.
        output_dir: Parent directory to save the weights to.
        save_period: Period (in epochs) to save the model weights, optional.
        max_grad_norm: Maximum gradient norm for clipping, optional.
    """

    for epoch in range(num_epochs):
        # Train for a single epoch
        train_one_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            data=train_data,
            epoch=epoch,
            accelerator=accelerator,
            max_grad_norm=max_grad_norm,
        )

        # Evaluate the model
        val_losses, val_metrics = evaluate(
            model=model,
            criterion=criterion,
            evaluator=evaluator,
            data=val_data,
            epoch=epoch,
            accelerator=accelerator,
        )

        # Log the validation losses and learning rates for this epoch
        if accelerator.is_main_process:
            learning_rates = {group.get("name", i): group["lr"] for i, group in enumerate(optimizer.param_groups)}
            wandb.log({"val": {"loss": val_losses, "metric": val_metrics}, "lr": learning_rates}, step=wandb.run.step)

        # Save the model weights
        if (epoch + 1) % save_period == 0 or (epoch + 1) == num_epochs:
            checkpoint_dir = Path(output_dir) / f"{epoch + 1}"
            accelerator.save_state(checkpoint_dir)


def train_one_epoch(
    model: DETR,
    optimizer: Optimizer,
    criterion: Criterion,
    scheduler: _LRScheduler,
    data: DataLoader,
    epoch: int,
    accelerator: Accelerator,
    *,
    max_grad_norm: float = 0.1,
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
        accelerator: Accelerator object.
        max_grad_norm: Maximum gradient norm for clipping, optional.
    """

    # Set the model to training mode
    model.train()

    data = tqdm(data, desc=f"Training (Epoch {epoch})", dynamic_ncols=True, disable=not accelerator.is_main_process)
    for images, targets in data:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass and loss computation
        with accelerator.autocast():
            predictions = model(images)
            losses = criterion(predictions, targets, accelerator)

        # Backward pass & optimizer step
        accelerator.backward(losses["overall"])
        accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        # Step the learning rate scheduler every update
        scheduler.step()

        losses = {k: accelerator.gather(v).mean().item() for k, v in losses.items()}

        # Log the loss
        if accelerator.is_main_process:
            step_size = len(images) * accelerator.num_processes
            wandb.log({"train": {"loss": losses}}, step=wandb.run.step + step_size)

    accelerator.wait_for_everyone()


@torch.no_grad()
def evaluate(
    model: DETR,
    criterion: Criterion,
    evaluator: Evaluator,
    data: DataLoader,
    epoch: int,
    accelerator: Accelerator,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Evaluate a model.

    Args:
        model: Model to evaluate.
        criterion: Loss function.
        evaluator: Evaluator to compute metrics.
        data: Validation data.
        epoch: Current epoch.
        accelerator: Accelerator object.

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

    data = tqdm(data, desc=f"Validation (Epoch {epoch})", dynamic_ncols=True, disable=not accelerator.is_local_main_process)
    for images, targets in data:
        # Forward pass and loss computation
        with accelerator.autocast():
            predictions = model(images)

            batch_losses = criterion(predictions, targets)

        losses = {k: losses.get(k, 0) + accelerator.gather_for_metrics(v).mean().item() for k, v in batch_losses.items()}

        # Update the evaluator
        evaluator.update(predictions, targets, accelerator)

    accelerator.wait_for_everyone()

    # Calculate the average losses and metrics
    losses = {k: v / len(data) for k, v in losses.items()}

    metrics = evaluator.compute()

    return losses, metrics
