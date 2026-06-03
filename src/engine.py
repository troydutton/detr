import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import wandb
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import Criterion
from data import CocoDataset
from data.transforms import DiscreteRandomResize
from evaluators import Evaluator
from models import DETR
from utils.checkpoint import save_checkpoint


def train(
    model: DETR,
    ema_model: AveragedModel,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    criterion: Criterion,
    evaluator: Evaluator,
    train_data: DataLoader,
    finetune_data: DataLoader,
    val_data: DataLoader,
    batch_resize: DiscreteRandomResize,
    accelerator: Accelerator,
    output_dir: Union[str, Path],
    num_epochs: int,
    num_finetune_epochs: int = 0,
    num_warmup_epochs: int = 0,
    num_cooldown_epochs: int = 0,
    start_epoch: int = 0,
    save_period: int = 1,
    max_grad_norm: float = 0.1,
    *,
    enable_wandb: bool = True,
) -> None:
    """
    Train and evaluate a model for a number of epochs.


    The training schedule consists of warmup, active, cooldown, and finetuning periods defined below.

                    0               [W]                         [N - C]         [N - F]             [N]
                    |----------------|-----------------------------|---------------|-----------------|
    Heavy Aug.      |    [ OFF ]     |           [ ON ]            |    [ OFF ]    |     [ OFF ]     |
                    |................|.............................|...............|.................|
    Multi-Scale     |    [ ON ]      |           [ ON ]            |    [ ON ]     |     [ OFF ]     |
    Resizing        '----------------'-----------------------------'---------------'-----------------'

    Args:
        model: Model to train.
        ema_model: EMA model to update and evaluate.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        criterion: Loss function.
        evaluator: Evaluator to compute metrics.
        train_data: Training data.
        finetune_data: Fine-tuning data.
        val_data: Validation data.
        batch_resize: Batch-level random resize transformation.
        accelerator: Accelerator object.
        num_epochs: Number of epochs to train for.
        num_finetune_epochs: Number of epochs to fine-tune for at the end of training, optional.
        num_warmup_epochs: Number of epochs at the start of training to skip heavy augmentations, optional.
        num_cooldown_epochs: Number of epochs at the end of training to skip heavy augmentations, optional.
        start_epoch: Epoch to start training from, optional.
        output_dir: Parent directory to save the weights to.
        save_period: Period (in epochs) to save the model weights, optional.
        max_grad_norm: Maximum gradient norm for clipping, optional.
        enable_wandb: Whether to log to Weights & Biases, optional.
    """

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.perf_counter()

        if epoch == num_warmup_epochs:
            logging.info("Enabling heavy augmentations.")
        elif epoch == num_epochs - num_cooldown_epochs:
            logging.info("Disabling heavy augmentations.")
        if epoch == num_epochs - num_finetune_epochs:
            logging.info("Disabling all augmentations and multi-scale resizing.")

        data = train_data if epoch < num_epochs - num_finetune_epochs else finetune_data
        batch_resize = batch_resize if epoch < num_epochs - num_finetune_epochs else None

        if isinstance(data.dataset, CocoDataset):
            data.dataset.epoch = epoch

        # Train for a single epoch
        train_one_epoch(
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            data=data,
            epoch=epoch,
            accelerator=accelerator,
            max_grad_norm=max_grad_norm,
            batch_resize=batch_resize,
            enable_wandb=enable_wandb,
        )

        # Evaluate the model
        val_losses, val_metrics = evaluate(
            model=ema_model,
            criterion=criterion,
            evaluator=evaluator,
            data=val_data,
            epoch=epoch,
            accelerator=accelerator,
        )

        epoch_duration = time.perf_counter() - epoch_start_time

        # Log the losses, metrics, and learning rates for this epoch
        if accelerator.is_main_process and enable_wandb:
            learning_rates = {str(group["name"]).removesuffix(".no_decay"): group["lr"] for group in optimizer.param_groups}
            wandb.log({"val": {"loss": val_losses, "metric": val_metrics}, "lr": learning_rates}, commit=False)

        logging.info(f" Epoch {epoch + 1} | {timedelta(seconds=int(epoch_duration))} ".center(65, "="))
        logging.info(", ".join(f"{k}: {v * 100:.1f}" for k, v in val_metrics["overall"].items()))
        logging.info("=" * 65)

        # Save the model weights
        if (epoch + 1) % save_period == 0 or (epoch + 1) == num_epochs:
            checkpoint_dir = Path(output_dir) / f"{epoch + 1}"
            save_checkpoint(accelerator, checkpoint_dir, model, ema_model)

        torch.cuda.empty_cache()


def train_one_epoch(
    model: DETR,
    ema_model: AveragedModel,
    optimizer: Optimizer,
    criterion: Criterion,
    scheduler: _LRScheduler,
    data: DataLoader,
    epoch: int,
    accelerator: Accelerator,
    batch_resize: Optional[DiscreteRandomResize] = None,
    max_grad_norm: float = 0.1,
    *,
    enable_wandb: bool = True,
) -> None:
    """
    Train a model for a single epoch.

    Args:
        model: Model to train.
        ema_model: EMA model to update.
        optimizer: Optimizer.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        data: Training data.
        epoch: Current epoch.
        accelerator: Accelerator object.
        batch_resize: Batch-level random resize transformation, optional.
        max_grad_norm: Maximum gradient norm for clipping, optional.
        enable_wandb: Whether to log to Weights & Biases, optional.
    """

    # Set the model to training mode
    model.train()

    data = tqdm(data, desc=f"Training (Epoch {epoch + 1})", dynamic_ncols=True, disable=not accelerator.is_main_process, smoothing=0)
    for images, targets in data:
        with accelerator.accumulate(model):
            # Apply batch-level random resizing if enabled
            if batch_resize is not None:
                images, targets = batch_resize(images, targets)

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            predictions = model(images, targets)

            # Calculate the loss
            losses = criterion(predictions, targets, accelerator)

            # Backward pass
            accelerator.backward(losses["overall"])

            # Clip gradients and calculate the average losses across processes for logging
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                losses = {k: torch.mean(accelerator.reduce(v.detach(), reduction="mean")).item() for k, v in losses.items()}

            # Update the parameters and learning rate
            optimizer.step()
            scheduler.step()

            # Update the EMA model
            if accelerator.sync_gradients:
                ema_model.update_parameters(model)

            # Log the loss
            if accelerator.is_main_process and accelerator.sync_gradients and enable_wandb:
                wandb.log({"train": {"loss": losses}})

    accelerator.wait_for_everyone()


@torch.no_grad()
def evaluate(
    model: DETR,
    criterion: Criterion,
    evaluator: Evaluator,
    data: DataLoader,
    epoch: int,
    accelerator: Accelerator,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
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

    data = tqdm(data, desc=f"Validation (Epoch {epoch + 1})", dynamic_ncols=True, disable=not accelerator.is_main_process)
    for images, targets in data:
        # Forward pass
        predictions = model(images)

        # Calculate the loss
        batch_losses = criterion(predictions, targets, accelerator)

        losses = {k: losses.get(k, 0) + torch.mean(accelerator.reduce(v, reduction="mean")).item() for k, v in batch_losses.items()}

        # Update the evaluator
        evaluator.update(predictions, targets, accelerator)

    accelerator.wait_for_everyone()

    # Calculate the average losses and metrics
    losses = {k: v / len(data) for k, v in losses.items()}

    metrics = evaluator.compute()

    return losses, metrics
