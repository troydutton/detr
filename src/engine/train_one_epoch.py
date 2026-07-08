import logging
from typing import Optional

import torch
import wandb
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import Criterion
from data.transforms import DiscreteRandomResize
from models import DETR


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

    unwrapped_model: DETR = accelerator.unwrap_model(model)
    named_parameters = list(unwrapped_model.named_parameters())

    data = tqdm(data, desc=f"Training (Epoch {epoch + 1})", dynamic_ncols=True, disable=not accelerator.is_main_process, smoothing=0)
    for images, targets in data:
        with accelerator.accumulate(model):
            # Apply batch-level random resizing if enabled
            if batch_resize is not None:
                images, targets = batch_resize(images, targets)

            # Forward pass
            predictions = model(images, targets)

            # Calculate the loss
            losses = criterion(predictions, targets, accelerator)

            # Backward pass
            accelerator.backward(losses["overall"])

            # Clip gradients, check for NaN/Inf values, and skip the update if necessary
            should_update = True
            if accelerator.sync_gradients:
                # Determine if each parameter's gradient is finite
                named_gradients = [(name, param.grad) for name, param in named_parameters if param.grad is not None]
                is_finite = torch.stack([torch.isfinite(grad).all() for _, grad in named_gradients])

                # If any gradients are NaN or Inf, skip the optimizer step and log a warning
                if is_finite.all():
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                else:
                    should_update = False

                    bad_params = [name for (name, _), finite in zip(named_gradients, is_finite.tolist()) if not finite]
                    logging.warning(f"Skipping optimizer step due to NaN/Inf gradients in: {', '.join(bad_params)}")

            # Update the parameters and learning rate if the gradients are valid
            if should_update:
                optimizer.step()
                scheduler.step()

            # Zero the gradients
            optimizer.zero_grad(set_to_none=True)

            # Update the EMA model
            if accelerator.sync_gradients and should_update:
                ema_model.update_parameters(model)

            # Log the loss
            if accelerator.sync_gradients and should_update and enable_wandb:
                losses = {k: torch.mean(accelerator.reduce(v.detach(), reduction="mean")).item() for k, v in losses.items()}

                if accelerator.is_main_process:
                    wandb.log({"train": {"loss": losses}})

    accelerator.wait_for_everyone()
