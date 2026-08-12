import logging
import time
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import wandb
from accelerate import Accelerator
from torch import Tensor
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
    micro_batch_size: int,
    cumulative_step: int = 0,
    cumulative_images: int = 0,
    batch_resize: Optional[DiscreteRandomResize] = None,
    max_grad_norm: float = 0.1,
    *,
    enable_wandb: bool = True,
) -> Tuple[int, int]:
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
        micro_batch_size: Number of images to process in each micro batch.
        cumulative_step: Cumulative number of optimizer steps taken, optional.
        cumulative_images: Cumulative number of images seen, optional.
        batch_resize: Batch-level random resize transformation, optional.
        max_grad_norm: Maximum gradient norm for clipping, optional.
        enable_wandb: Whether to log to Weights & Biases, optional.

    Returns:
        cumulative_step: Cumulative number of optimizer steps taken.
        #### cumulative_images
        Cumulative number of images seen.
    """

    # Set the model to training mode
    model.train()

    unwrapped_model: DETR = accelerator.unwrap_model(model)
    named_parameters = list(unwrapped_model.named_parameters())

    # Throughput counters
    images_in_window, objects_in_window = 0, 0
    window_start_time = time.perf_counter()

    data = tqdm(data, desc=f"Training (Epoch {epoch + 1})", dynamic_ncols=True, disable=not accelerator.is_main_process, smoothing=0)
    for images, targets in data:
        # Apply batch-level random resizing if enabled
        if batch_resize is not None:
            images, targets = batch_resize(images, targets)

        # Accumulate throughput statistics over the window
        images_in_window += len(images)
        objects_in_window += sum(len(target["boxes"]) for target in targets)

        # Accumulate the gradient over micro batches
        image_micro_batches = images.split(micro_batch_size)
        target_micro_batches = [targets[i : i + micro_batch_size] for i in range(0, len(targets), micro_batch_size)]
        micro_batches = list(zip(image_micro_batches, target_micro_batches))

        losses = {}
        for index, (micro_images, micro_targets) in enumerate(micro_batches, start=1):
            # Only synchronize the gradients across processes on the final micro batch
            sync_context = nullcontext() if index == len(micro_batches) else accelerator.no_sync(model)

            with sync_context:
                # Forward pass
                predictions = model(micro_images, micro_targets)

                # Calculate the loss
                micro_losses = criterion(predictions, micro_targets, accelerator, normalizer_targets=targets)

                # Backward pass
                accelerator.backward(micro_losses["overall"])

            losses = {name: losses.get(name, 0.0) + loss.detach() for name, loss in micro_losses.items()}

        # Clip gradients, check for NaN/Inf values, and skip the update if necessary
        should_update = True

        # Determine if each parameter's gradient is finite
        named_gradients = [(name, param.grad) for name, param in named_parameters if param.grad is not None]
        is_finite = torch.stack([torch.isfinite(grad).all() for _, grad in named_gradients])

        # If any gradients are NaN or Inf, skip the optimizer step and log a warning
        if is_finite.all():
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()
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

        # If this is an invalid gradient step, skip logging and EMA
        if not should_update:
            continue

        # Update the EMA model
        ema_model.update_parameters(model)

        # Log the metrics for this step
        if enable_wandb:
            # Total throughput across all processes
            window_duration = time.perf_counter() - window_start_time
            throughput_counts = torch.tensor([images_in_window, objects_in_window], device=accelerator.device)
            throughput_counts: Tensor = accelerator.reduce(throughput_counts, reduction="sum")
            images_in_window, objects_in_window = throughput_counts.tolist()

            # Update the cumulative step and image counters
            cumulative_step += 1
            cumulative_images += images_in_window

            # Mean losses across all processes
            losses = {k: torch.mean(accelerator.reduce(v.detach(), reduction="mean")).item() for k, v in losses.items()}

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "train": {
                            "step": cumulative_step,
                            "images": cumulative_images,
                            "loss": losses,
                            "grad_norm": grad_norm,
                            "images_per_second": images_in_window / window_duration,
                            "steps_per_second": 1 / window_duration,
                            "objects_per_image": objects_in_window / images_in_window,
                        },
                    }
                )

        # Reset the throughput counters for the next window
        images_in_window, objects_in_window = 0, 0
        window_start_time = time.perf_counter()

    accelerator.wait_for_everyone()

    return cumulative_step, cumulative_images
