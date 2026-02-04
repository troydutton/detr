import logging
from typing import Optional

import torch
from torch import device
from torch.amp import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from models import Model


def load_checkpoint(
    checkpoint: Optional[str],
    model: Model,
    optimizer: Optimizer,
    scaler: GradScaler,
    scheduler: _LRScheduler,
    device: device,
) -> int:
    """
    Load the objects necessary to resume training from a checkpoint.

    Args:
        checkpoint: Path to the checkpoint file.
        model: Model to load the state into.
        optimizer: Optimizer to load the state into.
        scaler: Gradient scaler to load the state into.
        scheduler: LR scheduler to load the state into.
        device: Device to map the checkpoint tensors to.

    Returns:
        start_epoch: Epoch to resume training from.
    """

    if checkpoint is None:
        return 0

    logging.info(f"Loading checkpoint from '{checkpoint}'.")

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    scheduler.load_state_dict(state["scheduler"])

    return state["epoch"] + 1
