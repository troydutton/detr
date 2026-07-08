from typing import Dict, Tuple

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from criterion import Criterion
from evaluators import Evaluator
from models import DETR


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
