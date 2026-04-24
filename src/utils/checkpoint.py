from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch
from accelerate import Accelerator
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.optim.swa_utils import AveragedModel

if TYPE_CHECKING:
    from models import DETR

VALID_WEIGHT_FILES = ["ema_model.safetensors", "model.safetensors"]


def save_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: Union[str, Path],
    model: Optional[DETR] = None,
    ema_model: Optional[AveragedModel] = None,
) -> None:
    """
    Saves the accelerator state and optionally the EMA model weights.

    Args:
        accelerator: Accelerator object.
        checkpoint_path: Path to save the checkpoint to.
        model: The model to checkpoint, optional.
        ema_model: The EMA model to checkpoint, optional.
    """
    accelerator.save_state(checkpoint_path)

    # Retrieve metadata from the model if provided
    metadata = None
    if model is not None:
        model = accelerator.unwrap_model(model)
        metadata = {"categories": json.dumps(model.categories)}

    if model is not None and accelerator.is_main_process:
        model_path = Path(checkpoint_path) / "model.safetensors"
        state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, model_path, metadata=metadata)

    if ema_model is not None and accelerator.is_main_process:
        ema_path = Path(checkpoint_path) / "ema_model.safetensors"
        logging.info(f"EMA model weights saved in {ema_path}")
        state_dict = {k: v.cpu().contiguous() for k, v in ema_model.state_dict().items()}
        save_file(state_dict, ema_path, metadata=metadata)


def load_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: Union[str, Path],
    ema_model: Optional[AveragedModel] = None,
) -> None:
    """
    Loads the accelerator state and optionally the EMA model weights.

    Args:
        accelerator: Accelerator object.
        checkpoint_path: Path to load the checkpoint from.
        ema_model: The EMA model to load to, optional.
    """

    logging.info(f"Loading checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path)

    if ema_model is not None:
        ema_path = Path(checkpoint_path) / "ema_model.safetensors"
        ema_model.load_state_dict(load_file(ema_path, device="cpu"))
        logging.info("All EMA model weights loaded successfully")


def load_state_dict(pretrained_weights: Union[str, Path]) -> Dict[str, Tensor]:
    """
    Load model weights from a pretrained weights file.

    Args:
        pretrained_weights: Path to a pretrained weights file or an accelerate checkpoint directory.


    Args:
        pretrained_weights: Path to a pretrained weights file or an accelerate checkpoint directory.

    Returns:
        state_dict: Model weights.
    """

    # Resolve the path to the model weights file
    pretrained_weights = _get_weight_path(pretrained_weights)

    logging.info(f"Loading pretrained weights from '{pretrained_weights}'.")

    if pretrained_weights.suffix in [".pt", ".pth"]:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    elif pretrained_weights.suffix == ".safetensors":
        state_dict = load_file(pretrained_weights, device="cpu")
    else:
        raise ValueError(f"Unsupported pretrained weights format: {pretrained_weights}")

    # Strip module prefix and EMA metadata if present
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items() if k != "n_averaged"}

    return state_dict


def load_metadata(pretrained_weights: Union[str, Path]) -> Dict[str, str]:
    """
    Load metadata from a pretrained weights file.

    Args:
        pretrained_weights: Path to a pretrained weights file or an accelerate checkpoint directory.

    Returns:
        metadata: Metadata contained in the pretrained weights file.
    """

    # Resolve the path to the model weights file
    pretrained_weights = _get_weight_path(pretrained_weights)

    logging.info(f"Loading metadata from '{pretrained_weights}'.")

    if pretrained_weights.suffix != ".safetensors":
        raise NotImplementedError(f"Loading metadata from {pretrained_weights.suffix} files is not supported.")

    metadata = safe_open(pretrained_weights, framework="pt", device="cpu").metadata()

    return metadata


def _get_weight_path(pretrained_weights: Union[str, Path]) -> Path:
    """
    Find the path to the model weights file given a file or accelerate checkpoint directory.

    If the provided path is a directory, we attempt to find the model weights by searching
    for known weight file names (`VALID_WEIGHT_FILES`) in the directory and its subdirectories.
    This is useful for loading from the most recent accelerate checkpoint directory.

    Args:
        pretrained_weights: Path to a pretrained weights file or an accelerate checkpoint directory.

    Returns:
        pretrained_weights: Path to the model weights file.
    """

    pretrained_weights = Path(pretrained_weights)

    # Attempt to find the model weights in an accelerate checkpoint if a directory is provided
    if pretrained_weights.is_dir():

        def find_weight_file(directory: Path) -> Optional[Path]:
            for file in VALID_WEIGHT_FILES:
                if (directory / file).exists():
                    return directory / file
            return None

        # Search for weight files in the provided directory
        weight_path = find_weight_file(pretrained_weights)

        # If no weight file is found, search for checkpoint subdirectories and look for weight files there
        if weight_path is None:
            checkpoints = [d for d in pretrained_weights.iterdir() if d.is_dir() and find_weight_file(d) is not None]
            checkpoints.sort(key=lambda d: int(d.name) if d.name.isdecimal() else 0)

            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint directories containing model weights found in '{pretrained_weights}'.")

            weight_path = find_weight_file(checkpoints[-1])

        pretrained_weights = weight_path

    return pretrained_weights
