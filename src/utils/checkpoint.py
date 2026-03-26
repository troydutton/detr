import logging
from pathlib import Path
from typing import Optional, Union

from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from torch.optim.swa_utils import AveragedModel


def save_checkpoint(
    accelerator: Accelerator,
    checkpoint_path: Union[str, Path],
    ema_model: Optional[AveragedModel] = None,
) -> None:
    """
    Saves the accelerator state and optionally the EMA model weights.

    Args:
        accelerator: Accelerator object.
        checkpoint_path: Path to save the checkpoint to.
        ema_model: The EMA model to checkpoint, optional.
    """
    accelerator.save_state(checkpoint_path)

    if ema_model is not None and accelerator.is_main_process:
        ema_path = Path(checkpoint_path) / "ema_model.safetensors"
        logging.info(f"EMA model weights saved in {ema_path}")
        state_dict = {k: v.cpu() for k, v in ema_model.state_dict().items()}
        save_file(state_dict, ema_path)


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
