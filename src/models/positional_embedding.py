import math

import torch
from torch import Tensor


def build_pos_embed(references: Tensor, num_channels: int, temperature: float = 10000.0) -> Tensor:
    """
    Generates sinusoidal positional embeddings for reference points/boxes.

    Args:
        references: Tensor of shape (batch_size, num_references, num_axes) containing normalized coordinates
        num_channels: Total number of channels for the positional embedding (must be divisible by 2 * num_axes)
        temperature: Base for the frequency exponential scale.

    Returns:
        pos: Positional embeddings with shape (batch_size, num_references, num_channels).
    """

    # Get batch information
    device = references.device
    num_axes = references.shape[-1]

    if num_channels % (2 * num_axes) != 0:
        raise ValueError(f"Total channels must be divisible by {2 * num_axes}, got {num_channels=}.")

    # Each axis gets an equal share of the channels
    channels_per_axis = num_channels // num_axes

    # Calculate frequency term for each channel: 1 / (temperature ^ (2c / num_channels))
    freq_term = 2 * (torch.arange(channels_per_axis, dtype=torch.float32, device=device) // 2)
    div_term = torch.exp(freq_term * (math.log(temperature) / channels_per_axis))

    # Create frequency grid, interleaving sin and cos terms for each axis
    pos = (references[..., None] * 2 * math.pi) / div_term
    pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-3)

    return pos
