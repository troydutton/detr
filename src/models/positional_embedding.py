import torch
from torch import Tensor
import math


def generate_positional_embedding_sine(input_tensor: Tensor, temperature: float = 10000.0) -> Tensor:
    """
    Generates 2D sinusoidal positional embeddings for a given input tensor.

    Args:
        input_tensor: Input wth shape (batch_size, height, width, channels).
        temperature: Base for the frequency exponential scale.

    Returns:
        pos_embeddings: Positional embeddings with shape (batch_size, height, width, channels).
    """
    batch_size, height, width, total_channels = input_tensor.shape

    if total_channels % 4 != 0:
        raise ValueError(f"Total channels must be divisible by 4, got {total_channels = }.")

    # We use half the total channels for Height and half for Width
    dim_per_axis = total_channels // 2

    # Calculate frequency terms: 1 / (10000 ^ (2i / d_model))
    # We use log space for numerical stability
    freq_indices = torch.arange(0, dim_per_axis, 2).float()
    div_term = torch.exp(freq_indices * -(math.log(temperature) / dim_per_axis))

    # Generate linear positions for each axis
    y_coords = torch.arange(height).float()
    x_coords = torch.arange(width).float()

    # Create grids for broadcasting
    y_sin_input = torch.outer(y_coords, div_term)  # (height, dim_per_axis // 2)
    x_sin_input = torch.outer(x_coords, div_term)  # (width, dim_per_axis // 2)

    # Generate sine/cosine components for vertical axis
    y_emb = torch.zeros(height, dim_per_axis)
    y_emb[:, 0::2] = torch.sin(y_sin_input)
    y_emb[:, 1::2] = torch.cos(y_sin_input)

    # Generate sine/cosine components for horizontal axis
    x_emb = torch.zeros(width, dim_per_axis)
    x_emb[:, 0::2] = torch.sin(x_sin_input)
    x_emb[:, 1::2] = torch.cos(x_sin_input)

    # Expand and concatenate to create (height, width, total_channels)
    pos_embeddings = torch.cat([y_emb.unsqueeze(1).repeat(1, width, 1), x_emb.unsqueeze(0).repeat(height, 1, 1)], dim=-1)

    # Repeat across the batch dimension
    pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return pos_embeddings.to(input_tensor.device)
