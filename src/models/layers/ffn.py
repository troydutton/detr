from math import sqrt
from typing import Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.misc import take_annotation_from


class FFN(nn.Module):
    """
    Feedforward network, a.k.a multi-layer perceptron.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        num_layers: Number of layers.
        output_dim: Output dimension.
        dropout: Dropout rate, optional.
        activation: Activation function, optional.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        layers = []

        for i in range(num_layers - 1):
            in_dim = input_dim if i == 0 else hidden_dim

            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def forward(self, input_tensor: Tensor) -> Tensor:
        """
        Forward pass of the multi-layer perceptron.

        Args:
            input_tensor: Input tensor of shape (batch_size, input_dim).

        Returns:
            output_tensor: Output tensor of shape (batch_size, output_dim).
        """

        return self.mlp(input_tensor)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the multi-layer perceptron.
        """

        # Weights in hidden layers get initialized uniformly to preserve variance
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=sqrt(5))
                nn.init.zeros_(m.bias)

        # The last layer is initialized to zero to prevent any initial bias
        nn.init.zeros_(self.mlp[-1].weight)


class SwiGLUFFN(nn.Module):
    """
    Swish Gated Linear Unit Feedforward Network (SwiGLU) with one hidden layer.

    Args:
        input_dim: Input dimension.
        hidden_dim: Hidden dimension.
        output_dim: Output dimension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()

        # Reduce the hidden dimension for parameter parity and round for hardware alignment
        hidden_dim = (int(hidden_dim * 2 / 3) + 63) // 64 * 64

        self.input_proj = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=True)

        self._initialize_weights()

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Apply the SwiGLU feedforward network to the input embeddings.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            embeddings: Output embeddings with shape (batch_size, seq_length, embed_dim).
        """

        embeddings = self.input_proj(embeddings)

        gate, signal = embeddings.chunk(2, dim=-1)

        embeddings = F.silu(gate) * signal

        return self.output_proj(embeddings)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the multi-layer perceptron.
        """

        # The input projection is initialized to preserve variance
        nn.init.kaiming_uniform_(self.input_proj.weight, a=sqrt(5))
        nn.init.zeros_(self.input_proj.bias)

        # The output projection is initialized to zero to prevent any initial bias
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
