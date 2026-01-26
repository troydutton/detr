from math import sqrt

from torch import Tensor, nn

from utils.misc import take_annotation_from


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron or feedforward network.

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
        num_layers: int,
        output_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU,
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

    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the multi-layer perceptron.
        """

        # Weights in hidden layers get initialized uniformly to preserve variance
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, a=sqrt(5))
                nn.init.zeros_(m.bias.data)

        # The last layer is initialized to zero to prevent any initial bias
        nn.init.zeros_(self.mlp[-1].weight.data)
