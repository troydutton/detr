import torch
from torch import Tensor, nn

from utils.misc import take_annotation_from


class TargetGatingLayer(nn.Module):
    """
    A dynamic gating mechanism used to blend features with their residual updates.

    We use it to replace standard addition-based residual connections in the transformer.
    It allows features to dynamically switch their focus on different targets across layers,
    compensating for the removal of computationally expensive projection layers.

    Args:
        embed_dim: Embedding dimension of the features.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.gate = nn.Linear(2 * embed_dim, 2 * embed_dim)

        self._initialize_weights()

    def forward(self, current: Tensor, residual: Tensor) -> Tensor:
        """
        Computes the gated combination of the current features and the residual update.

        Args:
            current: Current features with shape (..., embed_dim).
            residual: Feature residuals to be blended with shape (..., embed_dim).

        Returns:
            updated: Updated features with shape (..., embed_dim).
        """

        logits = self.gate(torch.cat([current, residual], dim=-1))

        current_weight, residual_weight = torch.sigmoid(logits).chunk(2, dim=-1)

        return (current_weight * current) + (residual_weight * residual)

    @torch.no_grad()
    def _initialize_weights(self):
        """
        Initialize the weights of the target gating layer.
        """

        # Initialize the gate to produce equal weights
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
