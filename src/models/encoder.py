from hydra.utils import instantiate
from torch import nn

from models.backbone import Features
from utils.misc import take_annotation_from


class TransformerEncoder(nn.Module):
    """
    Transformer encoder composed of a stack of N encoder layers.

    Args:
        num_layers: Number of encoder layers.
        embed_dim: Embedding dimension.
        kwargs: Arguments to construct the encoder layers.
            See `models.layers.encoder.EncoderLayer` or `models.layers.encoder.DeformableEncoderLayer`.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        **kwargs,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList([instantiate(kwargs["layer"]) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: Features) -> Features:
        """
        Forward pass for the transformer encoder.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
        """

        assert features.embed.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.embed.shape=}"

        for layer in self.layers:
            features = layer(features)

        # Final normalization because we're using Pre-LN layers
        features.embed = self.norm(features.embed)

        return features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
