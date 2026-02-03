from torch import Tensor, nn

from models.layers import MultiLayerPerceptron
from utils.misc import take_annotation_from


class TransformerEncoder(nn.Module):
    """
    Transformer encoder composed of a stack of N encoder layers.

    Args:
        num_layers: Number of encoder layers.
        embed_dim: Embedding dimension.
        ffn_dim: Feedforward network dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate, optional.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=embed_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: Tensor, positional_embeddings: Tensor = None) -> Tensor:
        """
        Forward pass for the transformer encoder.

        If provided, positional encodings are added to the queries and keys in the
        self-attention operation at every layer to maintain spatial context.

        Args:
            features: Features with shape (batch_size, num_features, embed_dim).
            positional_embeddings: Positional embeddings with shape (batch_size, num_features, embed_dim).

        Returns:
            features: Features with the same shape as the input.
        """

        for layer in self.layers:
            features = layer(features, positional_embeddings)

        # Final normalization because we're using Pre-LN layers
        features = self.norm(features)

        return features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of the transformer encoder.

    Consists of a self-attention layer followed by a feedforward network, both of which
    use pre-layer normalization for gradient stability.

    Args:
        embed_dim: Embedding dimension.
        ffn_dim: Feedforward network dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate, optional.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MultiLayerPerceptron(
            input_dim=embed_dim,
            hidden_dim=ffn_dim,
            output_dim=embed_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features: Tensor, feature_pos: Tensor = None) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.

        Args:
            features: Features with shape (batch_size, num_features, embed_dim).
            feature_pos: Feature positional embeddings with shape (batch_size, num_features, embed_dim).

        Returns:
            features: Features with the same shape as the input.
        """

        assert features.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.shape=}"

        # Self-attention
        v = self.norm1(features)
        q = k = v if feature_pos is None else v + feature_pos
        features = features + self.dropout1(self.self_attention(q, k, v, need_weights=False)[0])

        # Feed forward network
        features = features + self.dropout2(self.ffn(self.norm2(features)))

        return features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
