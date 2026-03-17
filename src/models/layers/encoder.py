from torch import nn

from models.backbone import Features
from models.deformable_attention import MultiHeadDeformableAttention
from models.layers.ffn import FFN
from utils.misc import take_annotation_from


class EncoderLayer(nn.Module):
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
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim, embed_dim, 2, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features: Features) -> Features:
        """
        Forward pass for a single transformer encoder layer.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
        """

        # Self-attention
        v = self.norm1(features.embed)
        q = k = v + features.pos
        features.embed = features.embed + self.dropout1(self.self_attention(q, k, v, need_weights=False)[0])

        # Feed forward network
        features.embed = features.embed + self.dropout2(self.ffn(self.norm2(features.embed)))

        return features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DeformableEncoderLayer(nn.Module):
    """
    Single deformable layer of the transformer encoder.

    Consists of a deformable self-attention layer followed by a feedforward network,
    both of which use pre-layer normalization for gradient stability.

    Args:
        embed_dim: Embedding dimension.
        ffn_dim: Feedforward network dimension.
        num_heads: Number of attention heads.
        num_points: Number of sampling points.
        num_levels: Number of feature levels.
        dropout: Dropout rate, optional.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_points: int,
        num_levels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadDeformableAttention(embed_dim, num_heads, num_levels, num_points)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim, embed_dim, 2, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features: Features) -> Features:
        """
        Forward pass for a single transformer encoder layer.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
        """

        # Self-attention
        v = self.norm1(features.embed)
        q = v + features.pos
        features.embed = features.embed + self.dropout1(
            self.self_attention(
                queries=q,
                query_reference=features.reference,
                features=v,
                dimensions=features.dimensions,
            )
        )

        # Feed forward network
        features.embed = features.embed + self.dropout2(self.ffn(self.norm2(features.embed)))

        return features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
