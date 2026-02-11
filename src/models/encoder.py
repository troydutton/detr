from typing import Optional

import torch
from hydra.utils import instantiate
from torch import nn

from models.backbone import Features
from models.decoder import Queries
from models.deformable_attention import MultiHeadDeformableAttention
from models.layers import FFN
from utils.misc import take_annotation_from


class TransformerEncoder(nn.Module):
    """
    Transformer encoder composed of a stack of N encoder layers.

    Args:
        num_layers: Number of encoder layers.
        embed_dim: Embedding dimension.
        kwargs: Arguments to construct the encoder layers.
            See `models.encoder.EncoderLayer` or `models.encoder.DeformableEncoderLayer`.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        *,
        two_stage: bool = False,
        num_queries: Optional[int] = None,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.two_stage = two_stage

        self.layers = nn.ModuleList([instantiate(kwargs["layer"]) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        if self.two_stage:
            self.bbox_head = FFN(embed_dim, embed_dim, output_dim=4, num_layers=3)
            self.class_head = nn.Linear(embed_dim, num_classes)

    def forward(self, features: Features) -> Features:
        """
        Forward pass for the transformer encoder.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
            #### queries
            Initial object query proposals with shape (batch_size, num_queries, embed_dim).
        """

        for layer in self.layers:
            features = layer(features)

        # Final normalization because we're using Pre-LN layers
        features.embed = self.norm(features.embed)

        # If using two-stage, we generate object query proposals from the encoder features
        queries = self.generate_query_proposals(features) if self.two_stage else None

        return features, queries

    def generate_query_proposals(self, features: Features) -> Queries:
        # Get batch information
        device = features.embed.device

        # Score each feature by its maximum class probability
        scores = torch.sigmoid(self.class_head(features.embed)).max(dim=-1).values

        # Gather the top-k features
        topk_indices = scores.topk(self.num_queries, dim=1).indices
        topk_indices = topk_indices.unsqueeze(-1).expand(-1, -1, features.embed.size(-1))
        feature_embed = torch.gather(features.embed, dim=1, index=topk_indices)

        # The reference boxes use the feature pixel's position as the center point
        # and a level-based width and height (scale * 2^level)
        xy = torch.gather(features.reference, dim=1, index=topk_indices)
        levels = torch.cat([torch.full((w * h,), l, device=device) for l, (w, h) in enumerate(features.dimensions)])
        levels = torch.gather(levels, dim=0, index=topk_indices.squeeze(-1))
        wh = (0.05 * (2**levels)).unsqueeze(-1).expand(-1, 2)
        query_ref = torch.cat([xy, wh], dim=-1)

        # Refine the reference in logit space
        offsets = self.bbox_head(feature_embed)
        query_ref = (query_ref.logit(1e-5) + offsets).sigmoid()

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


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
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
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

        assert features.embed.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.embed.shape=}"

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
        self.self_attention = MultiHeadDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
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

        assert features.embed.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.embed.shape=}"

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
