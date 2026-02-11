from dataclasses import dataclass
from typing import Optional

import torch
from hydra.utils import instantiate
from torch import Tensor, nn

from models.backbone import Features
from models.deformable_attention import MultiHeadDeformableAttention
from models.layers import FFN
from utils.misc import take_annotation_from


@dataclass
class Queries:
    embed: Tensor
    pos: Tensor
    reference: Tensor


class TransformerDecoder(nn.Module):
    """
    Transformer decoder composed of a stack of N decoder layers.

    Args:
        num_layers: Number of decoder layers.
        embed_dim: Embedding dimension.
        num_queries: Number of object queries.
        two_stage: Initialize object queries using encoder proposals or learned parameters, optional.
        return_intermediates: Whether to return intermediate transformer outputs, optional.
        kwargs: Arguments to construct the decoder layers.
            See `models.decoder.DecoderLayer` or `models.decoder.DeformableDecoderLayer`.

    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_queries: int,
        *,
        two_stage: bool = False,
        return_intermediates: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.return_intermediates = return_intermediates
        self.two_stage = two_stage

        # In single-stage, we initialize the object queries as image-agnostic learned parameters
        if not self.two_stage:
            self.queries = nn.Embedding(num_queries, embed_dim)  # Learnable content
            self.query_pos = nn.Embedding(num_queries, embed_dim)  # Learnable positional embeddings
            self.reference_points = nn.Linear(embed_dim, 2)  # Mapping from positional embeddings to reference points

        self.layers = nn.ModuleList([instantiate(kwargs["layer"]) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        self._initialize_weights()

    def forward(self, features: Features, queries: Optional[Queries] = None) -> Tensor:
        """
        Forward pass for the transformer decoder.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
            queries: Initial object query proposals with shape (batch_size, num_queries, embed_dim).

        Returns:
            query_embed: Query embeddings with shape (batch_size, num_layers, num_queries, embed_dim).
            #### query_ref
            Query reference boxes with shape (batch_size, num_layers, num_queries, 4).
        """

        # In single-stage, we initialize the object queries as image-agnostic learned parameters
        if not self.two_stage:
            queries = self._initialize_object_queries(features)

        # Iteratively decode the object queries
        query_embed, query_ref = [], []
        for i, layer in enumerate(self.layers):
            queries: Queries = layer(queries, features)

            # Because we use Pre-LN, the output of each decoder layer still needs to be normalized
            if self.return_intermediates or i == len(self.layers) - 1:
                query_embed.append(self.norm(queries.embed))
                query_ref.append(queries.reference)

        # (batch_size, num_layers, num_queries, embed_dim)
        query_embed = torch.stack(query_embed, dim=1)
        query_ref = torch.stack(query_ref, dim=1)

        return query_embed, query_ref

    def _initialize_object_queries(self, features: Features) -> Queries:
        """
        Initializes the object queries for the transformer decoder.

        Args:
            Features: Multi-level features, with shape (batch_size, num_features, embed_dim).

        Returns:
            queries: Object queries with the following fields
        """

        # Get batch information
        batch_size, _, _ = features.embed.shape

        # Learned object queries and positional embeddings
        query_embed = self.queries.weight.unsqueeze(0)
        query_pos = self.query_pos.weight.unsqueeze(0)

        # The reference boxes use a learned mapping from the positional embeddings
        # as the center point and a fixed width and height
        xy = torch.sigmoid(self.reference_points(query_pos))
        wh = torch.full_like(xy, 0.1)
        query_ref = torch.cat([xy, wh], dim=-1)

        # Expand the queries across the batch size
        query_embed = query_embed.expand(batch_size, -1, -1)
        query_pos = query_pos.expand(batch_size, -1, -1)
        query_ref = query_ref.expand(batch_size, -1, -1)

        return Queries(embed=query_embed, pos=query_pos, reference=query_ref)

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """Initialize the transformer decoder weights."""

        if not self.two_stage:
            nn.init.xavier_uniform_(self.reference_points.weight, gain=1.0)
            nn.init.zeros_(self.reference_points.bias)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DecoderLayer(nn.Module):
    """
    Single layer of the transformer decoder.

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

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward Network
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim, embed_dim, 2, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, queries: Queries, features: Features) -> Queries:
        """
        Forward pass for a single transformer decoder layer.

        Args:
            queries: Object queries with shape (batch_size, num_queries, embed_dim).
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            queries: Object queries with the same shape as the input.
        """

        assert queries.embed.ndim == 3, f"Expected queries of shape (batch_size, num_queries, embed_dim), got {queries.embed.shape=}"
        assert features.embed.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.embed.shape=}"

        # Self-attention
        v = self.norm1(queries.embed)
        q = k = v + queries.pos
        queries.embed = queries.embed + self.dropout1(self.self_attention(q, k, v, need_weights=False)[0])

        # Cross-attention
        q = self.norm2(queries.embed) + queries.pos
        k = features.embed + features.pos
        v = features.embed
        queries.embed = queries.embed + self.dropout2(self.cross_attention(q, k, v, need_weights=False)[0])

        # Feedforward Network
        queries.embed = queries.embed + self.dropout3(self.ffn(self.norm3(queries.embed)))

        return queries

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DeformableDecoderLayer(nn.Module):
    """
    Single layer of the transformer decoder.

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

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Deformable cross-attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = MultiHeadDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward Network
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim, embed_dim, 2, dropout=dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, queries: Queries, features: Features) -> Queries:
        """
        Forward pass for a single transformer decoder layer.

        Args:
            queries: Object queries with shape (batch_size, num_queries, embed_dim).
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            queries: Object queries with the same shape as the input.
        """

        assert queries.embed.ndim == 3, f"Expected queries of shape (batch_size, num_queries, embed_dim), got {queries.embed.shape=}"
        assert features.embed.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.embed.shape=}"

        # Self-attention
        v = self.norm1(queries.embed)
        q = k = v + queries.pos
        queries.embed = queries.embed + self.dropout1(self.self_attention(q, k, v, need_weights=False)[0])

        # Cross-attention
        queries.embed = queries.embed + self.dropout2(
            self.cross_attention(
                queries=self.norm2(queries.embed) + queries.pos,
                query_reference=queries.reference,
                features=features.embed,
                dimensions=features.dimensions,
            )
        )

        # Feedforward Network
        queries.embed = queries.embed + self.dropout3(self.ffn(self.norm3(queries.embed)))

        return queries

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
