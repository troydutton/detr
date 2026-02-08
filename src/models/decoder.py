from dataclasses import dataclass

import torch
from hydra.utils import instantiate
from torch import Tensor, nn

from models.backbone import Features
from models.layers import MultiLayerPerceptron
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
        return_intermediates: Whether to return intermediate transformer outputs, optional.
        kwargs: Arguments to construct the decoder layers.
            See `models.decoder.DecoderLayer` or `models.decoder.DeformableDecoderLayer`.

    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_queries: int,
        return_intermediates: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.return_intermediates = return_intermediates

        self.queries = nn.Embedding(num_queries, embed_dim)  # Learnable content for initializing object queries
        self.query_pos = nn.Embedding(num_queries, embed_dim)  # Learnable positional embeddings for object queries
        # self.reference_head = nn.Linear(embed_dim, 2)  # Mapping from positional embeddings to reference points

        self.layers = nn.ModuleList([instantiate(kwargs["layer"]) for _ in range(num_layers)])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: Features) -> Tensor:
        """
        Forward pass for the transformer decoder.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            query_embed: Query embeddings with shape (batch_size, num_layers, num_queries, embed_dim).
            query_ref: Query reference boxes with shape (batch_size, num_layers, num_queries, 4).
        """

        # Initialize the object queries
        queries = self._intialize_object_queries(features)

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

    def _intialize_object_queries(self, features: Features) -> Queries:
        """
        Initializes the object queries for the transformer decoder.

        Args:
            Features: Multi-level features, with shape (batch_size, num_features, embed_dim).

        Returns:
            queries: Object queries with the following fields
        """

        # Get batch information
        batch_size, num_features, _ = features.embed.shape

        # Initialize the object queries
        query_embed = self.queries.weight.unsqueeze(0)
        query_pos = self.query_pos.weight.unsqueeze(0)

        # Calculate initial reference boxes
        # xy = torch.sigmoid(self.reference_head(query_pos))
        # wh = torch.full_like(xy, 0.1)
        # query_ref = torch.cat([xy, wh], dim=-1)
        query_ref = torch.zeros(1, num_features, 4, device=features.embed.device)

        # Expand the queries across the batch size
        query_embed = query_embed.expand(batch_size, -1, -1)
        query_pos = query_pos.expand(batch_size, -1, -1)
        query_ref = query_ref.expand(batch_size, -1, -1)

        return Queries(embed=query_embed, pos=query_pos, reference=query_ref)

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
        self.ffn = MultiLayerPerceptron(
            input_dim=embed_dim,
            hidden_dim=ffn_dim,
            output_dim=embed_dim,
            num_layers=2,
            dropout=dropout,
        )
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
