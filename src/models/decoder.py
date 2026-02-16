import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from hydra.utils import instantiate
from torch import Tensor, nn

from models.backbone import Features
from models.deformable_attention import MultiHeadDeformableAttention
from models.layers import FFN
from models.positional_embedding import build_reference_positional_embeddings
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
        refine_boxes: Whether to iteratively refine the reference boxes across decoder layers, optional.
        num_classes: Number of object classes, required if `two_stage` is True.
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
        refine_boxes: bool = False,
        num_classes: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        assert not two_stage or num_classes is not None, "num_classes is required when two_stage is True"

        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.two_stage = two_stage
        self.refine_boxes = refine_boxes
        self.num_classes = num_classes

        # Create the bounding box and classification heads
        if self.refine_boxes:
            self.bbox_head = nn.ModuleList([FFN(embed_dim, embed_dim, 4, 3) for _ in range(num_layers)])
        else:
            self.bbox_head = FFN(embed_dim, embed_dim, 4, 3)

        self.class_head = nn.Linear(embed_dim, num_classes)

        if self.two_stage:  # Encoder proposals as initial queries
            self.encoder_bbox_head = FFN(embed_dim, embed_dim, output_dim=4, num_layers=3)
            self.encoder_class_head = nn.Linear(embed_dim, num_classes)
            self.encoder_projection = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
            self.positional_projection = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        else:  # Learnable parameters as initial queries
            self.queries = nn.Embedding(num_queries, embed_dim)
            self.query_pos = nn.Embedding(num_queries, embed_dim)
            self.reference_points = nn.Linear(embed_dim, 2)

        self.layers = nn.ModuleList([instantiate(kwargs["layer"]) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)

        self._initialize_weights()

    def forward(self, features: Features) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass for the transformer decoder.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            boxes: Normalized CXCYWH bounding box predictions with shape (batch_size, num_layers, num_queries, 4).
            #### logits
            Class logits with shape (batch_size, num_layers, num_queries, num_classes).
            #### encoder_boxes
            Normalized CXCYWH encoder proposal box predictions with shape (batch_size, 1, num_features, 4).
            #### encoder_logits
            Encoder proposal class logits with shape (batch_size, 1, num_features, num_classes).
        """

        assert features.embed.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.embed.shape=}"

        # Initialize the object queries
        encoder_boxes = encoder_logits = None
        if self.two_stage:
            queries, encoder_boxes, encoder_logits = self._generate_query_proposals(features)
        else:
            queries = self._initialize_object_queries(len(features.embed))

        # Iteratively decode the object queries
        boxes, logits = [], []
        for i, layer in enumerate(self.layers):
            queries: Queries = layer(queries, features)

            # Predict bounding boxes (x + (Δx * w), y + (Δy * h), w * exp(Δw), h * exp(Δh))
            bbox_head = self.bbox_head[i] if self.refine_boxes else self.bbox_head
            offsets: Tensor = bbox_head(query_embed := self.norm(queries.embed))
            xy = queries.reference[..., :2] + (offsets[..., :2] * queries.reference[..., 2:])
            wh = queries.reference[..., 2:] * offsets[..., 2:].exp()
            layer_boxes = torch.cat([xy, wh], dim=-1)

            # Refine the reference boxes for the next layer
            if self.refine_boxes:
                queries.reference = layer_boxes.detach()

            # Predict class logits
            layer_logits = self.class_head(query_embed)

            boxes.append(layer_boxes)
            logits.append(layer_logits)

        # Stack the outputs from each layer into a single tensor
        boxes = torch.stack(boxes, dim=1)
        logits = torch.stack(logits, dim=1)

        return boxes, logits, encoder_boxes, encoder_logits

    def _initialize_object_queries(self, batch_size: int) -> Queries:
        """
        Initializes the object queries for the transformer decoder.

        Args:
            batch_size: Batch size to expand the queries to.

        Returns:
            queries: Initial object queries.
        """

        # Learned object queries and positional embeddings
        query_embed = self.queries.weight.unsqueeze(0)
        query_pos = self.query_pos.weight.unsqueeze(0)

        # The reference boxes use a learned mapping from the positional embeddings
        # as the center point and a fixed width and height
        feature_xy = torch.sigmoid(self.reference_points(query_pos))
        feature_wh = torch.full_like(feature_xy, 0.1)
        query_ref = torch.cat([feature_xy, feature_wh], dim=-1)

        # Expand the queries across the batch size
        query_embed = query_embed.expand(batch_size, -1, -1)
        query_pos = query_pos.expand(batch_size, -1, -1)
        query_ref = query_ref.expand(batch_size, -1, -1)

        return Queries(embed=query_embed, pos=query_pos, reference=query_ref)

    def _generate_query_proposals(self, features: Features) -> Tuple[Queries, Tensor, Tensor]:
        # Get batch information
        batch_size, _, _ = features.embed.shape
        device = features.embed.device

        # Project the encoder features into a separate latent space
        feature_embed = self.encoder_projection(features.embed)

        # Predict proposal boxes, using the feature pixel's position as the center
        # and a level-specific scale as the width and height (scale * 2^level)
        # Boxes are refined using (x + (Δx * w), y + (Δy * h), w * exp(Δw), h * exp(Δh))
        xy = features.reference
        wh = torch.full_like(features.reference, 0.05) * (2**features.levels).unsqueeze(-1)
        offsets = self.encoder_bbox_head(feature_embed)
        xy = xy + (offsets[..., :2] * wh)
        wh = wh * offsets[..., 2:].exp()
        boxes = torch.cat([xy, wh], dim=-1)

        # Identify the features with the highest proposal scores (max class probability)
        logits: Tensor = self.encoder_class_head(feature_embed)
        scores = logits.max(dim=-1).values
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
        topk_indices = scores.topk(self.num_queries, dim=1).indices
        query_embed = feature_embed[batch_indices, topk_indices]
        query_ref = boxes[batch_indices, topk_indices]

        # The embeddings and positional embeddings are generated by projecting the
        # positional encodings of the reference points into a shared latent space
        ref_pos = build_reference_positional_embeddings(query_ref, 2 * self.embed_dim)
        query_pos = self.positional_projection(ref_pos)

        # Add a dummy layer dimension
        boxes = boxes.unsqueeze(1)
        logits = logits.unsqueeze(1)

        return Queries(embed=query_embed, pos=query_pos, reference=query_ref.detach()), boxes, logits

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """Initialize the transformer decoder weights."""

        # We bias the classifier to predict a small probability for objects
        # because the majority of queries are not matched to objects
        bias = -math.log((1 - (object_prob := 0.01)) / object_prob)
        nn.init.constant_(self.class_head.bias, bias)

        if self.two_stage:
            nn.init.constant_(self.encoder_class_head.bias, bias)
        else:
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
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
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
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # Deformable cross-attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = MultiHeadDeformableAttention(embed_dim, num_heads, num_levels, num_points)
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
