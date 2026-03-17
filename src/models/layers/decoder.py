from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import Dropout, LayerNorm, Module

from models.backbone import Features
from models.deformable_attention import MultiHeadDeformableAttention
from models.layers.ffn import FFN
from utils.misc import take_annotation_from

if TYPE_CHECKING:
    from models.decoder import Queries


class DecoderLayer(Module):
    """
    Single layer of the transformer decoder.

    Implements standard self-attention within queries, cross-attention between queries
    and features, and a two-layer feedforward network.

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

        self.num_heads = num_heads

        # Self-attention
        self.norm1 = LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)

        # Cross-attention
        self.norm2 = LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout2 = Dropout(dropout)

        # Feedforward Network
        self.norm3 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim, embed_dim, 2, dropout=dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, queries: Queries, features: Features) -> Queries:
        """
        Forward pass for a single transformer decoder layer.

        Args:
            queries: Object queries with shape (batch_size, num_queries, embed_dim).
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            queries: Object queries with the same shape as the input.
        """

        # Prepare shapes and split queries
        batch_size, _, _ = queries.embed.shape
        num_object_queries = queries.num_groups * queries.num_queries

        # Split object & denoising queries if any exist
        if queries.num_denoise_queries > 0:
            obj_embed, denoise_embed = queries.embed.split([num_object_queries, queries.num_denoise_queries], dim=1)
            obj_pos, denoise_pos = queries.pos.split([num_object_queries, queries.num_denoise_queries], dim=1)
        else:
            obj_embed = queries.embed
            obj_pos = queries.pos

        # Object query self-attention, moving groups into the batch dimension to prevent
        # attention between groups without requring a costly attention mask
        obj_embed = obj_embed.reshape(batch_size * queries.num_groups, queries.num_queries, -1)
        obj_pos = obj_pos.reshape(batch_size * queries.num_groups, queries.num_queries, -1)

        v = self.norm1(obj_embed)
        q = k = v + obj_pos
        obj_embed: Tensor = obj_embed + self.dropout1(self.self_attention(q, k, v, need_weights=False)[0])

        queries.embed = obj_embed.reshape(batch_size, num_object_queries, -1)

        # Denoising query self-attention, we need to use attention masks here because the
        # size and number of denoising query groups vary from image to image
        if queries.num_denoise_queries > 0:
            v = self.norm1(denoise_embed)
            q = k = v + denoise_pos

            denoise_pad = queries.denoise_padding_mask
            denoise_attn = queries.denoise_attention_mask.repeat_interleave(self.num_heads, dim=0)

            denoise_embed = denoise_embed + self.dropout1(
                self.self_attention(
                    query=q,
                    key=k,
                    value=v,
                    key_padding_mask=denoise_pad,
                    attn_mask=denoise_attn,
                    need_weights=False,
                )[0]
            )

            queries.embed = torch.cat([queries.embed, denoise_embed], dim=1)

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
        return Module.__call__(self, *args, **kwargs)


class DeformableDecoderLayer(Module):
    """
    Single deformable layer of the transformer decoder.

    Implements standard self-attention within queries, deformable cross-attention
    between queries and features, and a two-layer feedforward network.

    Args:
        embed_dim: Embedding dimension.
        ffn_dim: Feedforward network dimension.
        num_heads: Number of attention heads.
        num_deformable_heads: Number of deformable attention heads.
        num_points: Number of sampling points.
        num_levels: Number of feature levels.
        dropout: Dropout rate, optional.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_deformable_heads: int,
        num_points: int,
        num_levels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        # Self-attention
        self.norm1 = LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)

        # Deformable cross-attention
        self.norm2 = LayerNorm(embed_dim)
        self.cross_attention = MultiHeadDeformableAttention(embed_dim, num_deformable_heads, num_levels, num_points)
        self.dropout2 = Dropout(dropout)

        # Feedforward Network
        self.norm3 = LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_dim, embed_dim, 2, dropout=dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, queries: Queries, features: Features) -> Queries:
        """
        Forward pass for a single transformer decoder layer.

        Args:
            queries: Object queries with shape (batch_size, num_queries, embed_dim).
            features: Multi-level features with shape (batch_size, num_features, embed_dim).

        Returns:
            queries: Object queries with the same shape as the input.
        """

        # Prepare shapes and split queries
        batch_size, _, _ = queries.embed.shape
        num_object_queries = queries.num_groups * queries.num_queries

        # Split object & denoising queries if any exist
        if queries.num_denoise_queries > 0:
            obj_embed, denoise_embed = queries.embed.split([num_object_queries, queries.num_denoise_queries], dim=1)
            obj_pos, denoise_pos = queries.pos.split([num_object_queries, queries.num_denoise_queries], dim=1)
        else:
            obj_embed = queries.embed
            obj_pos = queries.pos

        # Object query self-attention, moving groups into the batch dimension to prevent
        # attention between groups without requring a costly attention mask
        obj_embed = obj_embed.reshape(batch_size * queries.num_groups, queries.num_queries, -1)
        obj_pos = obj_pos.reshape(batch_size * queries.num_groups, queries.num_queries, -1)

        v = self.norm1(obj_embed)
        q = k = v + obj_pos
        obj_embed: Tensor = obj_embed + self.dropout1(self.self_attention(q, k, v, need_weights=False)[0])

        queries.embed = obj_embed.reshape(batch_size, num_object_queries, -1)

        # Denoising query self-attention, we need to use attention masks here because the
        # size and number of denoising query groups vary from image to image
        if queries.num_denoise_queries > 0:
            v = self.norm1(denoise_embed)
            q = k = v + denoise_pos

            denoise_pad = queries.denoise_padding_mask
            denoise_attn = queries.denoise_attention_mask.repeat_interleave(self.num_heads, dim=0)

            denoise_embed = denoise_embed + self.dropout1(
                self.self_attention(
                    query=q,
                    key=k,
                    value=v,
                    key_padding_mask=denoise_pad,
                    attn_mask=denoise_attn,
                    need_weights=False,
                )[0]
            )

            queries.embed = torch.cat([queries.embed, denoise_embed], dim=1)

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
        return Module.__call__(self, *args, **kwargs)
