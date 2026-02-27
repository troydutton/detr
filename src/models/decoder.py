import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from hydra.utils import instantiate
from torch import Tensor, nn
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList, Sequential
from torchvision.ops.boxes import box_convert

from data.coco_dataset import Target
from models.backbone import Features
from models.deformable_attention import MultiHeadDeformableAttention
from models.layers import FFN
from models.positional_embedding import build_pos_embed
from utils.misc import take_annotation_from


@dataclass
class Queries:
    embed: Tensor
    pos: Tensor
    reference: Tensor
    padding_mask: Tensor
    attention_mask: Tensor


class TransformerDecoder(Module):
    """
    Transformer decoder composed of a stack of N decoder layers.

    Args:
        embed_dim: Embedding dimension.
        num_classes: Number of classes.
        num_layers: Number of decoder layers.
        num_queries: Number of object queries.
        num_groups: Number of query groups, optional.
        two_stage: Initialize object queries using encoder proposals or learned parameters, optional.
        refine_boxes: Whether to iteratively refine the reference boxes across decoder layers, optional.
        kwargs: Arguments to construct the decoder layers.
            See `models.decoder.DecoderLayer` or `models.decoder.DeformableDecoderLayer`.

    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_layers: int,
        num_queries: int,
        num_groups: int = 1,
        pos_noise_scale: float = 0.4,
        neg_noise_scale: float = 0.8,
        label_noise_prob: float = 0.5,
        *,
        two_stage: bool = False,
        refine_boxes: bool = False,
        denoise_queries: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_groups = num_groups
        self.pos_noise_scale = pos_noise_scale
        self.neg_noise_scale = neg_noise_scale
        self.label_noise_prob = label_noise_prob
        self.two_stage = two_stage
        self.refine_boxes = refine_boxes
        self.denoise_queries = denoise_queries

        # Create the bounding box and classification heads
        self.bbox_head = FFN(embed_dim, embed_dim, 4, 3)

        self.class_head = Linear(embed_dim, num_classes)

        # Create the object query initialization components
        if self.two_stage:  # Encoder proposals as initial queries
            self.encoder_bbox_head = ModuleList([FFN(embed_dim, embed_dim, 4, 3) for _ in range(self.num_groups)])
            self.encoder_class_head = ModuleList([Linear(embed_dim, num_classes) for _ in range(self.num_groups)])
            self.encoder_projection = ModuleList([Sequential(Linear(embed_dim, embed_dim), LayerNorm(embed_dim)) for _ in range(self.num_groups)])  # fmt: skip
        else:  # Learnable parameters as initial queries
            self.queries = nn.Embedding(num_groups * num_queries, embed_dim)
            self.reference_points = Linear(embed_dim, 2)
        if self.denoise_queries:  # Denoising queries
            self.label_embed = nn.Embedding(num_classes, embed_dim)
            self.object_embed = nn.Embedding(1, embed_dim)
            self.denoise_embed = nn.Embedding(1, embed_dim)

        self.pos_projection = FFN(2 * embed_dim, embed_dim, embed_dim, 2)

        # Create the decoder layers
        self.layers = ModuleList([instantiate(kwargs["layer"]) for _ in range(num_layers)])
        self.norm = LayerNorm(embed_dim)

        self._initialize_weights()

    def forward(
        self,
        features: Features,
        targets: List[Target] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass for the transformer decoder.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
            targets: List of targets for each image, optional.

        Returns:
            boxes: Decoder box predictions.
            #### logits
            Decoder class logits.
            #### encoder_boxes
            Encoder proposal box predictions.
            #### encoder_logits
            Encoder proposal class logits.
            #### denoise_boxes
            Denoising query box predictions.
            #### denoise_logits
            Denoising query class logits.
        """

        assert features.embed.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.embed.shape=}"

        # Get batch information
        batch_size, _, _ = features.embed.shape

        # Initialize the object queries, during inference we only user the first query group
        num_groups = self.num_groups if self.training else 1
        encoder_boxes = encoder_logits = None
        if self.two_stage:
            queries, encoder_boxes, encoder_logits = self._generate_query_proposals(features, num_groups)
        else:
            queries = self._initialize_object_queries(batch_size, num_groups)

        # Add a learnable task embedding to distinguish between object and denoising queries
        if self.denoise_queries:
            queries.embed += self.object_embed.weight[None, ...]

        if self.training and self.denoise_queries:
            queries = self._generate_denoising_queries(queries, targets)

        # Iteratively decode the object queries
        boxes, logits = [], []
        for layer in self.layers:
            queries: Queries = layer(queries, features)

            # Predict bounding boxes (x + (Δx * w), y + (Δy * h), w * exp(Δw), h * exp(Δh))
            offsets = self.bbox_head(query_embed := self.norm(queries.embed))
            xy = queries.reference[..., :2] + (offsets[..., :2] * queries.reference[..., 2:])
            wh = queries.reference[..., 2:] * offsets[..., 2:].exp()
            layer_boxes = torch.cat([xy, wh], dim=-1)

            # Refine the reference boxes and positional embeddings for the next layer
            if self.refine_boxes:
                queries.reference = layer_boxes.detach()
                queries.pos = self.pos_projection(build_pos_embed(queries.reference, 2 * self.embed_dim))

            # Predict class logits
            layer_logits = self.class_head(query_embed)

            boxes.append(layer_boxes)
            logits.append(layer_logits)

        # Stack the outputs from each layer into a single tensor
        boxes = torch.stack(boxes, dim=1)
        logits = torch.stack(logits, dim=1)

        # Replace predictions from padded queries with zeros
        boxes = boxes.masked_fill(queries.padding_mask[:, None, :, None], 0.0)
        logits = logits.masked_fill(queries.padding_mask[:, None, :, None], 0.0)

        # Separate the denoising query predictions if they exist
        _, total_queries, _ = queries.embed.shape
        num_object_queries = num_groups * self.num_queries
        num_denoise_queries = total_queries - num_object_queries

        denoise_boxes = denoise_logits = None
        if num_denoise_queries > 0:
            boxes, denoise_boxes = boxes.split([num_object_queries, num_denoise_queries], dim=2)
            logits, denoise_logits = logits.split([num_object_queries, num_denoise_queries], dim=2)

            denoise_boxes = denoise_boxes.reshape(batch_size, self.num_layers, 1, num_denoise_queries, -1)
            denoise_logits = denoise_logits.reshape(batch_size, self.num_layers, 1, num_denoise_queries, -1)

        boxes = boxes.reshape(batch_size, self.num_layers, num_groups, self.num_queries, -1)
        logits = logits.reshape(batch_size, self.num_layers, num_groups, self.num_queries, -1)

        return boxes, logits, encoder_boxes, encoder_logits, denoise_boxes, denoise_logits

    def _initialize_object_queries(self, batch_size: int, num_groups: int) -> Queries:
        """
        Initializes the object queries for the transformer decoder.

        Args:
            batch_size: Batch size to expand the queries to.
            num_groups: Number of query groups to use.

        Returns:
            queries: Initial object queries.
        """

        # Learned object query embeddings (1, num_groups * num_queries, embed_dim)
        query_embed = self.queries.weight[: num_groups * self.num_queries].unsqueeze(0)

        # Generate reference boxes from the query embeddings
        feature_xy = torch.sigmoid(self.reference_points(query_embed))
        feature_wh = torch.full_like(feature_xy, 0.1)
        query_ref = torch.cat([feature_xy, feature_wh], dim=-1)

        # Generate positional embeddings from reference boxes
        query_pos: Tensor = self.pos_projection(build_pos_embed(query_ref, 2 * self.embed_dim))

        # Expand the queries across the batch size
        query_embed = query_embed.expand(batch_size, -1, -1)
        query_pos = query_pos.expand(batch_size, -1, -1)
        query_ref = query_ref.expand(batch_size, -1, -1)

        # Create masks to force queries to attend within their group
        padding_mask = torch.zeros(batch_size, num_groups * self.num_queries, dtype=torch.bool, device=query_embed.device)
        group_indices = torch.arange(num_groups * self.num_queries, device=query_embed.device) // self.num_queries
        attention_mask = (group_indices[:, None] != group_indices[None, :]).unsqueeze(0).repeat(batch_size, 1, 1)

        return Queries(embed=query_embed, pos=query_pos, reference=query_ref, padding_mask=padding_mask, attention_mask=attention_mask)

    def _generate_query_proposals(self, features: Features, num_groups: int) -> Tuple[Queries, Tensor, Tensor]:
        """
        Generates initial object queries from the encoder features.

        Args:
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
            num_groups: Number of query groups to use.

        Returns:
            queries: Initial object queries.
            #### encoder_boxes
            Normalized CXCYWH encoder proposal box predictions with shape (batch_size, 1, num_features, 4).
            #### encoder_logits
            Encoder proposal class logits with shape (batch_size, 1, num_features, num_classes).
        """

        # Get batch information
        batch_size, num_features, _ = features.embed.shape
        device = features.embed.device

        query_embed, query_ref = [], []
        encoder_boxes, encoder_logits = [], []

        # TODO: Optimize proposal generation by batching group calculations
        for g in range(num_groups):
            # Project the encoder features into a separate latent space
            feature_embed = self.encoder_projection[g](features.embed)

            # Predict proposal boxes, using the feature pixel's position as the center
            # and a level-specific scale as the width and height (scale * 2^level)
            # Boxes are refined using (x + (Δx * w), y + (Δy * h), w * exp(Δw), h * exp(Δh))
            xy = features.reference
            wh = torch.full_like(features.reference, 0.05) * (2**features.levels).unsqueeze(-1)
            offsets = self.encoder_bbox_head[g](feature_embed)
            xy = xy + (offsets[..., :2] * wh)
            wh = wh * torch.exp(offsets[..., 2:])
            boxes = torch.cat([xy, wh], dim=-1)

            # Identify the features with the highest proposal scores (max class probability)
            logits: Tensor = self.encoder_class_head[g](feature_embed)
            scores = logits.max(dim=-1).values
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
            topk_indices = scores.topk(self.num_queries, dim=1).indices

            query_embed.append(feature_embed[batch_indices, topk_indices])
            query_ref.append(boxes[batch_indices, topk_indices])
            encoder_boxes.append(boxes)
            encoder_logits.append(logits)

        query_embed = torch.cat(query_embed, dim=1)
        query_ref = torch.cat(query_ref, dim=1)
        encoder_boxes = torch.cat(encoder_boxes, dim=1)
        encoder_logits = torch.cat(encoder_logits, dim=1)

        # Generate positional embeddings from the reference boxes
        query_pos = self.pos_projection(build_pos_embed(query_ref, 2 * self.embed_dim))

        # Treat the references as fixed priors for the decoder
        query_ref = query_ref.detach()

        # Create masks to force queries to attend within their group
        padding_mask = torch.zeros(batch_size, num_groups * self.num_queries, dtype=torch.bool, device=query_embed.device)
        group_indices = torch.arange(num_groups * self.num_queries, device=query_embed.device) // self.num_queries
        attention_mask = (group_indices[:, None] != group_indices[None, :]).unsqueeze(0).repeat(batch_size, 1, 1)

        # Prepare outputs
        queries = Queries(embed=query_embed, pos=query_pos, reference=query_ref, padding_mask=padding_mask, attention_mask=attention_mask)
        encoder_boxes = encoder_boxes.view(batch_size, 1, num_groups, num_features, -1)
        encoder_logits = encoder_logits.view(batch_size, 1, num_groups, num_features, -1)

        return queries, encoder_boxes, encoder_logits

    def _generate_denoising_queries(self, queries: Queries, targets: List[Target]) -> Queries:
        """
        Generates denoising queries for the transformer decoder during training.

        Args:
            queries: Initial object queries.
            targets: List of targets for each image.

        Returns:
            queries: Object and denoising queries.
        """

        # Get batch information
        device = queries.embed.device
        batch_size, num_object_queries, _ = queries.embed.shape
        max_objects = max(len(target["labels"]) for target in targets)

        # Skip denoising if there are no objects in the batch
        if max_objects == 0:
            return queries

        # We dynamically adjust the number of denoising query groups to
        # ensure a constant number of denoising queries across batches
        num_denoise_groups = self.num_queries // (2 * max_objects)
        max_queries = num_denoise_groups * (2 * max_objects)

        # Generate noisy versions of the target boxes and labels
        query_embed = torch.zeros(batch_size, max_queries, self.embed_dim, device=device)
        query_pos = torch.zeros(batch_size, max_queries, self.embed_dim, device=device)
        query_ref = torch.zeros(batch_size, max_queries, 4, device=device)
        padding_mask = torch.ones(batch_size, max_queries, dtype=torch.bool, device=device)
        attention_mask = torch.ones(batch_size, max_queries, max_queries, dtype=torch.bool, device=device)

        for i, target in enumerate(targets):
            boxes, labels = target["boxes"], target["labels"]
            num_objects = len(labels)

            # Skip processing if there are no objects in the image
            if len(labels) == 0:
                continue

            # Repeat the boxes and labels for each denoising group
            boxes = boxes.repeat(2 * num_denoise_groups, 1)
            labels = labels.repeat(2 * num_denoise_groups)
            num_queries = len(labels)

            # Convert to xyxy format for easier scaling
            boxes = boxes.reshape(num_denoise_groups, 2, num_objects, 4)
            box_scales = torch.cat([boxes[..., 2:] / 2, boxes[..., 2:] / 2], dim=-1)
            boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

            # Add uniform noise to the boxes, with different scales for positive and negative samples
            box_noise = torch.rand_like(boxes)  # [0, 1)
            box_noise[:, 0] *= self.pos_noise_scale  # [0, λ1)
            box_noise[:, 1] *= self.neg_noise_scale - self.pos_noise_scale  # [0, λ2 - λ1)
            box_noise[:, 1] += self.pos_noise_scale  # [λ1, λ2)
            box_noise *= torch.randint_like(boxes, 0, 2) * 2 - 1  # (-λ1, λ1) or (-λ2, -λ1] U [λ1, λ2)
            box_noise *= box_scales  # |∆x| < λw/2, |∆y| < λh/2, |∆w| < λw, |∆y| < λh
            boxes = (boxes + box_noise).clamp(0, 1)

            # The model expects boxes in cxcywh format
            boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
            boxes = boxes.reshape(num_queries, 4)

            # Randomly perturb labels to make the denoising task more challenging
            label_noise = torch.rand_like(labels, dtype=torch.float32) < self.label_noise_prob
            labels = torch.where(label_noise, torch.randint_like(labels, 0, self.num_classes), labels)

            # Embeddings come from labels, positional embeddings come from boxes, and references are the boxes themselves
            query_embed[i, :num_queries] = self.label_embed(labels)
            query_pos[i, :num_queries] = self.pos_projection(build_pos_embed(boxes, 2 * self.embed_dim))
            query_ref[i, :num_queries] = boxes

            # Mask out the padding queries at the end of the sequence
            padding_mask[i, :num_queries] = False

            # Only allow queries within the same group to attend to each other
            denoise_group_indices = torch.arange(num_queries, device=device) // (2 * num_objects)
            attention_mask[i, :num_queries, :num_queries] = denoise_group_indices[:, None] != denoise_group_indices[None, :]

        # Preserve the original object query masks
        full_padding_mask = torch.ones(batch_size, num_object_queries + max_queries, dtype=torch.bool, device=device)
        full_attention_mask = torch.ones(batch_size, num_object_queries + max_queries, num_object_queries + max_queries, dtype=torch.bool, device=device)  # fmt: skip
        full_padding_mask[..., :num_object_queries] = queries.padding_mask
        full_attention_mask[..., :num_object_queries, :num_object_queries] = queries.attention_mask
        full_padding_mask[..., num_object_queries:] = padding_mask
        full_attention_mask[..., num_object_queries:, num_object_queries:] = attention_mask

        # Add a learnable task embedding to distinguish between object and denoising queries
        query_embed += self.denoise_embed.weight[None, ...]

        # Append the denoising queries onto the object queries
        queries.embed = torch.cat([queries.embed, query_embed], dim=1)
        queries.pos = torch.cat([queries.pos, query_pos], dim=1)
        queries.reference = torch.cat([queries.reference, query_ref], dim=1)
        queries.padding_mask = full_padding_mask
        queries.attention_mask = full_attention_mask

        return queries

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """Initialize the transformer decoder weights."""

        # We bias the classifier to predict a small probability for objects
        # because the majority of queries are not matched to objects
        bias = -math.log((1 - (object_prob := 0.01)) / object_prob)
        nn.init.constant_(self.class_head.bias, bias)

        if self.two_stage:
            for g in range(self.num_groups):
                nn.init.constant_(self.encoder_class_head[g].bias, bias)
        else:
            nn.init.xavier_uniform_(self.reference_points.weight, gain=1.0)
            nn.init.zeros_(self.reference_points.bias)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return Module.__call__(self, *args, **kwargs)


class DecoderLayer(Module):
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
        num_groups: int = 1,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_groups = num_groups

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

        # During training, we use multiple object/denoising query groups which attend separately
        # in self-attention, but during inference we only use the first group of object queries
        padding_mask = queries.padding_mask if self.training else None
        attention_mask = queries.attention_mask.repeat_interleave(self.num_heads, dim=0) if self.training else None

        # Self-attention
        v = self.norm1(queries.embed)
        q = k = v + queries.pos
        queries.embed = queries.embed + self.dropout1(
            self.self_attention(
                query=q,
                key=k,
                value=v,
                key_padding_mask=padding_mask,
                attn_mask=attention_mask,
                need_weights=False,
            )[0]
        )

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
        num_groups: int = 1,
        num_queries: int = 300,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_groups = num_groups
        self.num_queries = num_queries

        # Self-attention
        self.norm1 = LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.dropout1 = Dropout(dropout)

        # Deformable cross-attention
        self.norm2 = LayerNorm(embed_dim)
        self.cross_attention = MultiHeadDeformableAttention(embed_dim, num_heads, num_levels, num_points)
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

        # During training, we use multiple object/denoising query groups which attend separately
        # in self-attention, but during inference we only use the first group of object queries
        padding_mask = queries.padding_mask if self.training else None
        attention_mask = queries.attention_mask.repeat_interleave(self.num_heads, dim=0) if self.training else None

        # Self-attention
        v = self.norm1(queries.embed)
        q = k = v + queries.pos
        queries.embed = queries.embed + self.dropout1(
            self.self_attention(
                query=q,
                key=k,
                value=v,
                key_padding_mask=padding_mask,
                attn_mask=attention_mask,
                need_weights=False,
            )[0]
        )

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
