import logging
from dataclasses import dataclass
from typing import List, Protocol

import timm
import torch
import torch.nn as nn
from timm.models import FeatureInfo
from torch import Tensor

from models.positional_embedding import build_pos_embed
from utils.misc import take_annotation_from

logger = logging.getLogger("detr")


@dataclass
class Features:
    embed: Tensor
    pos: Tensor
    reference: Tensor
    levels: Tensor
    dimensions: Tensor


class Backbone(nn.Module):
    """
    Args:
        name: Name of the backbone to load from timm.
        embed_dim: Dimension of the output embeddings.
        pretrained: Whether to load pretrained weights, optional.
    """

    def __init__(self, name: str, embed_dim: int, num_levels: int = 1, *, pretrained: bool = True) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_levels = num_levels

        # Take the last `num_levels` feature maps from the backbone
        out_indices = list(range(-num_levels, 0))

        # Build the backbone in feature-only mode
        self.backbone: BackboneType = timm.create_model(
            name,
            features_only=True,
            out_indices=out_indices,
            pretrained=pretrained,
        )

        # Create projections from each backbone output to the desired embedding dimension
        self.projections = nn.ModuleList()
        for i in out_indices:
            in_channels = self.backbone.feature_info[i]["num_chs"]
            self.projections.append(nn.Conv2d(in_channels, embed_dim, kernel_size=1))

        # Per-level positional embeddings
        self.level_pos = nn.Embedding(num_levels, embed_dim)

        self._initialize_weights()

        # Optimize memory format to align with DDP bucket views
        self.backbone.to(memory_format=torch.channels_last)
        self.projections.to(memory_format=torch.channels_last)

    def forward(self, images: Tensor) -> Features:
        """
        Args:
            images: Image with shape (batch_size, 3, height, width).

        Returns:
            features: Multi-level features with the following fields
            - `features`: Embeddings with shape (batch_size, num_features, embed_dim).
            - `pos`: Positional embeddings with shape (batch_size, num_features, embed_dim).
            - `reference`: Reference points for the features with shape (batch_size, num_features, 2).
            - `levels`: Level index for each feature with shape (num_features,).
            - `dimensions`: Width and height of each feature level with shape (num_levels, 2).
        """

        # Get batch information
        batch_size, _, _, _ = images.shape
        device = images.device

        # Extract features
        backbone_features = self.backbone(images)  # List of features with shape (batch_size, channels, feature_height, feature_width)

        # Build multi-level features, positional embeddings, and reference points for the transformer
        all_features, all_pos, all_references, all_levels, dimensions = [], [], [], [], []

        for level, (features, projection, level_pos) in enumerate(zip(backbone_features, self.projections, self.level_pos.weight)):
            _, _, height, width = features.shape

            # Project into the desired embedding dimension
            features: Tensor = projection(features)
            features = features.flatten(2).transpose(1, 2)

            # Build reference points (center of each pixel normalized to [0, 1])
            x = torch.linspace(0.5, width - 0.5, width, device=device) / width
            y = torch.linspace(0.5, height - 0.5, height, device=device) / height
            feature_reference = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)
            feature_reference = feature_reference.reshape(1, width * height, 2)

            # Build positional embeddings for this level, adding the learnable level embedding
            feature_pos = build_pos_embed(feature_reference, self.embed_dim) + level_pos.view(1, 1, self.embed_dim)

            # Repeat positional information across the batch (batch_size, height * width, -1)
            feature_reference = feature_reference.repeat(batch_size, 1, 1)
            feature_pos = feature_pos.repeat(batch_size, 1, 1)

            # Keep track of the level each feature came from
            levels = torch.full((width * height,), level, device=device)

            all_features.append(features)
            all_pos.append(feature_pos)
            all_references.append(feature_reference)
            all_levels.append(levels)
            dimensions.append((width, height))

        # Concatenate features from all levels
        all_features = torch.cat(all_features, dim=1)
        all_pos = torch.cat(all_pos, dim=1)
        all_references = torch.cat(all_references, dim=1)
        all_levels = torch.cat(all_levels, dim=0)
        dimensions = torch.tensor(dimensions, device=device)

        return Features(
            embed=all_features,
            pos=all_pos,
            reference=all_references,
            levels=all_levels,
            dimensions=dimensions,
        )

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """Initialize the backbone weights."""

        # Initialize projection weights to maintain variance
        for projection in self.projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class BackboneType(Protocol):
    feature_info: FeatureInfo

    def to(self, *args, **kwargs) -> "BackboneType": ...
    def __call__(self, Images: Tensor) -> List[Tensor]: ...
