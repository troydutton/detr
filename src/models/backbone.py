import logging
from dataclasses import dataclass
from typing import List, Protocol

import timm
import torch
import torch.nn as nn
from timm.models import FeatureInfo
from torch import Tensor

from models.positional_embedding import build_positional_embedding
from utils.misc import take_annotation_from

logger = logging.getLogger("detr")


@dataclass
class Features:
    embed: Tensor
    pos: Tensor
    reference: Tensor
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
            - `dimensions`: Height and width of each feature level with shape (num_levels, 2).
        """

        # Get batch information
        batch_size, _, _, _ = images.shape
        device = images.device

        # Extract features
        backbone_features = self.backbone(images)  # List of features with shape (batch_size, channels, feature_height, feature_width)

        # Build multi-level features, positional embeddings, and reference points for the transformer
        multi_level_features, multi_level_pos, multi_level_reference, dimensions = [], [], [], []

        for features, projection, level_pos in zip(backbone_features, self.projections, self.level_pos.weight):
            _, _, height, width = features.shape

            # Project into the desired embedding dimension
            features: Tensor = projection(features)

            # Build positional embeddings for this level, adding the learnable level embedding
            feature_pos = build_positional_embedding(features) + level_pos.view(1, self.embed_dim, 1, 1)

            # Build reference points (center of each pixel normalized to [0, 1])
            x = torch.linspace(0.5, width - 0.5, width, device=device) / width
            y = torch.linspace(0.5, height - 0.5, height, device=device) / height
            feature_reference = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)
            feature_reference = feature_reference.repeat(batch_size, 1, 1, 1)  # (batch_size, height, width, 2)

            # Collapse the spatial dimension
            features = features.flatten(2).transpose(1, 2)
            feature_pos = feature_pos.flatten(2).transpose(1, 2)
            feature_reference = feature_reference.flatten(1, 2)

            multi_level_features.append(features)
            multi_level_pos.append(feature_pos)
            multi_level_reference.append(feature_reference)
            dimensions.append((height, width))

        # Concatenate features from all levels
        multi_level_features = torch.cat(multi_level_features, dim=1)
        multi_level_pos = torch.cat(multi_level_pos, dim=1)
        multi_level_reference = torch.cat(multi_level_reference, dim=1)
        dimensions = torch.tensor(dimensions, device=device)

        # Build the
        multi_level_features = Features(
            embed=multi_level_features,
            pos=multi_level_pos,
            reference=multi_level_reference,
            dimensions=dimensions,
        )

        return multi_level_features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class BackboneType(Protocol):
    feature_info: FeatureInfo

    def to(self, *args, **kwargs) -> "BackboneType": ...
    def __call__(self, Images: Tensor) -> List[Tensor]: ...
