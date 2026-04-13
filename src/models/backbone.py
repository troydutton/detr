import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from models.dinov2 import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
from models.layers.positional_embedding import build_pos_embed
from models.projector import Projector
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
        name: Name of the backbone (e.g. dinov2_vits14).
        embed_dim: Dimension of the output embeddings.
        out_feature_indices: Indices of the stages from which features should be extracted.
        pretrained: Whether to load pretrained weights, optional.
        **kwargs: Arguments to construct the projector. See `models.projector.Projector`.
    """

    def __init__(
        self,
        name: str,
        embed_dim: int,
        out_feature_indices: List[int] = [-1],
        *,
        enable_projector: bool = False,
        enable_level_pos: bool = False,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.out_feature_indices = out_feature_indices
        self.enable_projector = enable_projector

        # Configure DINOv2
        config = Dinov2WithRegistersConfig.from_pretrained(name)
        config.image_size = 512
        config.patch_size = 16
        self.backbone = Dinov2WithRegistersModel(config, out_feature_indices=out_feature_indices)

        feature_channels = [config.hidden_size for _ in out_feature_indices]
        feature_strides = [config.patch_size for _ in out_feature_indices]

        # Create projections from each backbone output to the desired embedding dimension
        self.projections = nn.ModuleList()
        for in_channels in feature_channels:
            self.projections.append(nn.Conv2d(in_channels, embed_dim, kernel_size=1))
        self.projections.to(memory_format=torch.channels_last)

        # Initialize the multi-scale projector if requested
        if enable_projector:
            self.projector: Projector = Projector(in_strides=feature_strides, **kwargs["projector"]).to(memory_format=torch.channels_last)

        # Per-level positional embeddings
        self.num_output_levels = self.projector.num_output_levels if enable_projector else len(out_feature_indices)

        self.level_pos = nn.Embedding(self.num_output_levels, embed_dim) if enable_level_pos else None

        self._initialize_weights()

    def forward(self, images: Tensor) -> Features:
        """
        Args:
            images: Image with shape (batch_size, 3, height, width).

        Returns:
            features: Multi-level features
        """

        # Extract backbone features
        backbone_features = self.backbone(images)

        # Project all features to the desired embedding dimension
        features: List[Tensor] = [projection(feature) for projection, feature in zip(self.projections, backbone_features)]

        if self.enable_projector:
            features = self.projector(features)

        return self._build_features(features)

    def _build_features(self, features: List[Tensor]) -> Features:
        # Build multi-level features
        all_features, all_pos, all_references, all_levels, dimensions = [], [], [], [], []

        for level, feature in enumerate(features):
            # Get feature information
            batch_size, _, height, width = feature.shape
            device = feature.device

            # Flatten the spatial dimension
            feature = feature.flatten(2).transpose(1, 2)

            # Build reference points (center of each pixel normalized to [0, 1])
            x = torch.linspace(0.5, width - 0.5, width, device=device) / width
            y = torch.linspace(0.5, height - 0.5, height, device=device) / height
            feature_reference = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)
            feature_reference = feature_reference.reshape(1, width * height, 2)

            # Build positional embeddings for this level
            feature_pos = build_pos_embed(feature_reference, self.embed_dim)

            if self.level_pos is not None:
                level_pos: Tensor = self.level_pos(torch.tensor(level, device=device))
                feature_pos = feature_pos + level_pos.view(1, 1, self.embed_dim)

            # Repeat positional information across the batch (batch_size, height * width, -1)
            feature_reference = feature_reference.repeat(batch_size, 1, 1)
            feature_pos = feature_pos.repeat(batch_size, 1, 1)

            # Keep track of the level each feature came from
            levels = torch.full((width * height,), level, device=device)

            all_features.append(feature)
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
