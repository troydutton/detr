import logging
from typing import List, Protocol, Tuple

import timm
import torch
import torch.nn as nn
from timm.models import FeatureInfo
from torch import Tensor

from models.positional_embedding import build_positional_embedding

logger = logging.getLogger("detr")


class Backbone(nn.Module):
    """
    Args:
        name: Name of the backbone to load from timm.
        embed_dim: Dimension of the output embeddings.
        pretrained: Whether to load pretrained weights, optional.
    """

    def __init__(self, name: str, embed_dim: int, *, pretrained: bool = True) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        try:
            self.backbone: BackboneType = timm.create_model(
                name,
                features_only=True,
                out_indices=[-1],
                pretrained=pretrained,
            )
        except Exception as e:
            raise ValueError(f"Error creating backbone '{name}': {e}")

        in_channels = self.backbone.feature_info[-1]["num_chs"]

        self.projection: ProjectionType = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        # Optimize memory format to align with DDP bucket views
        self.backbone.to(memory_format=torch.channels_last)
        self.projection.to(memory_format=torch.channels_last)

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            images: Image with shape (batch_size, 3, height, width).

        Returns:
            features: Features with shape (batch_size, embed_dim, feature_height, feature_width).
            #### feature_pos
            Positional embeddings with shape (batch_size, embed_dim, feature_height, feature_width).
        """

        # Extract features
        features = self.backbone(images)[-1]  # (batch_size, in_channels, feature_height, feature_width)

        # Project to the desired embedding dimension
        features = self.projection(features)  # (batch_size, embed_dim, feature_height, feature_width)

        # Build positional embeddings
        feature_pos = build_positional_embedding(features)

        return features, feature_pos


class BackboneType(Protocol):
    feature_info: FeatureInfo

    def to(self, *args, **kwargs) -> "BackboneType": ...
    def __call__(self, Images: Tensor) -> List[Tensor]: ...


class ProjectionType(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...
