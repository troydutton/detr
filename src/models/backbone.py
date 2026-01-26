import logging
from typing import List, Protocol, Tuple

import timm
import torch.nn as nn
from timm.models import FeatureInfo
from torch import Tensor

from models.positional_embedding import generate_positional_embedding_sine

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

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            images: Image with shape (batch_size, 3, height, width).

        Returns:
            features: Features with shape (batch_size, feature_height, feature_width, embed_dim).
            #### feature_pos
            Positional embeddings with shape (batch_size, feature_height, feature_width, embed_dim).
        """

        # Extract features
        features = self.backbone(images)[-1]  # (batch_size, in_channels, feature_height, feature_width)

        # Project to the desired embedding dimension
        features = self.projection(features)  # (batch_size, embed_dim, feature_height, feature_width)

        # Move channels to the end
        features = features.permute(0, 2, 3, 1)  # (batch_size, feature_height, feature_width, embed_dim)

        # Generate positional embeddings
        feature_pos = generate_positional_embedding_sine(features)

        return features, feature_pos


class BackboneType(Protocol):
    feature_info: FeatureInfo

    def __call__(self, x: Tensor) -> List[Tensor]: ...


class ProjectionType(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...
