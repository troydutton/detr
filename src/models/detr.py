from torch import nn, Tensor
import torch
import math
from typing import Dict
from models.backbone import Backbone
from models.transformer import Transformer
from utils.misc import take_annotation_from
from models.layers import MultiLayerPerceptron
import logging

logger = logging.getLogger("detr")


class DETR(nn.Module):
    """
    Implementation of ["End-to-End Object Detection with Transformers"](https://arxiv.org/abs/2005.12872).

    Args:
        num_classes: Number of object classes.
        pretrained_weights: Path to a pretrained weights file.
        kwargs: Arguments to construct the backbone and transformer.
            See `models.backbone.Backbone` and `models.transformer.Transformer`.
    """

    def __init__(self, num_classes: int, pretrained_weights: str = None, **kwargs) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Create & initialize the bounding box and classification heads
        self.bbox_head = MultiLayerPerceptron(
            input_dim=(embed_dim := kwargs["transformer"]["embed_dim"]),
            hidden_dim=embed_dim,
            output_dim=4,
            num_layers=3,
        )
        self.class_head = nn.Linear(embed_dim, num_classes)

        # Build the backbone and transformer
        self.backbone = Backbone(**kwargs["backbone"])
        self.transformer = Transformer(**kwargs["transformer"])

        self._initialize_weights(pretrained_weights=pretrained_weights)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Predict

        Args:
            images: A batch of images with shape (batch, channels, height, width).

        Returns:
            predictions: A dictionary containing normalized CXCYWH `boxes` and raw class `logits`.
        """

        # Extract image features
        features, feature_pos = self.backbone(images)

        # Decode object queries for the image
        object_queries = self.transformer(features, feature_pos)

        # Predict bounding boxes and class logits
        predictions = {
            "boxes": self.bbox_head(object_queries).sigmoid(),
            "logits": self.class_head(object_queries),
        }

        return predictions

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)

    def _initialize_weights(self, pretrained_weights: str = None) -> None:
        # Check for BatchNorm layers
        for module_name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                logger.warning(f"Backbone contains BatchNorm layer: {module_name}")

        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")

            incompatible = self.load_state_dict(state_dict, strict=False)

            if incompatible.missing_keys:
                logger.warning(f"Missing keys when loading pretrained weights: {incompatible.missing_keys}")

            if incompatible.unexpected_keys:
                logger.warning(f"Unexpected keys when loading pretrained weights: {incompatible.unexpected_keys}")
        else:
            # We bias the classifier to predict a small probability for objects
            # because the majority of queries are not matched to objects
            bias = -math.log((1 - (object_prob := 0.01)) / object_prob)
            self.class_head.bias.data = torch.ones(self.num_classes) * bias
