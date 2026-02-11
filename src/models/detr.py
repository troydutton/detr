import logging
import math
from typing import Dict

import torch
from torch import Tensor, nn

from models.backbone import Backbone
from models.decoder import TransformerDecoder
from models.encoder import TransformerEncoder
from models.layers import FFN

Predictions = Dict[str, Tensor]


class DETR(nn.Module):
    """
    Implementation of Detection Transformers orginally introduced in [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).

    Args:
        embed_dim: Embedding dimension.
        num_classes: Number of object classes.
        pretrained_weights: Path to a pretrained weights file.
        kwargs: Arguments to construct the backbone and transformer.
            See `models.backbone.Backbone`, `models.encoder.TransformerEncoder`, and `models.decoder.TransformerDecoder`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        pretrained_weights: str = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Build the backbone, transformer encoder, and transformer decoder
        self.backbone = Backbone(**kwargs["backbone"])
        self.encoder = TransformerEncoder(**kwargs["encoder"], num_classes=num_classes)
        self.decoder = TransformerDecoder(**kwargs["decoder"])

        # Create the bounding box and classification heads
        self.bbox_head = FFN(embed_dim, embed_dim, 4, 3)
        self.class_head = nn.Linear(embed_dim, num_classes)

        self._initialize_weights(pretrained_weights=pretrained_weights)

    def forward(self, images: Tensor) -> Predictions:
        """
        Predict bounding boxes and class logits for a batch of input images.

        Args:
            images: A batch of images with shape (batch, channels, height, width).
            return_intermediates: Whether to return intermediate transformer outputs.

        Returns:
            predictions: A dictionary containing normalized CXCYWH `boxes` and raw class `logits`.
        """

        # Extract image features
        features = self.backbone(images)

        # Encode the features
        features, queries = self.encoder(features)

        # Decode object queries
        query_embed, query_ref = self.decoder(features, queries)

        # Predict bounding boxes
        offsets = self.bbox_head(query_embed)
        boxes = (query_ref.logit(1e-5) + offsets).sigmoid()

        # Predict class logits
        logits = self.class_head(query_embed)

        return {
            "boxes": boxes,
            "logits": logits,
        }

    @torch.no_grad()
    def _initialize_weights(self, pretrained_weights: str = None) -> None:
        # Check for BatchNorm layers
        for module_name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                logging.warning(f"Backbone contains BatchNorm layer: {module_name}")

        # We bias the classifier to predict a small probability for objects
        # because the majority of queries are not matched to objects
        bias = -math.log((1 - (object_prob := 0.01)) / object_prob)
        nn.init.constant_(self.class_head.bias, bias)

        if pretrained_weights is None:
            return

        logging.info(f"Loading pretrained weights from '{pretrained_weights}'.")

        state_dict = torch.load(pretrained_weights, map_location="cpu")["model"]

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logging.warning(f"Missing keys when loading pretrained weights: {incompatible.missing_keys}")

        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys when loading pretrained weights: {incompatible.unexpected_keys}")
