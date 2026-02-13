import logging
from typing import Dict

import torch
from torch import Tensor, nn

from models.backbone import Backbone
from models.decoder import TransformerDecoder
from models.encoder import TransformerEncoder

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

    def __init__(self, num_classes: int, pretrained_weights: str = None, **kwargs) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Build the backbone, transformer encoder, and transformer decoder
        self.backbone = Backbone(**kwargs["backbone"])
        self.encoder = TransformerEncoder(**kwargs["encoder"])
        self.decoder = TransformerDecoder(**kwargs["decoder"], num_classes=num_classes)

        self._initialize_weights(pretrained_weights=pretrained_weights)

    def forward(self, images: Tensor) -> Predictions:
        """
        Predict bounding boxes and class logits for a batch of input images.

        Args:
            images: A batch of images with shape (batch, channels, height, width).
            return_intermediates: Whether to return intermediate transformer outputs.

        Returns:
            predictions: A dictionary containing normalized CXCYWH `boxes` and class `logits`.
        """

        # Extract image features
        features = self.backbone(images)

        # Encode the features
        features = self.encoder(features)

        # Decode the features into object predictions
        boxes, logits = self.decoder(features)

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

        if pretrained_weights is None:
            return

        logging.info(f"Loading pretrained weights from '{pretrained_weights}'.")

        state_dict = torch.load(pretrained_weights, map_location="cpu")["model"]

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logging.warning(f"Missing keys when loading pretrained weights: {incompatible.missing_keys}")

        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys when loading pretrained weights: {incompatible.unexpected_keys}")
