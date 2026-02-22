import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torchvision.ops import nms
from torchvision.ops.boxes import box_convert

from models.backbone import Backbone
from models.decoder import TransformerDecoder
from models.encoder import TransformerEncoder


@dataclass
class Predictions:
    boxes: Tensor
    logits: Tensor


@dataclass
class ModelPredictions:
    decoder: Predictions
    encoder: Optional[Predictions] = None


@dataclass
class Detections:
    boxes: Tensor
    labels: Tensor
    scores: Tensor


class DETR(nn.Module):
    """
    Implementation of Detection Transformers orginally introduced in [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).

    Args:
        embed_dim: Embedding dimension.
        pretrained_weights: Path to a pretrained weights file.
        kwargs: Arguments to construct the backbone and transformer.
            See `models.backbone.Backbone`, `models.encoder.TransformerEncoder`, and `models.decoder.TransformerDecoder`.
    """

    def __init__(self, pretrained_weights: str = None, **kwargs) -> None:
        super().__init__()

        # Build the backbone, transformer encoder, and transformer decoder
        self.backbone = Backbone(**kwargs["backbone"])
        self.encoder = TransformerEncoder(**kwargs["encoder"])
        self.decoder = TransformerDecoder(**kwargs["decoder"])

        self._initialize_weights(pretrained_weights=pretrained_weights)

    def forward(self, images: Tensor) -> ModelPredictions:
        """
        Predict bounding boxes and class logits for a batch of input images.

        Args:
            images: A batch of images with shape (batch, channels, height, width).

        Returns:
            predictions: Decoder, and optionally encoder predictions, with normalized CXCYWH `boxes` and class `logits`.
        """

        # Extract image features
        features = self.backbone(images)

        # Encode the features
        features = self.encoder(features)

        # Decode the features into object predictions
        boxes, logits, encoder_boxes, encoder_logits = self.decoder(features)

        decoder_predictions = Predictions(boxes, logits)

        # Supervise encoder predictions if two-stage is enabled
        if self.decoder.two_stage:
            encoder_predictions = Predictions(encoder_boxes, encoder_logits)
        else:
            encoder_predictions = None

        return ModelPredictions(decoder_predictions, encoder_predictions)

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> Union[Detections, List[Tensor]]:
        """
        Predict bounding boxes and class logits for a batch of input images.

        Args:
            images: A batch of images with shape (batch, channels, height, width) or (channels, height, width).
            confidence_threshold: Minimum confidence score for a prediction to be kept.
            iou_threshold: IoU threshold for non-maximum suppression.

        Returns:
            detections: A detection for every image, containg the `boxes`, `labels`, and `scores` for the predicted objects.
        """

        # Handle single image input
        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Extract image features
        features = self.backbone(images)

        # Encode the features
        features = self.encoder(features)

        # Decode the features into object predictions
        boxes, logits, _, _ = self.decoder(features)

        # Only use predictions from the final layer and first query group
        boxes = boxes[:, -1, 0]
        logits = logits[:, -1, 0]

        scores, labels = logits.sigmoid().max(dim=-1)

        # Filter predictions
        detections = []
        for image_boxes, image_labels, image_scores in zip(boxes, labels, scores):
            # Apply confidence thresholding
            keep = image_scores > confidence_threshold
            image_boxes, image_scores, image_labels = image_boxes[keep], image_scores[keep], image_labels[keep]

            # Apply non-maximum suppression
            keep = nms(box_convert(image_boxes, "cxcywh", "xyxy"), image_scores, iou_threshold)
            image_boxes, image_scores, image_labels = image_boxes[keep], image_scores[keep], image_labels[keep]

            detections.append(Detections(image_boxes, image_labels, image_scores))

        return detections if len(detections) > 1 else detections[0]

    @torch.no_grad()
    def _initialize_weights(self, pretrained_weights: str = None) -> None:
        # Check for BatchNorm layers
        for module_name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                logging.warning(f"Backbone contains BatchNorm layer: {module_name}")

        if pretrained_weights is None:
            return

        logging.info(f"Loading pretrained weights from '{pretrained_weights}'.")

        if pretrained_weights.endswith((".pt", ".pth")):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
        elif pretrained_weights.endswith(".safetensors"):
            state_dict = load_file(pretrained_weights, device="cpu")
        else:
            raise ValueError(f"Unsupported pretrained weights format: {pretrained_weights}")

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logging.warning(f"Missing keys when loading pretrained weights: {incompatible.missing_keys}")

        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys when loading pretrained weights: {incompatible.unexpected_keys}")
