import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from data.coco_dataset import Target
from models.backbone import Backbone
from models.decoder import TransformerDecoder
from models.encoder import TransformerEncoder
from utils.misc import take_annotation_from
from utils.postprocess import Detections, postprocess

VALID_WEIGHT_FILES = ["ema_model.safetensors", "model.safetensors"]


@dataclass
class Predictions:
    boxes: Tensor
    logits: Tensor


class DETR(nn.Module):
    """
    Implementation of Detection Transformers orginally introduced in [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).

    Args:
        pretrained_weights: Path to a pretrained weights file.
        categories: Category names used for predictions, optional.
        kwargs: Arguments to construct the backbone and transformer.
            See `models.backbone.Backbone`, `models.encoder.TransformerEncoder`, and `models.decoder.TransformerDecoder`.
    """

    def __init__(self, pretrained_weights: str = None, categories: List[str] = None, **kwargs) -> None:
        super().__init__()

        # Build the backbone
        kwargs["backbone"]["enable_level_pos"] = kwargs["encoder"]["num_layers"] > 0
        self.backbone = Backbone(**kwargs["backbone"])

        # Build the transformer encoder and decoder
        kwargs["encoder"]["layer"]["num_levels"] = self.backbone.num_output_levels
        self.encoder = TransformerEncoder(**kwargs["encoder"])

        kwargs["decoder"]["layer"]["num_levels"] = self.backbone.num_output_levels
        self.decoder = TransformerDecoder(**kwargs["decoder"])

        # Label to category name mapping for use in predictions
        self.categories = categories

        self._initialize_weights(pretrained_weights=pretrained_weights)

    def forward(self, images: Tensor, targets: List[Target] = None) -> Tuple[Predictions, Optional[Predictions], Optional[Predictions]]:
        """
        Predict bounding boxes and class logits for a batch of input images.

        Args:
            images: Batch of images with shape (batch, channels, height, width).
            targets: List of targets for each image, optional.

        Returns:
            predictions: Decoder, optionally encoder, and optionally denoising query predictions, with normalized CXCYWH `boxes` and class `logits`.
        """

        # Extract image features
        features = self.backbone(images)

        # Encode the features
        features = self.encoder(features)

        # Decode the features into object predictions
        boxes, logits, encoder_boxes, encoder_logits, denoise_boxes, denoise_logits = self.decoder(features, targets=targets)

        decoder_predictions = Predictions(boxes, logits)

        encoder_predictions, denoise_predictions = None, None

        # Supervise encoder predictions if enabled
        if self.decoder.two_stage and self.training and encoder_boxes is not None and encoder_logits is not None:
            encoder_predictions = Predictions(encoder_boxes, encoder_logits)

        # Supervised denoising predictions if enabled
        if self.decoder.denoise_queries and self.training and denoise_boxes is not None and denoise_logits is not None:
            denoise_predictions = Predictions(denoise_boxes, denoise_logits)

        return decoder_predictions, encoder_predictions, denoise_predictions

    @torch.no_grad()
    def predict(
        self,
        images: Tensor,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        export: bool = False,
    ) -> Union[List[Detections], Tuple[Tensor, Tensor]]:
        """
        Predict bounding boxes and class logits for a batch of input images.

        Args:
            images: A batch of images with shape (batch, channels, height, width).
            confidence_threshold: Minimum confidence score for a prediction to be kept.
            iou_threshold: IoU threshold for non-maximum suppression.
            export: Whether to return raw predictions for ONNX export.

        Returns:
            detections: A list of `Detections` for each image in the batch, containing the filtered `boxes`, `labels`, `scores`, and optionally `categories`.
                Alternatively, returns the raw `boxes` and `logits` predictions for ONNX export.
        """

        # Extract image features
        features = self.backbone(images)

        # Encode the features
        features = self.encoder(features)

        # Decode the features into object predictions
        boxes, logits, _, _, _, _ = self.decoder(features, targets=None)

        # Only use predictions from the final layer and first query group
        boxes = boxes[:, -1, 0]
        logits = logits[:, -1, 0]

        # Return unfiltered predictions for ONNX export
        if export:
            return boxes, logits

        # Filter predictions
        return postprocess(boxes, logits, confidence_threshold, iou_threshold, self.categories)

    @torch.no_grad()
    def _initialize_weights(self, pretrained_weights: Union[str, Path] = None) -> None:
        """
        Initialize model weights, optionally loading from a pretrained checkpoint.

        Args:
            pretrained_weights: Path to a pretrained weights file or an accelerate checkpoint directory containing a `model.safetensors` file.
        """

        # Check for BatchNorm layers
        for module_name, module in self.backbone.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                logging.warning(f"Backbone contains BatchNorm layer: {module_name}")

        if pretrained_weights is None:
            return

        pretrained_weights = Path(pretrained_weights)

        # Attempt to find the model weights in an accelerate checkpoint if a directory is provided
        if pretrained_weights.is_dir():

            def find_weight_file(directory: Path) -> Optional[Path]:
                for file in VALID_WEIGHT_FILES:
                    if (directory / file).exists():
                        return directory / file
                return None

            weight_path = find_weight_file(pretrained_weights)

            if weight_path is None:
                # Weights point to a parent folder containing multiple checkpoint folders
                checkpoints = sorted(d for d in pretrained_weights.iterdir() if d.is_dir() and find_weight_file(d) is not None)

                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoint directories containing model weights found in '{pretrained_weights}'.")

                weight_path = find_weight_file(checkpoints[-1])

            pretrained_weights = weight_path

        logging.info(f"Loading pretrained weights from '{pretrained_weights}'.")

        if pretrained_weights.suffix in [".pt", ".pth"]:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
        elif pretrained_weights.suffix == ".safetensors":
            state_dict = load_file(pretrained_weights, device="cpu")
        else:
            raise ValueError(f"Unsupported pretrained weights format: {pretrained_weights}")

        # Strip module prefix and EMA metadata if present
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items() if k != "n_averaged"}

        # During inference we use a single query group
        if len(self.decoder.queries.weight) < len(state_dict["decoder.queries.weight"]):
            state_dict["decoder.queries.weight"] = state_dict["decoder.queries.weight"][: len(self.decoder.queries.weight)]

        # When changing datasets during training the number of classes may differ
        if len(self.decoder.class_head.weight) != len(state_dict["decoder.class_head.weight"]):
            for key in list(state_dict.keys()):
                if "class_head" in key or "label_embed" in key:
                    state_dict.pop(key)

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logging.warning(f"Missing keys when loading pretrained weights: {incompatible.missing_keys}")

        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys when loading pretrained weights: {incompatible.unexpected_keys}")

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
