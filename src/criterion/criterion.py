from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor

from criterion.hungarian_matcher import HungarianMatcher
from utils.boxes import pairwise_box_iou, pairwise_generalized_box_iou

if TYPE_CHECKING:
    from criterion.hungarian_matcher import MatchIndices
    from data import Target
    from models import Predictions


class Criterion:
    """
    Implements box and classification losses for object detection.

    Args:
        loss_weights: Weights for different loss components.
        alpha: Quality weighting parameter, optional.
        gamma: Focal loss gamma parameter, optional.
    """

    def __init__(
        self,
        loss_weights: Dict[str, float],
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        self.loss_weights = loss_weights
        self.alpha = alpha
        self.gamma = gamma

        self.matcher = HungarianMatcher(cost_weights=loss_weights, alpha=alpha, gamma=gamma)

    def __call__(
        self,
        predictions: Tuple[Predictions, Optional[Predictions], Optional[Predictions]],
        targets: List[Target],
        accelerator: Optional[Accelerator] = None,
    ) -> Dict[str, Tensor]:
        """
        Calculates losses for the given predictions and targets.

        Args:
            predictions: Decoder, encoder, and denoising predictions, with keys
                - `logits`: Class logits of shape (batch_size, num_layers, num_groups, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, num_groups, num_queries, 4).
            targets: List of targets for each image, with keys
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).
            accelerator: Distributed accelerator, optional.

        Returns:
            losses: Dictionary of losses, including `box`, `giou`, `class`, and `overall`.
        """

        decoder_predictions, encoder_predictions, denoise_predictions = predictions

        # Traditional object query losses are normalized by the number of objects across
        # all groups, excluding the number of layers to allow each layer to contribute equally.
        _, _, num_decoder_groups, num_decoder_queries, _ = decoder_predictions.logits.shape
        decoder_objects_per_image = [min(len(t["boxes"]), num_decoder_queries) for t in targets]
        num_decoder_targets = sum(decoder_objects_per_image) * num_decoder_groups

        if accelerator is not None:
            num_decoder_targets = torch.tensor(num_decoder_targets, dtype=torch.float, device=accelerator.device)
            num_decoder_targets = accelerator.reduce(num_decoder_targets, reduction="mean")

        num_decoder_targets = max(num_decoder_targets, 1)

        losses = {"box": 0.0, "giou": 0.0, "class": 0.0}

        # Decoder losses
        matched_indices = self.matcher(decoder_predictions, targets)
        decoder_box_loss, decoder_giou_loss = self._calculate_box_losses(decoder_predictions, targets, matched_indices)
        decoder_class_loss = self._calculate_class_loss(decoder_predictions, targets, matched_indices)

        losses["box"] += decoder_box_loss / num_decoder_targets
        losses["giou"] += decoder_giou_loss / num_decoder_targets
        losses["class"] += decoder_class_loss / num_decoder_targets

        # Encoder losses
        if encoder_predictions is not None:
            _, _, num_encoder_groups, num_encoder_queries, _ = encoder_predictions.logits.shape
            encoder_objects_per_image = [min(len(t["boxes"]), num_encoder_queries) for t in targets]
            num_encoder_targets = sum(encoder_objects_per_image) * num_encoder_groups

            if accelerator is not None:
                num_encoder_targets = torch.tensor(num_encoder_targets, dtype=torch.float, device=accelerator.device)
                num_encoder_targets = accelerator.reduce(num_encoder_targets, reduction="mean")

            num_encoder_targets = max(num_encoder_targets, 1)

            matched_indices = self.matcher(encoder_predictions, targets)
            encoder_box_loss, encoder_giou_loss = self._calculate_box_losses(encoder_predictions, targets, matched_indices)
            encoder_class_loss = self._calculate_class_loss(encoder_predictions, targets, matched_indices)

            losses["box"] += encoder_box_loss / num_encoder_targets
            losses["giou"] += encoder_giou_loss / num_encoder_targets
            losses["class"] += encoder_class_loss / num_encoder_targets

        # Denoising losses
        if denoise_predictions is not None:
            # Denoising losses are normalized by the number of positive denoising samples,
            # determined on the fly because we vary the number of denoising query groups
            num_denoise_queries = denoise_predictions.logits.shape[3]
            objects_per_image = [min(len(t["boxes"]), num_denoise_queries // 2) for t in targets]
            num_denoise_targets = sum(n * (num_denoise_queries // (2 * n)) if n > 0 else 0 for n in objects_per_image)

            if accelerator is not None:
                num_denoise_targets = torch.tensor(num_denoise_targets, dtype=torch.float, device=accelerator.device)
                num_denoise_targets = accelerator.reduce(num_denoise_targets, reduction="mean")

            num_denoise_targets = max(num_denoise_targets, 1)

            matched_indices = self._get_denoise_match_indices(denoise_predictions, targets)
            denoise_box_loss, denoise_giou_loss = self._calculate_box_losses(denoise_predictions, targets, matched_indices)
            denoise_class_loss = self._calculate_class_loss(denoise_predictions, targets, matched_indices)

            losses["box"] += denoise_box_loss / num_denoise_targets
            losses["giou"] += denoise_giou_loss / num_denoise_targets
            losses["class"] += denoise_class_loss / num_denoise_targets

        # The overall loss is a weighted sum of the individual loss components
        losses["overall"] = sum(v * self.loss_weights.get(k, 1) for k, v in losses.items())

        return losses

    def _calculate_box_losses(
        self,
        predictions: Predictions,
        targets: List[Target],
        matched_indices: MatchIndices,
    ) -> Tuple[Tensor, Tensor]:
        """
        Calculate L1 and GIoU loss between matched predictions and target boxes.

        Args:
            predictions: Model predictions, with keys
                - `logits`: Class logits of shape (batch_size, num_layers, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, num_queries, 4).
            targets: List of targets for each image, with keys
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).
            matched_indices: Matched prediction and target indices.

        Returns:
            box_loss: L1 loss between matched boxes (summed).
            #### giou_loss
            GIoU loss between matched boxes (summed).
        """

        # Select matched predictions and targets
        prediction_indices, target_indices = matched_indices

        prediction_boxes = predictions.boxes[prediction_indices]  # (num_matches, 4)
        target_boxes = torch.cat([t["boxes"][i] for t, i in zip(targets, target_indices)])  # (num_matches, 4)

        # No targets, return zero losses
        if target_boxes.numel() == 0:
            return torch.tensor(0.0, device=target_boxes.device), torch.tensor(0.0, device=target_boxes.device)

        # Minimize L1 Distance
        box_loss = F.l1_loss(prediction_boxes, target_boxes, reduction="sum")

        # Maximize GIoU
        giou_loss = (1 - pairwise_generalized_box_iou(prediction_boxes, target_boxes, box_format="cxcywh")).sum()

        return box_loss, giou_loss

    def _calculate_class_loss(
        self,
        predictions: Predictions,
        targets: List[Target],
        matched_indices: MatchIndices,
    ) -> Tensor:
        """
        Calculate the classification loss between predictions and target labels.

        Uses an IoU-aware focal loss which assigns quality-based targets for matched predictions,
        and zero targets for unmatched predictions. Quality targets are the weighted geometric mean
        of the predicted class probability and the IoU, encouraging the model to align the classification
        and regression tasks. Specifically,

        Matched:   - q * ((1 - p) ** γ) * log(p) - (1 - q) * (p ** γ) * log(1 - p)
        Unmatched: - (p ** γ) * log(1 - p)

        where q = (p^α) * (iou^(1-α))

        Args:
            predictions: Model predictions, with keys
                - `logits`: Class logits of shape (batch_size, num_layers, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, num_queries, 4).
            targets: List of targets for each image, with keys
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).
            matched_indices: Matched prediction and target indices.

        Returns:
            class_loss: Classification loss (summed).
        """

        # Get classification predictions and targets
        prediction_indices, target_indices = matched_indices

        prediction_logits = predictions.logits
        prediction_probs = prediction_logits.sigmoid()
        target_labels = torch.cat([t["labels"][i] for t, i in zip(targets, target_indices)])
        matched_prediction_probs = prediction_probs[*prediction_indices, target_labels]

        # Calculate quality scores for matched predictions: (p^α) * (iou^(1-α))
        with torch.no_grad():
            prediction_boxes = predictions.boxes[prediction_indices]
            target_boxes = torch.cat([t["boxes"][i] for t, i in zip(targets, target_indices)])

            iou = pairwise_box_iou(prediction_boxes, target_boxes, box_format="cxcywh")

            quality = ((matched_prediction_probs**self.alpha) * (iou ** (1 - self.alpha))).clamp(min=0.01)

        # Build positive weights: q * (1 - p) ** γ for matched, 0 for unmatched
        pos_weights = torch.zeros_like(prediction_probs)
        pos_weights[*prediction_indices, target_labels] = (quality * ((1 - matched_prediction_probs) ** self.gamma)).to(pos_weights.dtype)

        # Build negative weights: (1 -q) * (p ** γ) for matched, (p ** γ) for unmatched
        neg_weights = prediction_probs**self.gamma
        neg_weights[*prediction_indices, target_labels] *= (1 - quality).to(neg_weights.dtype)

        # Numerically stable variant of -positive_weights * log(p) - negative_weights * log(1 - p)
        class_loss = (neg_weights * prediction_logits - F.logsigmoid(prediction_logits) * (pos_weights + neg_weights)).sum()

        return class_loss

    def _get_denoise_match_indices(self, denoise_predictions: Predictions, targets: List[Target]) -> MatchIndices:
        """
        Calculates matched indices for denoising queries.

        Args:
            denoise_predictions: Denoise predictions, with keys:
                - `logits`: Class logits of shape (batch_size, num_layers, 1, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, 1, num_queries, 4).
            targets: List of targets for each image.

        Returns:
            matched_indices: Matched prediction and target indices.
        """

        # Get batch information
        _, num_layers, _, num_queries, _ = denoise_predictions.logits.shape
        device = denoise_predictions.logits.device
        objects_per_image = [min(len(t["boxes"]), num_queries // 2) for t in targets]

        batch_indices, layer_indices, group_indices, query_indices, target_indices = [], [], [], [], []

        # TODO: This is unreadable, probably chill though
        for b, num_objects in enumerate(objects_per_image):
            num_denoise_groups = num_queries // (2 * num_objects) if num_objects > 0 else 0

            batch_indices.append(torch.full((num_objects * num_denoise_groups * num_layers,), b))
            layer_indices.append(torch.arange(num_layers).repeat_interleave(num_objects * num_denoise_groups))
            group_indices.append(torch.zeros((num_objects * num_denoise_groups * num_layers,)))

            denoise_group_indices = (torch.arange(num_objects * num_denoise_groups) // num_objects).repeat(num_layers)
            denoise_group_offsets = denoise_group_indices * (2 * num_objects)
            query_index = torch.arange(num_objects).repeat(num_denoise_groups * num_layers) + denoise_group_offsets
            query_indices.append(query_index)

            target_index = torch.arange(num_objects, device=device).repeat(num_denoise_groups * num_layers)
            target_indices.append(target_index)

        batch_indices = torch.cat(batch_indices).to(device).int()
        layer_indices = torch.cat(layer_indices).to(device).int()
        group_indices = torch.cat(group_indices).to(device).int()
        query_indices = torch.cat(query_indices).to(device).int()

        return (batch_indices, layer_indices, group_indices, query_indices), target_indices
