from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor
from torchvision.ops.boxes import box_convert, box_iou, generalized_box_iou

from criterion.hungarian_matcher import HungarianMatcher

if TYPE_CHECKING:
    from criterion.hungarian_matcher import MatchIndices
    from data import Target
    from models import Predictions


class Criterion:
    """
    Implements set-based and denoising loss for object detection, following DETR and DINO.

    Args:
        loss_weights: Weights for different loss components.
        alpha: Focal loss alpha parameter, optional.
        gamma: Focal loss gamma parameter, optional.
    """

    def __init__(
        self,
        loss_weights: Dict[str, float],
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_groups: int = 1,
    ) -> None:
        self.loss_weights = loss_weights
        self.alpha = alpha
        self.gamma = gamma
        self.num_groups = num_groups

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
            predictions: Decoder, optionally encoder, and optionally denoise predictions, with keys
                - `logits`: Class logits of shape (batch_size, num_layers, num_groups, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, num_groups, num_queries, 4).
            targets: List of targets for each image, with keys
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).
            accelerator: Distributed accelerator, optional.

        Returns:
            losses: Dictionary of calculated losses.
        """

        decoder_predictions, encoder_predictions, denoise_predictions = predictions

        # We normalize the matching losses by the number of objects, averaged across all devices.
        objects_per_image = [len(t["boxes"]) for t in targets]
        num_targets = sum(objects_per_image)
        if accelerator is not None:
            num_targets = accelerator.reduce(torch.tensor(num_targets, dtype=torch.float, device=accelerator.device), reduction="mean")
        num_targets = max(num_targets, 1)

        losses = {"box": 0.0, "giou": 0.0, "class": 0.0}

        # Calculate loss for the decoder predictions
        matched_indices = self.matcher(decoder_predictions, targets)
        decoder_box_loss, decoder_giou_loss = self._calculate_box_losses(decoder_predictions, targets, matched_indices)
        decoder_class_loss = self._calculate_class_loss(decoder_predictions, targets, matched_indices)

        losses["box"] += decoder_box_loss / num_targets
        losses["giou"] += decoder_giou_loss / num_targets
        losses["class"] += decoder_class_loss / num_targets

        # Calculate loss for the encoder predictions
        if encoder_predictions is not None:
            matched_indices = self.matcher(encoder_predictions, targets)
            encoder_box_loss, encoder_giou_loss = self._calculate_box_losses(encoder_predictions, targets, matched_indices)
            encoder_class_loss = self._calculate_class_loss(encoder_predictions, targets, matched_indices)

            losses["box"] += encoder_box_loss / num_targets
            losses["giou"] += encoder_giou_loss / num_targets
            losses["class"] += encoder_class_loss / num_targets

        # Calculate loss for the denoising predictions
        if denoise_predictions is not None:
            matched_indices = self._get_denoise_match_indices(denoise_predictions, targets)
            denoise_box_loss, denoise_giou_loss = self._calculate_box_losses(denoise_predictions, targets, matched_indices)
            denoise_class_loss = self._calculate_class_loss(denoise_predictions, targets, matched_indices)

            # We normalize denoising losses by the number of positive samples, averaged across all devices.
            _, _, _, num_queries, _ = denoise_predictions.logits.shape
            num_targets = sum(n * (num_queries // (2 * n)) if n > 0 else 0 for n in objects_per_image)
            if accelerator is not None:
                num_targets = accelerator.reduce(torch.tensor(num_targets, dtype=torch.float, device=accelerator.device), reduction="mean")
            num_targets = max(num_targets, 1)

            losses["box"] += denoise_box_loss / num_targets
            losses["giou"] += denoise_giou_loss / num_targets
            losses["class"] += denoise_class_loss / num_targets

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

        prediction_boxes = box_convert(prediction_boxes, "cxcywh", "xyxy")
        target_boxes = box_convert(target_boxes, "cxcywh", "xyxy")

        # Maximize GIoU
        giou_loss = (1 - generalized_box_iou(prediction_boxes, target_boxes).diag()).sum()

        return box_loss, giou_loss

    def _calculate_class_loss(
        self,
        predictions: Predictions,
        targets: List[Target],
        matched_indices: MatchIndices,
    ) -> Tensor:
        """
        Calculate the classification loss between predictions and target labels.

        Uses varifocal loss which assigns IoU targets for matched predictions, and zero
        targets for unmatched predictions. Matched predictions are weighted by their IoU,
        while unmatched predictions are weighted according to the normal focal loss schema.

        For matched predictions:   - iou * (iou * log(p) + (1 - iou) * log(1 - p))
        For unmatched predictions: - (1 - α) * (p ** γ) * log(1 - p)

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
        prediction_logits = predictions.logits
        prediction_probs = prediction_logits.sigmoid()

        prediction_indices, target_indices = matched_indices
        target_labels = torch.cat([t["labels"][i] for t, i in zip(targets, target_indices)])  # (num_matches,)

        # Calculate IoU for matched predictions
        with torch.no_grad():
            prediction_boxes = predictions.boxes[prediction_indices]  # (num_matches, 4)
            target_boxes = torch.cat([t["boxes"][i] for t, i in zip(targets, target_indices)])  # (num_matches, 4)

            prediction_boxes = box_convert(prediction_boxes, "cxcywh", "xyxy")
            target_boxes = box_convert(target_boxes, "cxcywh", "xyxy")
            iou = box_iou(prediction_boxes, target_boxes).diag().clamp(min=0.01)

        # Build targets (IoU for matched predictions, 0 for unmatched)
        soft_target_labels = torch.zeros_like(prediction_probs)
        soft_target_labels[*prediction_indices, target_labels] = iou.to(soft_target_labels.dtype)

        # Build weights (IoU for matched predictions, focal weights for unmatched)
        weights = (1 - self.alpha) * (prediction_probs**self.gamma)
        weights[*prediction_indices, target_labels] = iou.to(weights.dtype)

        # Calculate the class loss
        bce_loss = F.binary_cross_entropy_with_logits(prediction_logits, soft_target_labels, reduction="none")
        class_loss = (weights * bce_loss).sum()

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
        objects_per_image = [len(t["boxes"]) for t in targets]

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
