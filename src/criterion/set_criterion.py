from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor
from torchvision.ops.boxes import box_convert, generalized_box_iou

from criterion.criterion import Criterion
from criterion.hungarian_matcher import HungarianMatcher

if TYPE_CHECKING:
    from criterion.hungarian_matcher import MatchIndices
    from data import Target
    from models import Predictions


class SetCriterion(Criterion):
    """

    Args:
        loss_weights: Weights for different loss components.
        class_weights: Weights for each class, optional.
        alpha: Focal loss alpha parameter, optional.
        gamma: Focal loss gamma parameter, optional.
    """

    def __init__(
        self,
        loss_weights: Dict[str, float],
        class_weights: Optional[Tensor] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_groups: int = 1,
    ) -> None:
        self.loss_weights = loss_weights
        self.class_weights = class_weights
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

        # Calculate loss for the decoder predictions
        matched_indices = self.matcher(decoder_predictions, targets)
        box_loss, giou_loss = self._calculate_box_losses(decoder_predictions, targets, matched_indices)
        class_loss = self._calculate_class_loss(decoder_predictions, targets, matched_indices)

        losses = {
            "box": box_loss,
            "giou": giou_loss,
            "class": class_loss,
        }

        # Calculate loss for the encoder predictions
        if encoder_predictions is not None:
            matched_indices = self.matcher(encoder_predictions, targets)
            box_loss, giou_loss = self._calculate_box_losses(encoder_predictions, targets, matched_indices)
            class_loss = self._calculate_class_loss(encoder_predictions, targets, matched_indices)

            losses["box"] += box_loss
            losses["giou"] += giou_loss
            losses["class"] += class_loss

        # Calculate loss for the denoising predictions
        if denoise_predictions is not None:
            matched_indices = self._get_denoise_match_indices(denoise_predictions, targets)
            box_loss, giou_loss = self._calculate_box_losses(denoise_predictions, targets, matched_indices)
            class_loss = self._calculate_class_loss(denoise_predictions, targets, matched_indices)

            losses["box"] += box_loss
            losses["giou"] += giou_loss
            losses["class"] += class_loss

        # Normalize losses by the number of target boxes (averaged across all devices if distributed)
        num_targets = sum(len(t["boxes"]) for t in targets)
        if accelerator is not None:
            num_targets = accelerator.reduce(torch.tensor(num_targets, dtype=torch.float, device=accelerator.device), reduction="mean")
        num_targets = max(num_targets, 1)
        losses = {k: v / num_targets for k, v in losses.items()}

        # Weighted sum
        losses["overall"] = sum(v * self.loss_weights.get(k, 1) for k, v in losses.items())

        return losses

    def _get_denoise_match_indices(self, denoise_predictions: Predictions, targets: List[Target]) -> MatchIndices:
        """
        Calculates matched indices for denoising queries.

        Args:
            denoise_predictions: Denoise predictions, with keys:
                - `logits`: Class logits of shape (batch_size, num_layers, num_groups, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, num_groups, num_queries, 4).
            targets: List of targets for each image.

        Returns:
            matched_indices: Matched prediction and target indices.
        """

        # Get batch information
        _, num_layers, _, num_queries, _ = denoise_predictions.logits.shape
        device = denoise_predictions.logits.device
        objects_per_image = [len(t["boxes"]) for t in targets]
        max_objects = max(objects_per_image)

        # Each group consists of a positive and negative for each object, so the
        # maximum size of each group is 2 * max_objects
        num_denoise_groups = num_queries // (2 * max_objects)

        batch_indices, layer_indices, group_indices, query_indices, target_indices = [], [], [], [], []

        # TODO: This is unreadable, probably chill though
        for b, num_objects in enumerate(objects_per_image):
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

        # We retain all  predictions to supervise unmatched predictions
        prediction_logits = predictions.logits
        prediction_probs = prediction_logits.sigmoid()

        # Build one-hot encoded target labels for the matched predictions
        prediction_indices, target_indices = matched_indices
        target_labels = torch.cat([t["labels"][i] for t, i in zip(targets, target_indices)])  # (num_matches,)

        one_hot_target_labels = torch.zeros_like(prediction_probs)
        one_hot_target_labels[*prediction_indices, target_labels] = 1.0

        # Numerically stable version of α * (1 - prob) ** γ * -prob.log()
        pos_class_cost = self.alpha * ((1 - prediction_probs) ** self.gamma) * (-F.logsigmoid(prediction_logits))

        # Numerically stable version of (1 - α) * prob ** γ * -(1 - prob).log()
        neg_class_cost = (1 - self.alpha) * (prediction_probs**self.gamma) * (-F.logsigmoid(-prediction_logits))

        class_loss = (pos_class_cost * one_hot_target_labels + neg_class_cost * (1 - one_hot_target_labels)).sum()

        return class_loss
