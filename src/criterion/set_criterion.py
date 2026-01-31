from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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
    ) -> None:
        self.loss_weights = loss_weights
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

        self.matcher = HungarianMatcher(cost_weights=loss_weights, alpha=alpha, gamma=gamma)

    def __call__(self, predictions: Predictions, targets: List[Target]) -> Dict[str, Tensor]:
        """
        Calculates losses for the given predictions and targets.

        Args:
            predictions: Model predictions, with keys
                - `logits`: Class logits of shape (batch_size, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_queries, 4).
            targets: List of targets for each image, with keys
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).

        Returns:
            losses: Dictionary of calculated losses.
        """
        matched_indices = self.matcher(predictions, targets)

        box_loss, giou_loss = self._calculate_box_losses(predictions, targets, matched_indices)
        class_loss = self._calculate_class_loss(predictions, targets, matched_indices)

        # Weighted sum
        losses = {"box": box_loss, "giou": giou_loss, "class": class_loss}
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
                - `logits`: Class logits of shape (batch_size, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_queries, 4).
            targets: List of targets for each image, with keys
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).
            matched_indices: Matched prediction and target indices for each image in the batch.

        Returns:
            box_loss: L1 loss between matched boxes.
            #### giou_loss
            GIoU loss between matched boxes.
        """

        # Select the matched predictions and targets
        batch_indices = torch.cat([torch.full_like(indices, i) for i, (indices, _) in enumerate(matched_indices)])
        query_indices = torch.cat([indices for (indices, _) in matched_indices])
        prediction_indices = (batch_indices, query_indices)

        prediction_boxes = predictions["boxes"][prediction_indices]
        target_boxes = torch.cat([t["boxes"][indices] for t, (_, indices) in zip(targets, matched_indices)])
        num_targets = target_boxes.shape[0]

        # Minimize L1 Distance
        box_loss = F.l1_loss(prediction_boxes, target_boxes, reduction="sum")
        box_loss = box_loss / num_targets

        prediction_boxes = box_convert(prediction_boxes, "cxcywh", "xyxy")
        target_boxes = box_convert(target_boxes, "cxcywh", "xyxy")

        # Maximize GIoU
        giou_loss = (1 - generalized_box_iou(prediction_boxes, target_boxes).diag()).sum()
        giou_loss = giou_loss / num_targets

        return box_loss, giou_loss

    def _calculate_class_loss(self, predictions: Predictions, targets: List[Target], matched_indices: MatchIndices) -> Tensor:
        """
        Calculate the classification loss focal between matched predictions and target labels.

        Args:
            predictions: Model predictions, with keys
                - `logits`: Class logits of shape (batch_size, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_queries, 4).
            targets: List of targets for each image, with keys
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).
            matched_indices: Matched prediction and target indices for each image in the batch.

        Returns:
            class_loss: Classification loss.
        """

        # Select the matched predictions and targets
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_indices)])
        query_indices = torch.cat([indices for (indices, _) in matched_indices])

        prediction_logits = predictions["logits"]  # (batch_size, num_queries, num_classes)

        target_labels = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, matched_indices)])  # (num_targets)
        num_targets = target_labels.shape[0]

        # We use binary focal loss for classification
        prediction_probs = prediction_logits.sigmoid()

        # One-hot encode target labels
        one_hot_target_labels = torch.zeros_like(prediction_probs)
        one_hot_target_labels[batch_indices, query_indices, target_labels] = 1.0

        # Numerically stable version of α * (1 - prob) ** γ * -prob.log()
        pos_class_cost = self.alpha * ((1 - prediction_probs) ** self.gamma) * (-F.logsigmoid(prediction_logits))

        # Numerically stable version of (1 - α) * prob ** γ * -(1 - prob).log()
        neg_class_cost = (1 - self.alpha) * (prediction_probs**self.gamma) * (-F.logsigmoid(-prediction_logits))

        class_loss = pos_class_cost * one_hot_target_labels + neg_class_cost * (1 - one_hot_target_labels)
        class_loss = class_loss.sum() / num_targets

        return class_loss
