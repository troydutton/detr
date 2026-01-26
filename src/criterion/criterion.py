from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.boxes import box_convert, box_iou, generalized_box_iou

from criterion.matcher import HungarianMatcher
from utils.misc import take_annotation_from

if TYPE_CHECKING:
    from criterion.matcher import Matches

Targets = List[Dict[str, Tensor]]


class SetCriterion(nn.Module):
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
        super().__init__()

        self.loss_weights = loss_weights
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

        self.matcher = HungarianMatcher(cost_weights=loss_weights, alpha=alpha, gamma=gamma)

    def forward(self, predictions: Dict[str, Tensor], targets: Targets) -> Dict[str, Tensor]:
        """
        Calculates losses for the given predictions and targets.

        Args:
            predictions: Model predictions, with keys:
                - `logits`: Class logits of shape (batch_size, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_queries, 4).
            targets: List of targets for each image, with keys:
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).

        Returns:
            losses: Dictionary of calculated losses.
        """
        matched_indices = self.matcher(predictions, targets)

        box_loss, giou_loss = self.calculate_box_losses(predictions, targets, matched_indices)
        class_loss = self.calculate_class_loss(predictions, targets, matched_indices)

        # Weighted sum
        losses = {"box": box_loss, "giou": giou_loss, "class": class_loss}
        losses["overall"] = sum(v * self.loss_weights.get(k, 1) for k, v in losses.items())

        return losses

    def calculate_box_losses(self, predictions: Dict[str, Tensor], targets: Targets, matched_indices: Matches) -> Tuple[Tensor, Tensor]:
        """
        Calculate L1 and GIoU loss between matched predictions and target boxes.

        Args:
            predictions: Model predictions, with keys:
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_queries, 4).

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

    def calculate_class_loss(self, predictions: Dict[str, Tensor], targets: Targets, matched_indices: Matches) -> Tensor:
        # Select the matched predictions and targets
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(matched_indices)])
        query_indices = torch.cat([indices for (indices, _) in matched_indices])

        prediction_logits = predictions["logits"]  # (batch_size, num_queries, num_classes)

        target_labels = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, matched_indices)])  # (num_targets)
        num_targets = target_labels.shape[0]

        # One-hot encode target labels
        one_hot_target_labels = torch.zeros_like(prediction_probs)
        one_hot_target_labels[batch_indices, query_indices, target_labels] = 1.0

        # We use binary focal loss for classification
        prediction_probs = prediction_logits.sigmoid()

        # Numerically stable version of α * (1 - prob) ** γ * -prob.log()
        pos_class_cost = self.alpha * ((1 - prediction_probs) ** self.gamma) * (-F.logsigmoid(prediction_logits))

        # Numerically stable version of (1 - α) * prob ** γ * -(1 - prob).log()
        neg_class_cost = (1 - self.alpha) * (prediction_probs**self.gamma) * (-F.logsigmoid(-prediction_logits))

        class_loss = pos_class_cost * one_hot_target_labels + neg_class_cost * (1 - one_hot_target_labels)
        class_loss = class_loss.sum() / num_targets

        return class_loss

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
