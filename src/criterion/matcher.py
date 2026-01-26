from __future__ import annotations
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Dict, List, Tuple, TYPE_CHECKING
from utils.misc import take_annotation_from
from torchvision.ops.boxes import box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment

import torch

if TYPE_CHECKING:
    from criterion.criterion import Targets

Matches = List[Tuple[Tensor, Tensor]]


class HungarianMatcher(nn.Module):
    """
    Hungarian Matching algorithm for Object Detection tasks.

    Args:
        cost_weights: Weights for cost components (`class`, `box`, `giou`).
        alpha: Focal loss alpha parameter, optional.
        gamma: Focal loss gamma parameter, optional.
    """

    def __init__(self, cost_weights: Dict[str, float], alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()

        self.cost_weights = cost_weights
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, predictions: Dict[str, Tensor], targets: Targets) -> Matches:
        """
        Perform Hungarian matching between predictions and targets.

        Args:
            predictions: Model predictions, with keys:
                - `logits`: Class logits of shape (batch_size, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_queries, 4).
            targets: List of targets for each image, with keys:
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).

        Returns:
            matched_indices: Matched prediction and target indices for each image in the batch.
        """

        batch_size, _, _ = predictions["logits"].shape

        # TODO: Evaluate if batch processing is more efficient
        matched_indices = []

        for i in range(batch_size):
            # Retrieve image predictions and ground truths
            prediction_logits = predictions["logits"][i]
            prediction_boxes = predictions["boxes"][i]
            target_labels = targets[i]["labels"]
            target_boxes = targets[i]["boxes"]

            # Calculate individual costs
            box_cost, giou_cost = self._calculate_box_costs(prediction_boxes, target_boxes)
            class_cost = self._calculate_class_cost(prediction_logits, target_labels)

            # Weighted sum (num_predictions, num_targets)
            costs = {"class": class_cost, "box": box_cost, "giou": giou_cost}
            total_cost = sum(self.cost_weights.get(k, 1) * v for k, v in costs.items())

            # Handle NaN and infinite values
            total_cost.nan_to_num_(nan=1e6, posinf=1e6, neginf=-1e6)
            total_cost.clamp_(-1e6, 1e6)

            # Solve the linear sum assignment problem
            prediction_indices, target_indices = linear_sum_assignment(total_cost.cpu())

            # Store the prediction and target indices for the image
            prediction_indices = torch.as_tensor(prediction_indices, dtype=torch.int64, device=total_cost.device)
            target_indices = torch.as_tensor(target_indices, dtype=torch.int64, device=total_cost.device)
            matched_indices.append((prediction_indices, target_indices))

        return matched_indices

    def _calculate_box_costs(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        """
        Calculate L1 and GIoU costs between prediction and target boxes.

        Args:
            prediction_boxes: Predicted bounding boxes of shape (num_predictions, 4).
            target_boxes: Target bounding boxes of shape (num_targets, 4).

        Returns:
           box_cost: L1 distance between prediction and target boxes.
           #### giou_cost
           Generalized IoU cost between prediction and target boxes.
        """

        # Minimize L1 Distance
        box_cost = torch.cdist(prediction_boxes, target_boxes, p=1)

        # Maximize GIoU
        prediction_boxes = box_convert(prediction_boxes, "cxcywh", "xyxy")
        target_boxes = box_convert(target_boxes, "cxcywh", "xyxy")

        giou_cost = -generalized_box_iou(prediction_boxes, target_boxes)

        return box_cost, giou_cost

    def _calculate_class_cost(self, prediction_logits: Tensor, target_labels: Tensor) -> Tensor:
        """
        Calculate classification cost between predicted logits and target labels.

        Args:
            prediction_logits: Predicted class logits of shape (num_predictions, num_classes).
            target_labels: Target class labels of shape (num_targets,).

        Returns:
            class_cost: Classification cost between predicted logits and target labels.
        """

        # We use binary focal loss for classification
        prediction_probs = prediction_logits.sigmoid()

        # Numerically stable version of α * (1 - prob) ** γ * -prob.log()
        pos_class_cost = self.alpha * ((1 - prediction_probs) ** self.gamma) * (-F.logsigmoid(prediction_logits))

        # Numerically stable version of (1 - α) * prob ** γ * -(1 - prob).log()
        neg_class_cost = (1 - self.alpha) * (prediction_probs**self.gamma) * (-F.logsigmoid(-prediction_logits))

        class_cost = pos_class_cost[..., target_labels] - neg_class_cost[..., target_labels]  # (num_preds, num_classes)

        return class_cost

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
