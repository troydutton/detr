from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops.boxes import box_convert, generalized_box_iou

if TYPE_CHECKING:
    from data import Target
    from models import Predictions

MatchIndices = List[Tuple[Tensor, Tensor]]


class HungarianMatcher:
    """
    Hungarian Matching algorithm for Object Detection tasks.

    Args:
        cost_weights: Weights for cost components (`class`, `box`, `giou`).
        alpha: Focal loss alpha parameter, optional.
        gamma: Focal loss gamma parameter, optional.
    """

    def __init__(self, cost_weights: Dict[str, float], alpha: float = 0.25, gamma: float = 2.0) -> None:
        self.cost_weights = cost_weights
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def __call__(self, predictions: Predictions, targets: List[Target]) -> MatchIndices:
        """
        Perform Hungarian matching between predictions and targets.

        Args:
            predictions: Model predictions, with keys:
                - `logits`: Class logits of shape (batch_size, num_layers, num_groups, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, num_groups, num_queries, 4).
            targets: List of targets for each image, with keys:
                - `labels`: Target class labels of shape (num_targets,).
                - `boxes`: Target bounding boxes of shape (num_targets, 4).

        Returns:
            matched_indices: Matched prediction and target indices for each image and layer in the batch.
        """

        # Get batch information
        matches_per_image = [len(target["labels"]) for target in targets]
        batch_size, num_layers, num_groups, num_queries, _ = predictions.logits.shape
        device = predictions.logits.device

        # TODO: Improve efficiency by batching cost calculations across images, layers, and groups
        batch_indices, layer_indices, group_indices, query_indices, target_indices = [], [], [], [], []

        for i in range(batch_size):
            # Retrieve targets for the image
            target_labels = targets[i]["labels"]
            target_boxes = targets[i]["boxes"]

            # Retrieve predictions for the image and flatten across layers and groups
            prediction_logits = predictions.logits[i].flatten(0, 2)
            prediction_boxes = predictions.boxes[i].flatten(0, 2)

            # Calculate individual costs
            box_cost, giou_cost = self._calculate_box_costs(prediction_boxes, target_boxes)
            class_cost = self._calculate_class_cost(prediction_logits, target_labels)

            # Weighted sum (num_predictions, num_targets)
            costs = {"class": class_cost, "box": box_cost, "giou": giou_cost}
            total_cost = sum(self.cost_weights.get(k, 1) * v for k, v in costs.items())

            # Handle NaN and infinite values
            total_cost.nan_to_num_(nan=1e6, posinf=1e6, neginf=-1e6)
            total_cost.clamp_(-1e6, 1e6)

            # Move to CPU once and reshape to (num_layers, num_groups, num_queries, num_targets)
            total_cost_cpu = total_cost.cpu().view(num_layers, num_groups, num_queries, -1)

            image_target_indices = []
            for l in range(num_layers):
                for g in range(num_groups):
                    # Extract the specific cost matrix for this layer and group
                    cost_matrix = total_cost_cpu[l, g]

                    # Solve the linear sum assignment problem
                    indices = linear_sum_assignment(cost_matrix)

                    query_indices.append(indices[0])
                    image_target_indices.append(indices[1])

                    group_indices.append(torch.full((matches_per_image[i],), g))
                layer_indices.append(torch.full((matches_per_image[i] * num_groups,), l))
            batch_indices.append(torch.full((matches_per_image[i] * num_layers * num_groups,), i))

            image_target_indices = torch.from_numpy(np.concatenate(image_target_indices))
            target_indices.append(image_target_indices.to(device))

        query_indices = torch.from_numpy(np.concatenate(query_indices))

        # Transfer all indices to the device in use
        batch_indices = torch.cat(batch_indices).to(device)
        layer_indices = torch.cat(layer_indices).to(device)
        group_indices = torch.cat(group_indices).to(device)
        query_indices = query_indices.to(device)

        return (batch_indices, layer_indices, group_indices, query_indices), target_indices

    def _calculate_box_costs(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tuple[Tensor, Tensor]:
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
