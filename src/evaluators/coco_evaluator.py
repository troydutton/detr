from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_convert

from evaluators.evaluator import Evaluator
from utils.misc import silence_stdout

if TYPE_CHECKING:
    from data import Target
    from models import Predictions


class CocoEvaluator(Evaluator):
    """
    COCO evaluator that computes AP metrics.

    Args:
        coco_targets: Ground truth COCO dataset.
    """

    def __init__(self, coco_targets: COCO) -> None:
        self.coco_targets = coco_targets
        self.predictions: List[Dict[str, Any]] = []

        # Mapping from contiguous 0-indexed labels to category ids
        category_ids = sorted(coco_targets.getCatIds())
        self.label_to_category_id = {i: cat_id for i, cat_id in enumerate(category_ids)}

    def reset(self) -> None:
        """Reset the internal state of the evaluator."""
        self.predictions.clear()

    def update(self, predictions: Predictions, targets: List[Target]) -> None:
        """
        Update the evaluator with a batch of predictions and targets.

        Args:
            predictions: Dictionary containing
                - `logits`: Logits with shape (batch_size, num_queries, num_classes).
                - `boxes`: Normalized boxes with shape (batch_size, num_queries, 4) in cxcywh format.
            targets: List of targets, where each target contains
                - `image_id`: Image ID.
                - `orig_size`: Original image size [height, width].
        """

        # Convert boxes from cxcywh to xywh
        prediction_boxes = predictions["boxes"]
        boxes = box_convert(prediction_boxes, in_fmt="cxcywh", out_fmt="xywh")

        # Get prediction scores and labels
        prediction_logits = predictions["logits"]
        scores, labels = prediction_logits.sigmoid().max(dim=-1)

        # Each target corresponds to an image in the batch
        for i, target in enumerate(targets):
            image_id = target["image_id"].item()
            height, width = target["orig_size"].tolist()

            # Scale boxes to original image size
            image_boxes = (boxes[i] * torch.tensor([width, height, width, height], device=boxes.device)).tolist()
            image_scores = scores[i].tolist()
            image_labels = labels[i].tolist()

            for box, score, label in zip(image_boxes, image_scores, image_labels):
                self.predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": self.label_to_category_id[label],
                        "bbox": box,
                        "score": score,
                    }
                )

    def compute(self) -> Dict[str, float]:
        """
        Compute the metrics.

        Returns:
            metrics: Dictionary mapping metric names to values.
        """

        if not self.predictions:
            return {}

        with silence_stdout():
            # Load predictions into COCO format
            coco_predictions = self.coco_targets.loadRes(self.predictions)

            coco_eval = COCOeval(self.coco_targets, coco_predictions, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = coco_eval.stats

        metrics = {
            "AP": stats[0],
            "AP50": stats[1],
            "AP75": stats[2],
            "APs": stats[3],
            "APm": stats[4],
            "APl": stats[5],
        }

        return metrics
