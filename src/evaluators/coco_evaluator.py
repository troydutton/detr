from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from accelerate import Accelerator
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

    def update(
        self,
        predictions: Tuple[Predictions, Optional[Predictions], Optional[Predictions]],
        targets: List[Target],
        accelerator: Optional[Accelerator] = None,
    ) -> None:
        """
        Update the evaluator with a batch of predictions and targets.

        Args:
            predictions: Decoder, optionally encoder, and optionally denoise predictions, with keys
                - `logits`: Class logits of shape (batch_size, num_layers, num_groups, num_queries, num_classes).
                - `boxes`: Predicted bounding boxes of shape (batch_size, num_layers, num_groups, num_queries, 4).
            targets: List of targets, where each target contains
                - `image_id`: Image ID.
                - `orig_size`: Original image size [height, width].
            accelerator: Distributed accelerator, optional.
        """

        # Take predictions from the first group in the final layer
        decoder_predictions, _, _ = predictions
        boxes = decoder_predictions.boxes[:, -1, 0]
        logits = decoder_predictions.logits[:, -1, 0]
        scores, labels = logits.sigmoid().max(dim=-1)

        # Extract image ids and scales
        image_scales = torch.empty(((batch_size := len(targets)), 4), dtype=boxes.dtype, device=(device := boxes.device))
        image_ids = torch.empty((batch_size,), dtype=torch.int, device=device)
        for i, target in enumerate(targets):
            image_ids[i] = target["image_id"].item()

            height, width = target["orig_size"].tolist()
            image_scales[i] = torch.tensor([width, height, width, height], dtype=boxes.dtype, device=device)

        # Convert boxes from normalized cxcywh to unnormalized xywh expected by COCO API
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xywh")
        boxes = boxes * image_scales[:, None, :]

        if accelerator is not None:
            image_ids = accelerator.gather_for_metrics(image_ids)
            boxes = accelerator.gather_for_metrics(boxes)
            scores = accelerator.gather_for_metrics(scores)
            labels = accelerator.gather_for_metrics(labels)

            if not accelerator.is_main_process:
                return

        image_ids = image_ids.tolist()
        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()

        for image_id, image_boxes, image_scores, image_labels in zip(image_ids, boxes, scores, labels):
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
