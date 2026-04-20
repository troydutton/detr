from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor
from torchvision.ops import nms
from torchvision.ops.boxes import box_convert


@dataclass
class Detections:
    boxes: Tensor
    labels: Tensor
    scores: Tensor
    categories: Optional[List[str]] = None


def postprocess(
    boxes: Tensor,
    logits: Tensor,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    categories: List[str] = None,
) -> List[Detections]:
    """
    Post-process raw predictions into detections by applying confidence thresholding and non-maximum suppression.

    Args:
        boxes: Predicted bounding boxes in normalized CXCYWH format with shape (batch_size, num_queries, 4).
        logits: Predicted class logits with shape (batch_size, num_queries, num_classes).
        confidence_threshold: Minimum confidence score for a prediction to be kept, optional.
        iou_threshold: IoU threshold for non-maximum suppression, optional.
        categories: Optional list of category names corresponding to class labels.

    Returns:
        detections: A list of `Detections` for each image in the batch, containing the filtered `boxes`, `labels`, `scores`, and optionally `categories`.
    """
    # Filter predictions
    scores, labels = logits.sigmoid().max(dim=-1)

    detections = []
    for image_boxes, image_labels, image_scores in zip(boxes, labels, scores):
        # Apply confidence thresholding
        keep = image_scores > confidence_threshold
        image_boxes, image_scores, image_labels = image_boxes[keep], image_scores[keep], image_labels[keep]

        # Apply non-maximum suppression
        keep = nms(box_convert(image_boxes, "cxcywh", "xyxy"), image_scores, iou_threshold)
        image_boxes, image_scores, image_labels = image_boxes[keep], image_scores[keep], image_labels[keep]

        if categories is not None:
            image_categories = [categories[label] for label in image_labels]
        else:
            image_categories = None

        detections.append(Detections(image_boxes, image_labels, image_scores, image_categories))

    return detections
