import torch
from torch import Tensor
from torchvision.ops.boxes import box_area, box_convert

EPSILON = 1e-4


def paired_box_iou(boxes1: Tensor, boxes2: Tensor, box_format: str = "xyxy") -> Tensor:
    """
    Calculate the paired Intersection over Union (IoU) of two sets of boxes.
    Equivalent to box_iou(boxes1, boxes2).diag() but avoids calculating the full NxN matrix.

    Args:
        boxes1: Bounding boxes with shape (N, 4).
        boxes2: Bounding boxes with shape (N, 4).
        box_format: The format of the bounding boxes ("xyxy", "cxcywh", etc.).

    Returns:
        iou: Pairwise IoU with shape (N,).
    """

    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(f"Both sets must have the same number of boxes, got {boxes1.shape[0]} and {boxes2.shape[0]}")

    boxes1 = box_convert(boxes1, box_format, "xyxy")
    boxes2 = box_convert(boxes2, box_format, "xyxy")

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter + EPSILON

    return inter / union


def paired_generalized_box_iou(boxes1: Tensor, boxes2: Tensor, box_format: str = "xyxy") -> Tensor:
    """
    Calculate the paired Generalized Intersection over Union (GIoU) of two sets of boxes.
    Equivalent to generalized_box_iou(boxes1, boxes2).diag() but avoids calculating the full NxN matrix.

    Args:
        boxes1: Bounding boxes with shape (N, 4).
        boxes2: Bounding boxes with shape (N, 4).
        box_format: The format of the bounding boxes ("xyxy", "cxcywh", etc.).

    Returns:
        giou: Pairwise GIoU with shape (N,).
    """

    if boxes1.shape[0] != boxes2.shape[0]:
        raise ValueError(f"Both sets must have the same number of boxes, got {boxes1.shape[0]} and {boxes2.shape[0]}")

    boxes1 = box_convert(boxes1, box_format, "xyxy")
    boxes2 = box_convert(boxes2, box_format, "xyxy")

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - inter + EPSILON
    iou = inter / union

    lti = torch.min(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    whi = (rbi - lti).clamp(min=0)
    areai = whi[:, 0] * whi[:, 1] + EPSILON

    return iou - (areai - union) / areai


def box_intersection(boxes1: Tensor, boxes2: Tensor, box_format: str = "xyxy") -> Tensor:
    """
    Computes the intersection between two sets of boxes, stolen from `torchvision.ops.boxes`.

    Args:
        boxes1: Bounding boxes with shape (N, 4).
        boxes2: Bounding boxes with shape (M, 4).
        box_format: The format of the bounding boxes ("xyxy", "cxcywh", etc.).

    Returns:
        intersection_area: The intersection between the boxes, with shape (N, M)
    """

    boxes1 = box_convert(boxes1, box_format, "xyxy")
    boxes2 = box_convert(boxes2, box_format, "xyxy")

    top_left = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N,M,2)
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N,M,2)

    width_height = (bottom_right - top_left).clamp(min=0)  # (N,M,2)
    intersection_area = width_height[:, :, 0] * width_height[:, :, 1]  # (N,M)

    return intersection_area


def clamp_boxes(boxes: Tensor, box_format: str = "cxcywh") -> Tensor:
    """
    Clamp boxes to lie within the unit square while remaining valid and non-degenerate.

    We clamp in XYXY: the bottom-right corner is clamped to [EPSILON, 1] and the top-left corner
    to [EPSILON, bottom_right - EPSILON]. This guarantees the box lies within the image, has a
    center point within the image, and has a non-zero area.

    Args:
        boxes: Bounding boxes with shape (..., 4).
        box_format: Format of the boxes ("xyxy", "cxcywh", ...), optional.

    Returns:
        boxes: Clamped bounding boxes with shape (..., 4).
    """

    boxes = box_convert(boxes, box_format, "xyxy")

    bottom_right = boxes[..., 2:].clamp(EPSILON, 1.0)
    top_left = boxes[..., :2].clamp(min=0).minimum(bottom_right - EPSILON)
    boxes = torch.cat([top_left, bottom_right], dim=-1)

    return box_convert(boxes, "xyxy", box_format)


def add_box_offsets(references: Tensor, offsets: Tensor) -> Tensor:
    """
    Updates the reference boxes using the predicted offsets.

    The updated reference boxes are defined as (x + (Δx * w), y + (Δy * h), w * exp(Δw), h * exp(Δh)).

    Args:
        references: Current reference boxes in CXCYWH with shape (..., 4).
        offsets: Predicted offsets with shape (..., 4).

    Returns:
        updated_references: Updated reference boxes with shape (..., 4).
    """

    xy = references[..., :2] + (offsets[..., :2] * references[..., 2:])
    wh = references[..., 2:] * offsets[..., 2:].clamp(-3.0, 3.0).exp()

    return clamp_boxes(torch.cat([xy, wh], dim=-1), box_format="cxcywh")
