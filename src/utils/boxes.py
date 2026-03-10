import torch
from torch import Tensor
from torchvision.ops.boxes import box_area, box_convert


def pairwise_box_iou(boxes1: Tensor, boxes2: Tensor, box_format: str = "xyxy") -> Tensor:
    """
    Calculate the pairwise Intersection over Union (IoU) of two sets of boxes.
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

    union = area1 + area2 - inter

    return inter / union


def pairwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor, box_format: str = "xyxy") -> Tensor:
    """
    Calculate the pairwise Generalized Intersection over Union (GIoU) of two sets of boxes.
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

    union = area1 + area2 - inter
    iou = inter / union

    lti = torch.min(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    whi = (rbi - lti).clamp(min=0)
    areai = whi[:, 0] * whi[:, 1]

    return iou - (areai - union) / areai


def box_intersection(boxes1: Tensor, boxes2: Tensor, box_format: str = "xyxy") -> Tensor:
    """
    Computes the intersection between two sets of boxes, stolen from `torchvision.ops.boxes`.

    Args:
        boxes1: First set of boxes, with shape (N, 4).
        boxes2: Second set of boxes, with shape (M, 4).
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
