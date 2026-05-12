import torch
from torch import Tensor
from torchvision.ops.boxes import box_convert

from utils.boxes import clamp_boxes


def make_edge_offset_weights(
    num_bins: int,
    offset_magnitude: float = 0.5,
    offset_curvature: float = 0.25,
) -> Tensor:
    """
    Generates a non-uniform weighting function W(n) for distribution-based regression.

    W(n) maps each discrete bin index n ∈ [0, num_bins] to a signed offset value.
    The function is symmetric around zero, with small curvature near the center
    (enabling fine-grained adjustments) and large curvature near the edges
    (enabling coarse corrections).

    Args:
        num_bins: Number of discrete offset bins.
        offset_magnitude: Absolute offset magnitude.
        offset_curvature: Curvature of the weighting function.

    Returns:
        weights: Bin weights with shape (num_bins + 1,).
    """

    weights = torch.zeros(num_bins + 1, dtype=torch.float32)

    # Create tensor for the step indices: [0, 1, ..., num_bins]
    bin_indices = torch.arange(num_bins + 1, dtype=torch.float32)

    # Common base factor for the exponential curve
    exponential_base = (offset_magnitude / offset_curvature) + 1.0

    # Create boolean masks for the two main piecewise conditions
    left_mask = (bin_indices >= 1) & (bin_indices < num_bins / 2)
    right_mask = (bin_indices >= num_bins / 2) & (bin_indices <= num_bins - 1)

    # Calculate exponents for the middle segments
    left_exponents = (num_bins - 2 * bin_indices) / (num_bins - 2)
    right_exponents = (-num_bins + 2 * bin_indices) / (num_bins - 2)

    # Starting boundary condition (n = 0)
    weights[0] = -2 * offset_magnitude

    # Left half of the curve (1 <= n < N/2)
    weights[left_mask] = offset_curvature - offset_curvature * (exponential_base ** left_exponents[left_mask])

    # Right half of the curve (N/2 <= n <= N - 1)
    weights[right_mask] = -offset_curvature + offset_curvature * (exponential_base ** right_exponents[right_mask])

    # Ending boundary condition (n = N)
    weights[num_bins] = 2 * offset_magnitude

    return weights


def add_edge_offset(references: Tensor, edge_logits: Tensor, edge_weights: Tensor) -> Tensor:
    """
    Updates the reference boxes using the predicted edge offsets.

    The edge offset is defined as the weighted sum of the edge probabilities: Σ P(n) * W(n).
    We interpret the edge predictions in (left, top, right, bottom) format, because it
    allows us to easily vectorize calculations with boxes in (x1, y1, x2, y2) format.

    The updated reference boxes are calculated by scaling the edge offsets by their
    respective box dimensions and adding them to the initial box coordinates.

    Args:
        references: Initial reference boxes in CXCYWH with shape (..., 4).
        edge_logits: Edge logits with shape (..., 4 * (num_bins + 1)).
        edge_weights: Weighting function with shape (num_bins + 1,).

    Returns:
        updated_references: Updated reference boxes with shape (..., 4).
    """

    # Calculate the edge offsets
    edge_logits = edge_logits.reshape(*edge_logits.shape[:-1], 4, -1)
    edge_probs = edge_logits.softmax(dim=-1)
    edge_offsets = (edge_probs * edge_weights.to(edge_probs.device)).sum(dim=-1)

    # Update the references
    width, height = references[..., 2], references[..., 3]
    references = box_convert(references, in_fmt="cxcywh", out_fmt="xyxy")
    boxes = references + (edge_offsets * torch.stack([-width, -height, width, height], dim=-1))
    boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")

    return clamp_boxes(boxes, box_format="cxcywh")


def calculate_edge_offset(references: Tensor, target_boxes: Tensor) -> Tensor:
    """
    Calculates the edge offsets between the reference and target boxes.

    The edge offsets are defined as the distance between the initial reference box
    coordinates and the target box coordinates, normalized by the box dimensions.

    Args:
        references: Initial reference boxes in CXCYWH with shape (..., 4).
        target_boxes: Ground truth boxes in CXCYWH with shape (..., 4).

    Returns:
        edge_offsets: Offsets (left, top, right, bottom) with shape (..., 4).
    """

    width, height = references[..., 2], references[..., 3]
    references = box_convert(references, in_fmt="cxcywh", out_fmt="xyxy")
    target_boxes = box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # Calculate the relative edge offsets
    offsets = target_boxes - references
    offsets = offsets / torch.stack([-width, -height, width, height], dim=1)

    return offsets


def calculate_edge_offset_probs(references: Tensor, target_boxes: Tensor, edge_weights: Tensor) -> Tensor:
    """
    Computes a discrete target distribution for the edge offsets between the reference and target boxes.

    The edge offsets are defined as the distance between the initial reference box
    coordinates and the target box coordinates, normalized by the box dimensions.
    The resulting probability mass is interpolated between the two adjacent bins if
    the offset falls in-between them, or to the outside bin if it exceeds the weight range.

    Args:
        references: Initial reference boxes in CXCYWH with shape (num_objects, 4).
        target_boxes: Ground truth boxes in CXCYWH with shape (num_objects, 4).
        edge_weights: Weighting function with shape (num_bins + 1,).

    Returns:
        target_distribution: Target discrete distributions with shape (4 * num_objects, num_bins + 1).
    """

    # Get batch information
    num_objects = len(references)
    num_bins = len(edge_weights) - 1
    edge_weights = edge_weights.to(references.device)

    # Calculate edge offsets
    offsets = calculate_edge_offset(references, target_boxes).reshape(-1)

    # Find corresponding bin indices
    bin_indices = ((edge_weights[None, :] - offsets[:, None]) <= 0).sum(dim=-1) - 1

    # Calculate the target probabilities for each edge and bin
    target_probs = torch.zeros((num_objects * 4, num_bins + 1), device=references.device)

    # Interpolate weights for offsets that fall in-between two bins
    interpolate = (bin_indices >= 0) & (bin_indices < num_bins)
    interpolate_indices = bin_indices[interpolate]

    left_edge_weights = edge_weights[interpolate_indices]
    right_edge_weights = edge_weights[interpolate_indices + 1]

    left_distance = (offsets[interpolate] - left_edge_weights).abs()
    right_distance = (offsets[interpolate] - right_edge_weights).abs()
    bin_distance = (right_edge_weights - left_edge_weights).abs().clamp(min=1e-6)

    target_probs[interpolate, interpolate_indices] = right_distance / bin_distance
    target_probs[interpolate, interpolate_indices + 1] = left_distance / bin_distance

    # Assign full weight to the outside bin if the offset exceeds the weight range
    target_probs[bin_indices < 0, 0] = 1.0
    target_probs[bin_indices >= num_bins, -1] = 1.0

    return target_probs
