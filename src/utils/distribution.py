import torch
from torch import Tensor


def make_edge_weights(
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
        num_bins: Number of discrete distribution bins.
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


def calculate_edge_offsets(edge_logits: Tensor, edge_weights: Tensor) -> Tensor:
    """
    Converts edge distribution logits into scalar edge offsets.

    The edge offset is defined as the weighted sum of the edge probabilities: Σ P(n) * W(n)

    Args:
        edge_logits: Edge logits with shape (..., 4 * (num_bins + 1)).
        edge_weights: Weighting function with shape (num_bins + 1,).

    Returns:
        edge_offsets: Edge offsets (top, bottom, left, right) with shape (..., 4).
    """

    # Get batch information
    *dims, _ = edge_logits.shape
    device = edge_logits.device

    # Separate the edges into (..., 4, num_bins + 1)
    edge_logits = edge_logits.reshape(*dims, 4, -1)

    # Weighted sum: (..., 4)
    edge_probs = edge_logits.softmax(dim=-1)
    edge_offsets = (edge_probs * edge_weights.to(device)).sum(dim=-1)

    return edge_offsets
