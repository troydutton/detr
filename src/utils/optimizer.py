import logging
from typing import Any, Dict, Optional

from models import DETR

PROJECTION_LAYERS = ["reference_points", "sampling_offsets"]


def build_parameter_groups(
    model: DETR,
    lr: float,
    lr_backbone: float,
    lr_projection: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build parameter groups for the optimizer with a separate learning rate for the backbone.

    Args:
        model: Model with parameters.
        lr: Default learning rate.
        lr_backbone: Backbone learning rate.
        lr_projection: Projection head learning rate, optional.

    Returns:
        param_groups: Parameter groups for the optimizer.
    """

    logging.info(f"Building parameter groups with {lr=:.1e}, {lr_backbone=:.1e}, {lr_projection=:.1e}.")

    # Separate out backbone parameters
    params, backbone_params, projection_params = [], [], []
    num_default, num_backbone, num_projection = 0, 0, 0
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            continue

        if "backbone" in name and "projection" not in name:
            backbone_params.append(param)
            num_backbone += param.numel()
        elif any(projection_layer in name for projection_layer in PROJECTION_LAYERS) and lr_projection is not None:
            projection_params.append(param)
            num_projection += param.numel()
        else:
            params.append(param)
            num_default += param.numel()

    # Report number of parameters in millions
    num_default, num_backbone, num_projection = num_default / 1e6, num_backbone / 1e6, num_projection / 1e6
    num_total = num_default + num_backbone + num_projection
    logging.info(f"{num_total:.1f}M (default: {num_default:.1f}M, backbone: {num_backbone:.1f}M, projection: {num_projection:.1f}M).")

    # Create parameter groups
    param_groups = [
        {"name": "default", "params": params, "lr": lr},
        {"name": "backbone", "params": backbone_params, "lr": lr_backbone},
    ]

    if projection_params:
        param_groups.append({"name": "projection", "params": projection_params, "lr": lr_projection})

    return param_groups
