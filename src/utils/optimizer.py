import logging
from typing import Any, Dict

from models import Model


def get_parameter_groups(model: Model, lr: float, lr_backbone: float) -> Dict[str, Any]:
    """
    Build parameter groups for the optimizer with a separate learning rate for the backbone.

    Args:
        model: Model with parameters.
        lr: Default learning rate.
        lr_backbone: Backbone learning rate.

    Returns:
        param_groups: Parameter groups for the optimizer.
    """

    logging.info(f"Building parameter groups with {lr = } and {lr_backbone = }")

    # Separate out backbone parameters
    params, backbone_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            continue

        if "backbone" in name and "projection" not in name:
            backbone_params.append(param)
        else:
            params.append(param)

    # Create parameter groups
    param_groups = [
        {"name": "default", "params": params, "lr": lr},
        {"name": "backbone", "params": backbone_params, "lr": lr_backbone},
    ]

    return param_groups
