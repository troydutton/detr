import logging
from typing import Any, Dict

from models import Model


def build_parameter_groups(model: Model, lr: float, lr_backbone: float) -> Dict[str, Any]:
    """
    Build parameter groups for the optimizer with a separate learning rate for the backbone.

    Args:
        model: Model with parameters.
        lr: Default learning rate.
        lr_backbone: Backbone learning rate.

    Returns:
        param_groups: Parameter groups for the optimizer.
    """

    logging.info(f"Building parameter groups with {lr=:.1e} and {lr_backbone=:.1e}.")

    # Separate out backbone parameters
    params, backbone_params = [], []
    num_params, num_backbone_params = 0, 0
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            continue

        if "backbone" in name and "projection" not in name:
            backbone_params.append(param)
            num_backbone_params += param.numel()
        else:
            params.append(param)
            num_params += param.numel()

    logging.info(f"Number of trainable parameters: {num_params / 1e6:.1f}M (default), {num_backbone_params / 1e6:.1f}M (backbone).")

    # Create parameter groups
    param_groups = [
        {"name": "default", "params": params, "lr": lr},
        {"name": "backbone", "params": backbone_params, "lr": lr_backbone},
    ]

    return param_groups
