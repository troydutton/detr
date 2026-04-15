import logging
from typing import Any, Dict

from models import DETR

EXCLUDE_BACKBONE_LAYERS = ["embeddings.position_embeddings", "embeddings.patch_embeddings.projection", "backbone.projector"]


def build_parameter_groups(
    model: DETR,
    lr: float,
    lr_backbone: float,
) -> Dict[str, Any]:
    """
    Build parameter groups for the optimizer with a separate learning rate for the backbone.

    The backbone embedding layers are kept at the default learning rate to allow them to adapt
    to new image & patch sizes. Even though the projector is inside the backbone, it is not
    pretrained and is also kept at the default learning rate.

    Args:
        model: Model with parameters.
        lr: Default learning rate.
        lr_backbone: Backbone learning rate.

    Returns:
        param_groups: Parameter groups for the optimizer.
    """

    logging.info(f"Building parameter groups with {lr=:.1e}, {lr_backbone=:.1e}.")

    # Separate out backbone parameters
    params, backbone_params = [], []
    num_default, num_backbone = 0, 0
    for name, param in model.named_parameters():
        if param.requires_grad == False:
            continue

        if "backbone" in name and not any(l in name for l in EXCLUDE_BACKBONE_LAYERS):
            backbone_params.append(param)
            num_backbone += param.numel()
        else:
            params.append(param)
            num_default += param.numel()

    # Report number of parameters in millions
    num_default, num_backbone = num_default / 1e6, num_backbone / 1e6
    num_total = num_default + num_backbone
    logging.info(f"{num_total:.1f}M (default: {num_default:.1f}M, backbone: {num_backbone:.1f}M).")

    # Create parameter groups
    param_groups = [
        {"name": "default", "params": params, "lr": lr},
        {"name": "backbone", "params": backbone_params, "lr": lr_backbone},
    ]

    return param_groups
