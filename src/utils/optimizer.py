import logging
import re
from typing import Dict, List

from torch import Tensor

from models import DETR

# We exclude the projector from
EXCLUDE_BACKBONE_LAYERS = ["backbone.projector"]

# We don't apply weight decay to embedding parameters because they act
# as learned priors rather than transformations that we need to regularize.
EMBEDDING_KEYS = [
    # ViT Tokens
    "embeddings.cls_token",
    "embeddings.register_tokens",
    # Positional embeddings
    "embeddings.position_embeddings",
    "backbone.level_pos",
    # Query embeddings
    "decoder.queries",
    # Denosing embeddings
    "decoder.denoise_embed",
    "decoder.object_embed",
    "decoder.label_embed",
]


def build_parameter_groups(
    model: DETR,
    lr: float,
    lr_backbone: float,
    weight_decay: float,
    backbone_layer_decay: float,
    num_backbone_layers: int,
) -> List[Dict[str, str | float | List[Tensor]]]:
    """
    Build parameter groups for the optimizer with a separate learning rate for the backbone.

    The backbone embedding layers are kept at the default learning rate to allow them to adapt
    to new image & patch sizes. Even though the projector is inside the backbone, it is not
    pretrained and is also kept at the default learning rate.

    Args:
        model: Model with parameters.
        lr: Default learning rate.
        lr_backbone: Backbone learning rate.
        weight_decay: L2 weight decay factor.
        backbone_layer_decay: Multiplier for layer-wise learning rate decay.
        num_backbone_layers: Total number of layers in the backbone.

    Returns:
        param_groups: Parameter groups for the optimizer.
    """

    logging.info(f"Building parameter groups with {lr=:.1e}, {lr_backbone=:.1e}, {backbone_layer_decay=:.2f}, {weight_decay=:.1e}.")

    param_groups: Dict[str, Dict[str, str | float | List[Tensor]]] = {}
    num_default, num_backbone = 0, 0

    for name, param in model.named_parameters():
        if param.requires_grad == False:
            continue

        # Determine learning rate
        if "backbone" in name and not any(l in name for l in EXCLUDE_BACKBONE_LAYERS):
            layer_id = _get_backbone_layer_id(name, num_backbone_layers)

            group_name = f"backbone.{layer_id}"
            group_lr = lr_backbone * (backbone_layer_decay ** (num_backbone_layers + 1 - layer_id))
            num_backbone += param.numel()
        else:
            group_name = "default"
            group_lr = lr
            num_default += param.numel()

        # Determine weight decay
        if _should_apply_weight_decay(name, param):
            group_weight_decay = weight_decay
        else:
            group_name += ".no_decay"
            group_weight_decay = 0.0

        if group_name not in param_groups:
            param_groups[group_name] = {
                "name": group_name,
                "params": [param],
                "lr": group_lr,
                "weight_decay": group_weight_decay,
            }
        else:
            param_groups[group_name]["params"].append(param)

    # Report number of parameters in millions
    num_default, num_backbone = num_default / 1e6, num_backbone / 1e6
    num_total = num_default + num_backbone
    logging.info(f"Training {num_total:.1f}M parameters (default: {num_default:.1f}M, backbone: {num_backbone:.1f}M).")

    return list(param_groups.values())


def _get_backbone_layer_id(name: str, num_layers: int) -> int:
    """
    Get the backbone layer ID for a given parameter name.

    The layer ID is used to apply layer-wise learning rate decay. We assign layer IDs as follows:
        - Embedding layers: 0
        - Encoder layers: 1 to num_layers
        - Final normalization: num_layers + 1

    Args:
        name: Parameter name.
        num_layers: Total number of layers in the backbone.
    """

    if ".embeddings." in name:
        return 0

    match = re.search(r"encoder\.layer\.(\d+)", name)

    return int(match.group(1)) + 1 if match else num_layers + 1


def _should_apply_weight_decay(name: str, param: Tensor) -> bool:
    """
    Determine whether to apply weight decay to a parameter based on its name.

    We apply weight decay to all parameters except for those that are
        - 1D (like biases or normalization parameters), or
        - explicitly identified as embeddings (like positional embeddings or query embeddings)

    Args:
        name: Parameter name.
        param: Parameter weight.

    Returns:
        should_apply: Whether to apply weight decay.
    """

    return not (param.ndim < 2 or any(k in name for k in EMBEDDING_KEYS))
