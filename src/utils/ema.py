import math
from typing import Callable, List

import torch
from torch import Tensor


def get_warmup_ema_multi_avg_fn(decay: float, warmup_steps: int) -> Callable[[List[Tensor], List[Tensor], int], None]:
    """
    Returns a multi-parameter EMA averaging function with exponential warmup.

    Args:
        decay: Maximum decay rate.
        warmup_steps: Number of warmup steps.

    Returns:
        ema_update:
    """

    def ema_update(
        averaged_model_parameters: List[Tensor],
        current_model_parameters: List[Tensor],
        num_steps: int,
    ) -> None:

        d = decay * (1 - math.exp(-num_steps / warmup_steps))

        # Use an efficient in-place operation if supported, otherwise fall back to a loop
        if torch.is_floating_point(averaged_model_parameters[0]) or torch.is_complex(averaged_model_parameters[0]):
            torch._foreach_lerp_(averaged_model_parameters, current_model_parameters, 1 - d)
        else:
            for ema_param, current_param in zip(averaged_model_parameters, current_model_parameters, strict=True):
                ema_param.copy_(d * ema_param + (1 - d) * current_param)

    return ema_update
