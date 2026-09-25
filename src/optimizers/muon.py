import math
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

State = Dict[str, Any]
ParamGroup = Dict[str, Any]

# Quintic Newton-Schulz coefficients from https://github.com/KellerJordan/Muon
NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

# Scales the orthogonalized update to match the RMS of an AdamW update, so that the learning rate
# is comparable to AdamW's. Follows `match_rms_adamw` in torch.optim.Muon.
RMS_MATCH_SCALE = 0.2


class Muon(Optimizer):
    """
    Muon with decoupled weight decay, falling back to Adam for parameters that are not weight matrices.

    Parameter groups flagged with `muon` are updated with an orthogonalized momentum step, scaled per matrix to match
    the update RMS of AdamW. All other groups are updated with decoupled weight decay Adam.

    Args:
        params: Parameters or parameter groups to optimize.
        lr: Learning rate, comparable to AdamW's.
        momentum: Momentum coefficient for the orthogonalized update.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations used to orthogonalize the update.
        betas: Adam coefficients for the running averages of the gradient and its square.
        eps: Adam term added to the denominator for numerical stability.
        weight_decay: Decoupled weight decay factor.
    """

    def __init__(
        self,
        params: Iterable[Tensor | ParamGroup],
        lr: float = 1e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, betas=betas, eps=eps, weight_decay=weight_decay, muon=False))  # fmt: skip

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [param for param in group["params"] if param.grad is not None]
            if not params:
                continue

            grads = [param.grad for param in params]
            states = [self.state[param] for param in params]

            if group["muon"]:
                self._muon_step(group, params, grads, states)
            else:
                self._adam_step(group, params, grads, states)

        return loss

    def _muon_step(self, group: ParamGroup, params: List[Tensor], grads: List[Tensor], states: List[State]) -> None:
        directions = self._muon_direction(group, params, grads, states)
        scales = [group["lr"] * RMS_MATCH_SCALE * math.sqrt(max(param.shape[0], param.numel() // param.shape[0])) for param in params]  # fmt: skip

        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_mul_(directions, scales)
        torch._foreach_sub_(params, directions)

    def _adam_step(self, group: ParamGroup, params: List[Tensor], grads: List[Tensor], states: List[State]) -> None:
        directions = self._adam_direction(group, params, grads, states)

        torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
        torch._foreach_add_(params, directions, alpha=-group["lr"])

    def _muon_direction(self, group: ParamGroup, params: List[Tensor], grads: List[Tensor], states: List[State]) -> List[Tensor]:
        for param, state in zip(params, states):
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(param)
        buffers = [state["momentum_buffer"] for state in states]

        torch._foreach_lerp_(buffers, grads, 1 - group["momentum"])
        updates = torch._foreach_lerp(grads, buffers, group["momentum"]) if group["nesterov"] else buffers

        return _orthogonalize_by_shape(updates, group["ns_steps"])

    def _adam_direction(self, group: ParamGroup, params: List[Tensor], grads: List[Tensor], states: List[State]) -> List[Tensor]:
        beta1, beta2 = group["betas"]

        for param, state in zip(params, states):
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1

        exp_avgs = [state["exp_avg"] for state in states]
        exp_avg_sqs = [state["exp_avg_sq"] for state in states]

        torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1 - beta2)

        bias_corrections1 = [1 - beta1 ** state["step"] for state in states]
        bias_corrections2 = [math.sqrt(1 - beta2 ** state["step"]) for state in states]

        denominators = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_div_(denominators, bias_corrections2)
        torch._foreach_add_(denominators, group["eps"])

        directions = torch._foreach_div(exp_avgs, denominators)
        torch._foreach_div_(directions, bias_corrections1)

        return directions


def _orthogonalize_by_shape(updates: List[Tensor], num_steps: int) -> List[Tensor]:
    """Orthogonalize each update, batching those that share a shape into a single Newton-Schulz call."""

    indices_by_shape: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for index, update in enumerate(updates):
        indices_by_shape[update.flatten(1).shape].append(index)

    directions: List[Tensor] = [None] * len(updates)
    for indices in indices_by_shape.values():
        matrices = torch.stack([updates[index].flatten(1) for index in indices])
        for index, direction in zip(indices, _orthogonalize(matrices, num_steps).unbind()):
            directions[index] = direction.view_as(updates[index])

    return directions


def _orthogonalize(matrices: Tensor, num_steps: int) -> Tensor:
    """Approximate the nearest semi-orthogonal matrices with Newton-Schulz iterations in bfloat16."""

    coef_a, coef_b, coef_c = NEWTON_SCHULZ_COEFFICIENTS

    transposed = matrices.shape[-2] > matrices.shape[-1]
    orthogonal = matrices.mT if transposed else matrices
    orthogonal = orthogonal.bfloat16()
    orthogonal = orthogonal / (orthogonal.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(num_steps):
        gram = orthogonal @ orthogonal.mT
        orthogonal = coef_a * orthogonal + (coef_b * gram + coef_c * gram @ gram) @ orthogonal

    orthogonal = orthogonal.mT if transposed else orthogonal

    return orthogonal.to(matrices.dtype)
