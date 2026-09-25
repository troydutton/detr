from typing import List

import torch
from torch import Tensor

from .muon import Muon, ParamGroup, State


class MuonH(Muon):
    """
    Muon with Hyperball projection on weight matrices, see https://arxiv.org/abs/2606.16899.

    Weight matrices are kept at their initial Frobenius norm and stepped by a fixed fraction `lr` of that norm along
    the orthogonalized update, so weight decay only applies to the Adam groups.

    Args:
        params: Parameters or parameter groups to optimize.
        lr: Angular step size for weight matrices and learning rate for Adam groups.
        momentum: Momentum coefficient for the orthogonalized update.
        nesterov: Whether to use Nesterov momentum.
        ns_steps: Number of Newton-Schulz iterations used to orthogonalize the update.
        betas: Adam coefficients for the running averages of the gradient and its square.
        eps: Adam term added to the denominator for numerical stability.
        weight_decay: Decoupled weight decay factor for Adam groups.
    """

    def _muon_step(self, group: ParamGroup, params: List[Tensor], grads: List[Tensor], states: List[State]) -> None:
        directions = self._muon_direction(group, params, grads, states)

        for param, state in zip(params, states):
            if "radius" not in state:
                state["radius"] = param.norm()
        radii = [state["radius"] for state in states]

        # Step by a fixed fraction of the radius along the update direction
        direction_norms = torch._foreach_norm(directions)
        torch._foreach_add_(direction_norms, group["eps"])
        scales = torch._foreach_div(radii, direction_norms)
        torch._foreach_mul_(scales, group["lr"])
        torch._foreach_mul_(directions, scales)
        torch._foreach_sub_(params, directions)

        # Project back onto the hypersphere
        torch._foreach_mul_(params, torch._foreach_div(radii, torch._foreach_norm(params)))
