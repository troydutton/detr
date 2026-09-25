from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from optimizers import Muon


def _param_groups(model: nn.Module, muon_lr: float = 1e-2) -> List[Dict]:
    matrices = [param for param in model.parameters() if param.ndim == 2]
    vectors = [param for param in model.parameters() if param.ndim < 2]
    return [
        {"params": matrices, "lr": muon_lr, "weight_decay": 1e-2, "muon": True},
        {"params": vectors, "weight_decay": 1e-2, "muon": False},
    ]


def _step(model: nn.Module, optimizer: torch.optim.Optimizer, num_steps: int, batch_size: int = 8) -> None:
    for _ in range(num_steps):
        optimizer.zero_grad()
        model(torch.randn(batch_size, 16)).square().mean().backward()
        optimizer.step()


class TestMuon:
    """
    Tests for the Muon optimizer.
    """

    def test_muon_update_is_orthogonal(self) -> None:
        """
        The update applied to a weight matrix has near-uniform singular values.

        The batch must exceed the matrix rank, otherwise the gradient is rank deficient and cannot be orthogonalized.
        """
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 4))
        matrix = model[0].weight
        before = matrix.detach().clone()

        optimizer = Muon(_param_groups(model), lr=1e-2)
        _step(model, optimizer, num_steps=1, batch_size=64)

        singular_values = torch.linalg.svdvals(matrix.detach() - before)
        assert singular_values.min() > 0.5 * singular_values.max()

    def test_adam_groups_match_adamw(self) -> None:
        """
        Parameters outside Muon groups follow the same trajectory as torch AdamW when matrices are frozen.
        """
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 4))
        reference = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 4))
        reference.load_state_dict(model.state_dict())

        optimizer = Muon(_param_groups(model, muon_lr=0.0), lr=1e-2)
        reference_optimizer = torch.optim.AdamW(_param_groups(reference, muon_lr=0.0), lr=1e-2)

        torch.manual_seed(1)
        _step(model, optimizer, num_steps=3)
        torch.manual_seed(1)
        _step(reference, reference_optimizer, num_steps=3)

        for param, reference_param in zip(model.parameters(), reference.parameters()):
            if param.ndim < 2:
                assert torch.allclose(param, reference_param, atol=1e-6)

    def test_state_dict_round_trip(self) -> None:
        """
        Momentum buffers survive a state dict round trip.
        """
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(16, 32), nn.LayerNorm(32), nn.Linear(32, 4))
        optimizer = Muon(_param_groups(model), lr=1e-2)
        _step(model, optimizer, num_steps=2)

        restored = Muon(_param_groups(model), lr=1e-2)
        restored.load_state_dict(optimizer.state_dict())

        for param in model.parameters():
            if param.ndim == 2:
                assert torch.equal(restored.state[param]["momentum_buffer"], optimizer.state[param]["momentum_buffer"])
