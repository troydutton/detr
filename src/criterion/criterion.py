from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

from torch import Tensor

if TYPE_CHECKING:
    from data import Target
    from models import Predictions


class Criterion(ABC):
    @abstractmethod
    def __call__(self, predictions: Predictions, targets: List[Target]) -> Dict[str, Tensor]: ...
