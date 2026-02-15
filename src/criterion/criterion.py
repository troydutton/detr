from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List

from torch import Tensor

if TYPE_CHECKING:
    from data import Target
    from models import ModelPredictions


class Criterion(ABC):
    @abstractmethod
    def __call__(self, predictions: ModelPredictions, targets: List[Target]) -> Dict[str, Tensor]: ...
