from __future__ import annotations

from abc import ABC, abstractmethod
from ast import Tuple
from typing import TYPE_CHECKING, Dict, List, Optional

from torch import Tensor

if TYPE_CHECKING:
    from data import Target
    from models.detr import Predictions


class Criterion(ABC):
    @abstractmethod
    def __call__(self, predictions: Tuple[Predictions, Optional[Predictions]], targets: List[Target]) -> Dict[str, Tensor]: ...
