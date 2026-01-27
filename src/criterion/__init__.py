from typing import TYPE_CHECKING, Dict, Protocol

from torch import Tensor

from .criterion import SetCriterion

if TYPE_CHECKING:
    from .criterion import Targets


class CriterionType(Protocol):
    def __call__(self, predictions: Dict[str, Tensor], targets: Targets) -> Dict[str, Tensor]: ...


__all__ = [SetCriterion, CriterionType]
