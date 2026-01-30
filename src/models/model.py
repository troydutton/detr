from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, nn

from utils.misc import take_annotation_from

Predictions = Dict[str, Tensor]


class Model(ABC, nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, Tensor]: ...

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
