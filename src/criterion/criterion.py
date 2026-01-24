from torch import nn, Tensor
from typing import Dict, List, Tuple
from utils.misc import take_annotation_from


class SetCriterion(nn.Module):
    """ """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """ """
        pass

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
