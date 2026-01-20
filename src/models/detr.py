from torch import nn, Tensor
from typing import Any, Dict
from models.backbone import Backbone, build_backbone
from models.transformer import Transformer, build_transformer
from utils.misc import take_annotation_from


class DETR(nn.Module):
    """
    Implementation of the Detection Transformer architecture from ["End-to-End Object Detection with Transformers"](https://arxiv.org/abs/2005.12872).

    Args:
        backbone_args: Arguments to construct the backbone model. See `models.backbone.Backbone`.
        transformer_args: Arguments to construct the transformer model. See `models.transformer.Transformer`.
        num_classes: Number of object classes.
        pretrained_weights: Path to a pretrained weights file.
    """

    def __init__(
        self,
        backbone_args: Dict[str, Any],
        transformer_args: Dict[str, Any],
        num_classes: int,
        pretrained_weights: str = None,
    ) -> None:
        super().__init__()

        self.backbone = build_backbone(**backbone_args)

        self.transformer = build_transformer(**transformer_args)

        self.num_classes = num_classes

    def forward(self, image: Tensor) -> Dict[str, Tensor]:
        pass

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
