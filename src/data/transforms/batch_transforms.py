from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Union

import torch
from torch import Tensor
from torchvision.tv_tensors import Image

from .image_blend import ImageBlend
from .object_copy import ObjectCopy
from .random_resize import DiscreteRandomResize

if TYPE_CHECKING:
    from data import Target


class BatchTransforms:
    """
    Applies a sequence of batch-level transformations to images and their corresponding annotations during training.

    Specifically, applies multi-scale resizing, image blending, and object copying with a configurable schedule defined below.

                    0               [W]                         [N - C]         [N - F]             [N]
                    |----------------|-----------------------------|---------------|-----------------|
    Blend + Copy    |    [ OFF ]     |           [ ON ]            |    [ OFF ]    |     [ OFF ]     |
                    |................|.............................|...............|.................|
    Resizing        |    [ ON ]      |           [ ON ]            |    [ ON ]     |     [ OFF ]     |
                    '----------------'-----------------------------'---------------'-----------------'

    Args:
        resolution: Base image resolution or list of resolutions to randomly resize to.
        num_epochs: Total number of epochs in training.
        num_warmup_epochs: Number of epochs at the start of training to skip heavy augmentations.
        num_cooldown_epochs: Number of epochs at the end of training to skip heavy augmentations.
        num_finetune_epochs: Number of epochs at the end of training to skip all augmentations and multi-scale resizing.
        blend_probability: Probability of applying image blending when heavy augmentations are enabled.
        copy_probability: Probability of applying object copy when heavy augmentations are enabled.
    """

    def __init__(
        self,
        resolution: Union[int, List[int]],
        num_epochs: int,
        num_warmup_epochs: int = 0,
        num_cooldown_epochs: int = 0,
        num_finetune_epochs: int = 0,
        blend_probability: float = 0.25,
        copy_probability: float = 0.5,
    ) -> None:

        assert blend_probability + copy_probability <= 1.0, "Blend and copy are mutually exclusive, P(Blend) + P(Copy) <= 1."
        assert num_cooldown_epochs >= num_finetune_epochs, "Cooldown period must be at least as long as the fine-tuning period."
        assert num_warmup_epochs + num_cooldown_epochs <= num_epochs, "Warmup and cooldown periods cannot exceed total number of epochs."

        self.num_epochs = num_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.num_cooldown_epochs = num_cooldown_epochs
        self.num_finetune_epochs = num_finetune_epochs
        self.blend_probability = blend_probability
        self.copy_probability = copy_probability

        self.resize = DiscreteRandomResize(resolution)
        self.image_blend = ImageBlend()
        self.object_copy = ObjectCopy()

    def __call__(self, batch: List[Tuple[Image, Target]]) -> Tuple[Tensor, List[Target]]:
        # Unzip the batch
        images, annotations = zip(*batch)
        images, annotations = torch.stack(images), list(annotations)

        # Apply heavy augmentations during the primary training phase (after warmup, before cooldown)
        epoch = annotations[0]["epoch"]
        if epoch >= self.num_warmup_epochs and epoch < self.num_epochs - self.num_cooldown_epochs:
            # Apply blending and object copying (mutually exclusive)
            uniform_sample = torch.rand(1)
            if uniform_sample < self.blend_probability:
                images, annotations = self.image_blend(images, annotations)
            elif uniform_sample < self.blend_probability + self.copy_probability:
                images, annotations = self.object_copy(images, annotations)

        # Apply multi-scale resizing, except during the finetuning phase
        if epoch < self.num_epochs - self.num_finetune_epochs:
            images, annotations = self.resize(images, annotations)

        return images, annotations
