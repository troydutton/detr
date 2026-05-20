from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from data import Target


class DiscreteRandomResize:
    """
    Randomly resizes a batch of images to one of the given resolutions.

    Args:
        resolutions: Base image resolution or list of resolutions to randomly choose from.
    """

    def __init__(self, resolutions: Union[int, List[int]]) -> None:
        self.resolutions = resolutions if isinstance(resolutions, Iterable) else [resolutions]

    def __call__(self, images: Tensor, targets: List[Target]) -> Tuple[Tensor, List[Target]]:
        # Randomly choose a resolution to resize to
        resolution = random.choice(self.resolutions)

        # No-op if the images are already at the target resolution
        _, _, height, width = images.shape

        if height == resolution and width == resolution:
            return images, targets

        # Resize the images and update the target sizes accordingly
        images = F.interpolate(images, size=(resolution, resolution), mode="bilinear")
        device = images.device
        for target in targets:
            target["size"] = torch.tensor([resolution, resolution], dtype=torch.int64, device=device)

        return images, targets
