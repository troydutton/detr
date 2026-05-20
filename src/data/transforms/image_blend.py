from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple

import torch
from torch import Tensor

from .constants import LABEL_KEYS

if TYPE_CHECKING:
    from data import Target


class ImageBlend:
    """
    A simple image blending augmentation that mixes a batch of images and their labels together (a.k.a MixUp).

    Specifically, we sample a blending factor from a Beta distribution for each image,
    and blend each image with another randomly selected image in the batch using that factor.

    Args:
        alpha: The alpha parameter for the Beta distribution used in image blending.
    """

    def __init__(self, alpha: float = 150):
        self.alpha = alpha
        self.blend_distribution = torch.distributions.Beta(alpha, alpha)

    def __call__(self, images: Tensor, annotations: List[Target]) -> Tuple[Tensor, List[Target]]:
        """
        Blend a batch of images and their corresponding targets together.

        Args:
            images: A batch of images wth shape (batch_size, 3, height, width).
            annotations: A list of annotations corresponding to each image in the batch.

        Returns:
            images: A batch of blended images with shape (batch_size, 3, height, width).
            annotations: A list of updated annotations corresponding to the blended images.
        """

        # Get batch information
        batch_size = len(images)

        # Sample a blending factor for each image in the batch
        blending_factor = self.blend_distribution.sample((batch_size, 1, 1, 1))

        # Randomly permute the batch to create pairs of images to blend
        blend_indices = torch.randperm(batch_size)
        images = blending_factor * images + (1 - blending_factor) * images[blend_indices]

        # Update annotations by concatenating the labels of the blended images
        blend_annotations = deepcopy([annotations[i] for i in blend_indices])
        for image_annotations, blend_image_annotations in zip(annotations, blend_annotations):
            for k in LABEL_KEYS:
                image_annotations[k] = torch.cat([image_annotations[k], blend_image_annotations[k]], dim=0)

        return images, annotations
