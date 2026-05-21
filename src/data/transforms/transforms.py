from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, List, Optional, Protocol, Tuple, Union

import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch import Tensor
from torchvision.tv_tensors import BoundingBoxes

from .constants import IMNET_MEAN, IMNET_STD, LABEL_KEYS
from .mosaic import Mosaic
from .random_erasing import RandomErasingBoxAware

if TYPE_CHECKING:
    from data.coco_dataset import Target


class Transformation(Protocol):
    def __call__(self, image: Image.Image, annotations: Optional[Target] = None) -> Tuple[Tensor, Target] | Tensor: ...


def make_transformations(
    split: str,
    resolution: Union[int, List[int]],
    num_epochs: int = 0,
    num_warmup_epochs: int = 0,
    num_cooldown_epochs: int = 0,
    *,
    normalize: bool = True,
) -> Transformation:
    """
    Create a transformation pipeline.

    Args:
        split: Which split to build transforms for. ("train", "val", or "test")
        resolution: Base image resolution used to compute resize sizes.
        num_epochs: Total number of epochs, optional.
        num_warmup_epochs: Number of epochs to skip heavy augmentations for at the start of training, optional.
        num_cooldown_epochs: Number of epochs to skip heavy augmentations for at the end of training, optional.
        normalize: Whether to normalize the image, optional.

    Returns:
        transformations: A callable that accepts an (image, annotations) pair, and returns the transformed pair.
    """

    logging.info(f"Building '{split}' transformations with {resolution=}.")

    # When provided multiple resolutions we resize to the largest resolution in the transformations,
    # and randomly resize images at the batch level to avoid the need for padding
    resolution = max(resolution) if isinstance(resolution, Iterable) else resolution

    # Sometimes we want to skip normalizaition (i.e. for visualization)
    normalize_transform = make_normalize_transform(normalize)

    def _get_labels(sample: Tuple[Image.Image, Target]) -> List[Union[Tensor, BoundingBoxes]]:
        return [v for k, v in sample[1].items() if k in LABEL_KEYS]

    if split == "train":
        return T.Compose(
            [
                T.ToImage(),
                T.RandomApply([T.RandomIoUCrop()], p=0.8),
                T.Resize((resolution, resolution)),
                T.RandomHorizontalFlip(),
                RandomErasingBoxAware(p=0.33, value="random"),
                T.RandomPhotometricDistort(),
                Mosaic(num_epochs=num_epochs, num_warmup_epochs=num_warmup_epochs, num_cooldown_epochs=num_cooldown_epochs),
                T.RandomApply([T.RandomZoomOut(side_range=(1.0, 2.0), p=1.0), T.Resize((resolution, resolution))], p=0.5),
                T.ToDtype(torch.float32, scale=True),
                normalize_transform,
                T.ClampBoundingBoxes(),
                T.SanitizeBoundingBoxes(labels_getter=_get_labels),
            ]
        )

    if split == "finetune":
        return T.Compose(
            [
                T.ToImage(),
                T.Resize(size=(resolution, resolution)),
                T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32, scale=True),
                normalize_transform,
                T.ClampBoundingBoxes(),
                T.SanitizeBoundingBoxes(labels_getter=_get_labels),
            ]
        )

    if split == "val":
        return T.Compose(
            [
                T.ToImage(),
                T.Resize(size=(resolution, resolution)),
                T.ToDtype(torch.float32, scale=True),
                normalize_transform,
                T.ClampBoundingBoxes(),
                T.SanitizeBoundingBoxes(labels_getter=_get_labels),
            ]
        )

    if split == "test":
        return T.Compose(
            [
                T.ToImage(),
                T.Resize(size=(resolution, resolution)),
                T.ToDtype(torch.float32, scale=True),
                normalize_transform,
            ]
        )

    raise ValueError(f"Invalid transformation split: {split}")


def make_normalize_transform(normalize: bool = True) -> Transformation:
    """
    Create a normalization transform with ImageNet mean and std.

    Separated because sometimes we want to delay normalization so that we can visualize
    the images after other transformations have been applied (i.e. during inference).

    Args:
        normalize: Whether to normalize the image, optional.

    Returns:
        normalize_transform: Normalization transform.
    """

    return T.Normalize(IMNET_MEAN, IMNET_STD) if normalize else T.Identity()
