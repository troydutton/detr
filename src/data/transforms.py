from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from typing import TYPE_CHECKING, List, Optional, Protocol, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image
from torch import Tensor
from torchvision.ops.boxes import box_area, box_convert
from torchvision.transforms.v2.functional import erase
from torchvision.tv_tensors import BoundingBoxes, wrap

from utils.boxes import box_intersection

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]
LABEL_KEYS = ["boxes", "labels", "area", "iscrowd"]

if TYPE_CHECKING:
    from data import Target


class Transformation(Protocol):
    def __call__(self, image: Image.Image, annotations: Optional[Target] = None) -> Tuple[Tensor, Target] | Tensor: ...


class RandomErasingBoxAware:
    """
    Box aware version of `torchvision.transforms.v2.RandomErasing`.

    Randomly erases a region from an image. Removes bounding boxes that significantly overlap
    the erased region, and updates partially erased boxes to not include the erased sections.
    Note that if the erased region overlaps only the corner of a box, the box is not shrunk
    because it still needs to cover the full extent of the visible object.

    Args:
        p: Probability that the random erasing operation will be performed.
        scale: Range of proportion of erased area against input image.
        ratio: Range of aspect ratio of erased area.
        value: Value to replace the erased region with.
        erased_threshold: Ratio of of erasure at which to remove a box.
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Sequence[float] = (0.02, 0.33),
        ratio: Sequence[float] = (0.3, 3.3),
        value: float = 0.0,
        erased_threshold: float = 0.95,
    ) -> None:
        self.p = p
        self.erased_threshold = erased_threshold
        self.random_erasing = T.RandomErasing(p=1.0, scale=scale, ratio=ratio, value=value)

    def __call__(self, image: Tensor, annotations: Target) -> Tuple[Tensor, Target]:
        # Skip with probability 1 - p
        if torch.rand(1) > self.p:
            return image, annotations

        # Get the region to erase, returning early if we fail to find one
        params = self.random_erasing.make_params([image])

        if params["v"] is None:
            return image, annotations

        # Erase the region from the image
        image = erase(image, **params)

        boxes = annotations["boxes"]

        if len(boxes) == 0:
            return image, annotations

        # Working in xyxy makes the calculations easier
        box_format = boxes.format.name.lower()
        boxes = box_convert(boxes, box_format, "xyxy")

        x, y, w, h = params["j"], params["i"], params["w"], params["h"]
        patch = torch.tensor([x, y, x + w, y + h])

        # Remove any boxes whose ratio of erased area exceeds the threshold
        erased_area = box_intersection(boxes, patch.unsqueeze(0)).squeeze(1)
        erased_ratio = erased_area / box_area(boxes)
        keep = erased_ratio < self.erased_threshold

        # Shrink remaining boxes if they are partially erased
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        rx1, ry1, rx2, ry2 = patch.unbind(dim=0)

        # Left side of the box overlaps with the erased region
        left_overlap = (x1 >= rx1) & (x1 < rx2) & (y1 >= ry1) & (y2 <= ry2)
        # Right side of the box overlaps with the erased region
        right_overlap = (x2 > rx1) & (x2 <= rx2) & (y1 >= ry1) & (y2 <= ry2)
        # Top of the box overlaps with the erased region
        top_overlap = (y1 >= ry1) & (y1 < ry2) & (x1 >= rx1) & (x2 <= rx2)
        # Bottom of the box overlaps with the erased region
        bottom_overlap = (y2 > ry1) & (y2 <= ry2) & (x1 >= rx1) & (x2 <= rx2)

        # Calculate new boxes
        x1_new = torch.where(left_overlap, torch.minimum(rx2, x2), x1)
        x2_new = torch.where(right_overlap, torch.maximum(rx1, x1), x2)
        y1_new = torch.where(top_overlap, torch.minimum(ry2, y2), y1)
        y2_new = torch.where(bottom_overlap, torch.maximum(ry1, y1), y2)
        boxes = torch.stack([x1_new, y1_new, x2_new, y2_new], dim=1)

        # Remove degenerate boxes
        keep = keep & (x2_new > x1_new) & (y2_new > y1_new)
        boxes = boxes[keep]

        for k, v in annotations.items():
            if k in LABEL_KEYS and k != "boxes":
                annotations[k] = v[keep]

        # Restore boxes to their original format
        boxes = box_convert(boxes, "xyxy", box_format)
        annotations["boxes"] = wrap(boxes, like=annotations["boxes"])

        return image, annotations


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


def make_transformations(
    split: str,
    resolution: Union[int, List[int]],
    *,
    normalize: bool = True,
) -> Transformation:
    """
    Create a transform pipeline.

    Args:
        split: Which split to build transforms for. ("train", "val", or "test")
        resolution: Base image resolution used to compute resize sizes.
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
                T.RandomIoUCrop(),
                T.Resize(size=(resolution, resolution)),
                T.RandomHorizontalFlip(),
                T.ToDtype(torch.float32, scale=True),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                RandomErasingBoxAware(p=0.33, value="random"),
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
