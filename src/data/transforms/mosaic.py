from __future__ import annotations

import collections
import copy
import random
from typing import TYPE_CHECKING, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops.boxes import box_convert
from torchvision.tv_tensors import wrap

from .constants import LABEL_KEYS

if TYPE_CHECKING:
    from data.coco_dataset import Target


class Mosaic:
    """
    Mosaic data augmentation that tiles nxn images and their labels together.

    Args:
        p: The probability of applying mosaic augmentation to each image.
        num_tiles: The number of tiles on each side of the mosaic (e.g. 2 creates a 2x2 mosaic).
        num_epochs: Total number of epochs in training.
        num_warmup_epochs: Number of epochs at the start of training to skip the transformation.
        num_cooldown_epochs: Number of epochs at the end of training to skip the transformation.
        max_cache_size: Maximum number of images to keep in the cache.
    """

    def __init__(
        self,
        p: float = 0.5,
        num_tiles: int = 2,
        num_epochs: int = 0,
        num_warmup_epochs: int = 0,
        num_cooldown_epochs: int = 0,
        max_cache_size: int = 160,
    ) -> None:

        self.p = p
        self.num_tiles = num_tiles
        self.num_epochs = num_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.num_cooldown_epochs = num_cooldown_epochs
        self.cache = collections.deque(maxlen=max_cache_size)

    def __call__(self, image: Tensor, annotations: Target) -> Tuple[Tensor, Target]:
        """
        Applies mosaic augmentation to an image.

        Args:
            image: Image with shape (3, height, width).
            annotations: Image information and object annotations.

        Returns:
            image: Mosaic image with shape (3, height, width).
            annotations: Image information and mosaic object annotations.
        """

        # Store the image and target in the cache for use in future mosaics
        image_clone, annotations_clone = self._clone_sample((image, annotations))
        self.cache.append((image_clone, annotations_clone))

        # Skip during warmup or cooldown periods
        epoch = annotations["epoch"]
        if epoch < self.num_warmup_epochs or epoch >= self.num_epochs - self.num_cooldown_epochs:
            return image, annotations

        # Skip if the cache doesn't have enough images to build a mosaic
        if len(self.cache) < self.num_tiles**2 - 1:
            return image, annotations

        # Skip w.p. 1 - p when active
        if torch.rand(1).item() > self.p:
            return image, annotations

        # Get batch information
        _, height, width = image.shape

        # Calculate tile size and offsets for placing images in the mosaic
        tile_height, tile_width = height // self.num_tiles, width // self.num_tiles
        tile_offsets = [(x * tile_width, y * tile_height) for x in range(self.num_tiles) for y in range(self.num_tiles)]

        # Blank canvas to build the mosaic
        canvas = torch.zeros_like(image)
        canvas_annotations = {k: v if k not in LABEL_KEYS else [] for k, v in annotations.items()}

        # Sample extra images from the cache (cloning to avoid mutating cached data)
        extra_samples: List[Tuple[Tensor, Target]] = random.sample(self.cache, self.num_tiles**2 - 1)
        extra_samples = [self._clone_sample(sample) for sample in extra_samples]

        all_images = torch.stack([image] + [extra_image for extra_image, _ in extra_samples])
        all_annotations = [annotations] + [extra_annotations for _, extra_annotations in extra_samples]

        # Resize images to fit in the mosaic tiles
        all_images = F.interpolate(all_images, size=(tile_height, tile_width), mode="bilinear")

        for tile_image, tile_annotations, (offset_x, offset_y) in zip(all_images, all_annotations, tile_offsets):
            # Paste the tile image onto the canvas
            canvas[:, offset_y : offset_y + tile_height, offset_x : offset_x + tile_width] = tile_image

            # Adjust bounding boxes
            boxes = tile_annotations["boxes"] / self.num_tiles
            boxes[:, 0] += offset_x
            boxes[:, 1] += offset_y

            # Clamp boxes (removing degenerate boxes in the process)
            box_format = tile_annotations["boxes"].format.name.lower()
            boxes = box_convert(boxes, box_format, "xyxy")
            boxes[:, 0] = boxes[:, 0].clamp(offset_x, offset_x + tile_width)
            boxes[:, 1] = boxes[:, 1].clamp(offset_y, offset_y + tile_height)
            boxes[:, 2] = boxes[:, 2].clamp(offset_x, offset_x + tile_width)
            boxes[:, 3] = boxes[:, 3].clamp(offset_y, offset_y + tile_height)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            tile_annotations["boxes"] = box_convert(boxes, "xyxy", box_format)

            for k in LABEL_KEYS:
                if k in tile_annotations:
                    canvas_annotations[k].append(tile_annotations[k][keep])

        for k in LABEL_KEYS:
            if len(canvas_annotations[k]) > 0:
                canvas_annotations[k] = torch.cat(canvas_annotations[k], dim=0)
            else:
                canvas_annotations[k] = annotations[k]

        canvas_annotations["boxes"] = wrap(canvas_annotations["boxes"], like=annotations_clone["boxes"])

        return canvas, canvas_annotations

    def _clone_sample(self, sample: Tuple[Tensor, Target]) -> Tuple[Tensor, Target]:
        """
        Creates a deep copy of an image and its annotations.

        Args:
            sample: A tuple containing an image and its corresponding annotations.

        Returns:
            A tuple containing the cloned image and annotations.
        """
        image, annotations = sample

        cloned_image = image.clone()
        cloned_annotations = {k: v.clone() if isinstance(v, Tensor) else copy.deepcopy(v) for k, v in annotations.items()}

        return cloned_image, cloned_annotations
