from __future__ import annotations

import random
from typing import TYPE_CHECKING, List, Tuple

import torch
from torch import Tensor
from torchvision.ops.boxes import box_area, box_convert

from utils.boxes import box_intersection

from .constants import LABEL_KEYS

if TYPE_CHECKING:
    from data import Target


class ObjectCopy:
    """
    Batch-level augmentation that copies objects from one image to another.

    Args:
        num_copies: Maximum number of objects to copy into each image.
        alpha: The alpha parameter for the Beta distribution used in patch blending.
        beta: The beta parameter for the Beta distribution used in patch blending.
        overlap_threshold: Area overlap ratio threshold for keeping secondary objects.
    """

    def __init__(
        self,
        num_copies: int = 3,
        alpha: float = 100,
        beta: float = 25,
        overlap_threshold: float = 0.05,
        min_area: float = 100.0,
    ) -> None:
        self.num_copies = num_copies
        self.blend_distribution = torch.distributions.Beta(alpha, beta)
        self.overlap_threshold = overlap_threshold
        self.min_area = min_area

    def __call__(self, images: Tensor, targets: List[Target]) -> Tuple[Tensor, List[Target]]:
        """
        Copies objects into other images within the batch.

        Args:
            images: Images with shape (batch_size, 3, height, width).
            targets: A list of targets corresponding to each image in the batch.

        Returns:
            images: A batch of updated images with shape (batch_size, 3, height, width).
            targets: A list of updated targets corresponding to the modified images.
        """

        # Get batch information
        batch_size = len(images)

        # Pool objects across the batch, keeping track of which image they came from
        boxes = box_convert(torch.cat([t["boxes"] for t in targets]), "cxcywh", "xywh")
        indices = torch.cat([torch.full((len(t["boxes"]),), i, dtype=torch.int64) for i, t in enumerate(targets)])
        labels = {k: torch.cat([t[k] for t in targets]) for k in LABEL_KEYS if k != "boxes"}

        # Remove small objects
        _, _, height, width = images.shape
        keep = (boxes[:, 2] * width) * (boxes[:, 3] * height) >= self.min_area

        boxes = boxes[keep]
        indices = indices[keep]
        labels = {k: v[keep] for k, v in labels.items()}

        # We can't copy objects if there are none
        if (num_objects := len(boxes)) == 0:
            return images, targets

        original_images = images.clone()

        for i in range(batch_size):
            num_copies = random.randint(1, min(self.num_copies, num_objects))
            copy_indices = random.sample(range(num_objects), num_copies)

            for copy_index in copy_indices:
                # Get the patch to copy
                x, y, w, h = boxes[copy_index]

                # Select a location to copy the object to (that doesn't go out of bounds)
                target_x = random.uniform(0, 1 - w)
                target_y = random.uniform(0, 1 - h)

                offsets = torch.tensor([target_x - x, target_y - y, target_x - x, target_y - y])

                # Get the images and their dimensions
                source_index = indices[copy_index]
                source_image = original_images[source_index]
                _, image_height, image_width = source_image.shape
                target_image = images[i]

                # Scale the patch coordinates to the image dimensions
                source_x, source_y = int(x * image_width), int(y * image_height)
                target_x, target_y = int(target_x * image_width), int(target_y * image_height)
                patch_w, patch_h = int(w * image_width), int(h * image_height)

                # Get the patches in the source and target images
                source_patch = source_image[:, source_y : source_y + patch_h, source_x : source_x + patch_w]
                target_patch = target_image[:, target_y : target_y + patch_h, target_x : target_x + patch_w]

                # Blend the source patch into the target patch to create a smoother effect
                blending_factor = self.blend_distribution.sample((1,))
                blended_patch = blending_factor * source_patch + (1 - blending_factor) * target_patch
                target_image[:, target_y : target_y + patch_h, target_x : target_x + patch_w] = blended_patch

                # Update the target annotations with the new object(s)
                source_boxes = boxes[indices == source_index]
                source_labels = {k: v[indices == source_index] for k, v in labels.items()}

                # Copy over any objects that significantly overlap with the source patch
                source_boxes = box_convert(source_boxes, "xywh", "xyxy")
                patch = torch.tensor([[x, y, x + w, y + h]])

                overlap = box_intersection(source_boxes, patch).squeeze(1)
                overlap_ratio = overlap / (box_area(source_boxes) + 1e-4)

                keep = overlap_ratio > self.overlap_threshold
                source_boxes = source_boxes[keep]
                source_labels = {k: v[keep] for k, v in source_labels.items()}

                # Clip boxes to the boundaries of the source patch
                source_boxes[:, :2] = torch.max(source_boxes[:, :2], patch[:, :2])
                source_boxes[:, 2:] = torch.min(source_boxes[:, 2:], patch[:, 2:])

                # Remove degenerate boxes after clipping
                keep = (source_boxes[:, 2] > source_boxes[:, 0]) & (source_boxes[:, 3] > source_boxes[:, 1])
                source_boxes = source_boxes[keep]
                source_labels = {k: v[keep] for k, v in source_labels.items()}

                # Map the source boxes to the new location in the target image
                source_boxes += offsets
                source_boxes = box_convert(source_boxes, "xyxy", "cxcywh")

                # Append the new boxes and labels to the target annotations
                targets[i]["boxes"] = torch.cat([targets[i]["boxes"], source_boxes])

                for k in LABEL_KEYS:
                    if k != "boxes":
                        targets[i][k] = torch.cat([targets[i][k], source_labels[k]])

        return images, targets
