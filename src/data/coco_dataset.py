import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import PIL.Image
import torch
from pycocotools.coco import COCO
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.ops.boxes import box_convert, clip_boxes_to_image
from torchvision.transforms.v2.functional import to_dtype, to_image
from torchvision.tv_tensors import BoundingBoxes, Image

from data.transforms import Transformation
from utils.misc import silence_stdout

ImageAnnotations = List[Dict[str, Any]]
Target = Dict[str, Tensor | BoundingBoxes]

# Allow loading large images without hitting decompression bomb errors
PIL.Image.MAX_IMAGE_PIXELS = None


class CocoDataset(Dataset):
    """
    COCO-style object detection dataset.

    Expects the following directory structure:
    root/
    ├── _annotations.coco.json
    ├── images/
    ├──── image0.jpg
    ├──── image1.jpg
    ├──── ...

    Args:
        roots: Root for each dataset.
        transforms: Transformations to apply to each image, optional.
        image_directory: Name of the image directory within each root, optional.
        annotation_name: Name of the annotation file within each root, optional.
    """

    def __init__(
        self,
        roots: str | List[str],
        transforms: Transformation = None,
        image_directory: str = "images",
        annotation_name: str = "_annotations.coco.json",
    ) -> None:
        # We allow multiple roots since we may want to merge multiple datasets
        if isinstance(roots, (str, Path)):
            roots = [roots]

        self.roots = [Path(roots) for roots in roots]
        self.transforms = transforms
        self.image_directory = image_directory
        self.annotation_name = annotation_name

        # Log the roots being loaded, showing up to 3 for brevity
        log_roots = [f"`{str(root)}`" for root in self.roots[:3]]
        if len(self.roots) > 3:
            log_roots.append(f"and {len(self.roots) - 3} other datasets")
        logging.info(f"Loading {', '.join(log_roots)}.")

        # Create a single COCO dataset from the provided roots
        self.coco = self._create_coco_dataset(self.roots)
        self.image_ids = list(self.coco.imgs.keys())

        # Mapping from category ids to contiguous 0-indexed labels
        category_ids = sorted(self.coco.getCatIds())
        self.category_id_to_label = {cat_id: i for i, cat_id in enumerate(category_ids)}
        self.num_classes = len(self.category_id_to_label)

        logging.info(f"Loaded {len(self.image_ids):,} images with {self.num_classes} classes.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[Image, Target]:
        """
        Retrieve an image and its corresponding annotations.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            image: Image tensor with shape (channels, height, width).
            #### target
            Image annotations, including
                - `boxes`: Bounding boxes in CXCYWH format, normalized to [0, 1].
                - `labels`: Class labels (category ids).
                - `area`: Area of each bounding box.
                - `iscrowd`: Crowd indicators.
                - `image_id`: Image identifier.
                - `orig_size`: Original image size (height, width).
                - `size`: Transformed image size (height, width).
        """

        # Retrieve annotations and image information
        image_id = self.image_ids[idx]
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        image_info = self.coco.loadImgs(image_id)[0]

        # Load the image
        image = PIL.Image.open(image_info["file_name"]).convert("RGB")
        w, h = image.size

        # Convert COCO object annotations to our expected format
        target = self._convert_object_annotations(annotations, height=h, width=w)

        # Add image & dataset information
        target["image_id"] = torch.tensor([image_id], dtype=torch.int64)
        target["orig_size"] = torch.tensor([h, w], dtype=torch.int64)

        # Apply the transformations, converting to tensor if no transforms are provided
        if self.transforms is not None:
            target["boxes"] = BoundingBoxes(target["boxes"], format="CXCYWH", canvas_size=(h, w))
            image, target = self.transforms(image, target)
        else:
            image = to_dtype(to_image(image), torch.float32, scale=True)

        # Normalize bounding box coordinates to [0, 1]
        h, w = image.shape[-2:]
        target["boxes"] = target["boxes"] / torch.tensor([w, h, w, h])
        target["size"] = torch.tensor([h, w], dtype=torch.int64)

        return image, target

    def _convert_object_annotations(self, annotations: ImageAnnotations, height: int, width: int) -> Target:
        """
        Convert COCO annotations to an expected format.

        Expects boxes in XYWH and returns boxes in CXCYWH.

        Args:
            annotations: Object annotations with bbox, category_id, area, iscrowd.
            height: Image height.
            width: Image width.

        Returns:
            annotations: Object annotations with boxes, labels, area, and iscrowd.
        """

        # Remove crowd annotations
        annotations = [obj for obj in annotations if "iscrowd" not in obj or obj["iscrowd"] == 0]

        # Get per-object annotations
        boxes = torch.tensor([obj["bbox"] for obj in annotations], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([self.category_id_to_label[obj["category_id"]] for obj in annotations], dtype=torch.int64)
        area = torch.tensor([obj["area"] for obj in annotations], dtype=torch.float32)
        iscrowd = torch.zeros_like(labels)

        # Convert boxes to XYXY for ease of use (COCO format is XYWH)
        boxes = box_convert(boxes, "xywh", "xyxy")
        boxes = clip_boxes_to_image(boxes, (height, width))

        # Remove invalid boxes (X2 > X1 and Y2 > Y1)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes, labels, area, iscrowd = boxes[keep], labels[keep], area[keep], iscrowd[keep]

        # We expect bounding boxes to be in CXCYWH
        boxes = box_convert(boxes, "xyxy", "cxcywh")

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
        }

        return target

    def _create_coco_dataset(self, roots: List[Path]) -> COCO:
        """
        Create a single COCO dataset from multiple roots.

        Args:
            roots: List of dataset roots to merge.

        Returns:
            coco: Combined COCO dataset.
        """

        dataset = {"images": [], "annotations": [], "categories": []}

        image_id = 1
        annotation_id = 1

        for root_index, root in enumerate(roots):
            with open(root / self.annotation_name) as f:
                root_dataset = json.load(f)

            # Ensure categories are consistent across datasets
            root_categories = sorted(root_dataset["categories"], key=lambda category: category["id"])
            if len(dataset["categories"]) == 0:
                dataset["categories"] = root_categories

            if root_categories != dataset["categories"]:
                raise ValueError(f"Inconsistent categories in {root}: expected {dataset['categories']}, got {root_categories}")

            image_id_map = {}

            for image_info in root_dataset["images"]:
                # Remap image ids to a contigous range to ensure uniqueness
                image_id_map[image_info["id"]] = image_id
                image_info["id"] = image_id
                image_id += 1

                # Resolve image paths now to make loading easier later
                file_name = Path(image_info["file_name"])
                if not file_name.is_absolute():
                    file_name = root / self.image_directory / file_name

                # Fail fast if any images are missing
                if not file_name.exists():
                    raise FileNotFoundError(f"Image not found: {file_name}")

                image_info["file_name"] = str(file_name)

                # Add dataset metadata
                image_info["dataset"] = root_index

            dataset["images"].extend(root_dataset["images"])

            # Annotations
            for annotation in root_dataset["annotations"]:
                # Remap annotation ids to a contigous range to ensure uniqueness
                annotation["id"] = annotation_id
                annotation_id += 1

                # Update the image ids in the annotations to match the new image ids
                annotation["image_id"] = image_id_map[annotation["image_id"]]

            dataset["annotations"].extend(root_dataset["annotations"])

        # Create a COCO object from the merged dataset
        with silence_stdout():
            coco = COCO()
            coco.dataset = dataset
            coco.createIndex()

        return coco

    def get_categories(self) -> List[str]:
        """
        Retrieve the category names in the dataset.

        Returns:
            category_names: List of category names.
        """

        category_names = [self.coco.cats[i]["name"] for i in sorted(self.coco.cats.keys())]

        return category_names


def collate_fn(batch: List[Tuple[Image, Target]]) -> Tuple[Tensor, List[Target]]:
    # Unzip the batch
    images, targets = zip(*batch)

    return torch.stack(images), list(targets)
