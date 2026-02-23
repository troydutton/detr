import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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

COCOAnnotations = List[Dict[str, Any]]
Target = Dict[str, Union[Tensor, BoundingBoxes]]


class CocoDataset(Dataset):
    """
    COCO-style object detection dataset.

    Args:
        dataset_root: Root of the dataset.
        split: Dataset split to use ("train", "val", or "test").
        transforms: Transformations to apply to each image, optional.
    """

    def __init__(self, dataset_root: Union[str, Path], split: str, transforms: Transformation = None) -> None:
        self.root = Path(dataset_root)
        self.split = split
        self.transforms = transforms

        logging.info(f"Loading '{self.root / self.split}'.")

        # Initialize COCO API (silencing stdout to avoid clutter)
        with silence_stdout():
            self.coco = COCO(self.root / f"annotations/{split}.json")
            self.coco.createIndex()
        self.image_ids = list(self.coco.imgs.keys())

        # Mapping from category ids to contiguous 0-indexed labels
        category_ids = sorted(self.coco.getCatIds())
        self.category_id_to_label = {cat_id: i for i, cat_id in enumerate(category_ids)}
        self.num_classes = len(self.category_id_to_label)

        logging.info(f"Loaded {len(self.image_ids):,} images with {self.num_classes} classes from '{self.root / self.split}'.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[Image, Target]:
        """
        Retrieve an image and its corresponding annotations.

        Args:
            idx: Index of the image to retrieve.

        Returns:
            image: Image tensor.
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
        image = PIL.Image.open(self.root / self.split / image_info["file_name"]).convert("RGB")
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

    def _convert_object_annotations(self, annotations: COCOAnnotations, height: int, width: int) -> Target:
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

    def calculate_class_weights(self, *, beta: float = 0.5) -> Tensor:
        """
        Calculates class weights based on their frequency.

        Specifically the weight for class i is given by w_i = (Σ n_j) / ((Σ n_j^(1-β)) * n_i^β))
        where n_i is the number of samples in class i.

        Args:
            beta: Power-law reweighting factor, β < 1 → smoother weights, β > 1 → more extreme weights.

        Returns:
            class_weights: Weights for each class with shape (num_classes)
        """

        logging.info(f"Calculating class weights with {beta=}.")

        # Count the number of occurences of each class
        frequencies = torch.zeros(self.num_classes)

        for annotation in self.coco.anns.values():
            frequencies[self.category_id_to_label[annotation["category_id"]]] += 1

        # Normalization factor s.t. the expected weight is 1
        # When β=1 this reduces to  (Σ n_i) / num_classes
        k = frequencies.sum() / frequencies.pow(1 - beta).sum()

        # Weights are inversely proportional to frequency
        weights = torch.ones_like(frequencies)
        weights[frequencies > 0] = k * frequencies[frequencies > 0].pow(-beta)

        return weights

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
