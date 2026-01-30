import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from data import CocoDataset, make_transformations


@pytest.fixture
def coco_path(tmp_path: Path) -> Path:
    """
    Creates a dummy COCO dataset structure in a temporary directory.
    Includes an image with non-square dimensions and multiple annotations
    to test coordinate handling and class differentiation.
    """

    root = tmp_path
    split = "val2017"

    # Replicate COCO directory structure
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / split).mkdir(parents=True, exist_ok=True)

    # Dummy image parameters
    width, height = 100, 80
    image_id = 1
    file_name = "000000000001.jpg"

    # Dummy image
    image = Image.new("RGB", (width, height), color="red")
    image.save(root / split / file_name)

    # Dummy annotations
    annotations = {
        "images": [{"id": image_id, "file_name": file_name, "height": height, "width": width}],
        "annotations": [
            {"id": 1, "image_id": image_id, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 2, "image_id": image_id, "category_id": 2, "bbox": [50, 10, 20, 30], "area": 600, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }
    with open(root / f"annotations/instances_{split}.json", "w") as f:
        json.dump(annotations, f)

    return root


def test_coco_dataset_basic(coco_path: Path) -> None:
    """
    Test basic functionality of the CocoDataset without transformations.
    Verifies output shapes, target fields, and box formats for multiple objects.
    """
    split = "val2017"
    dataset = CocoDataset(str(coco_path), split, transforms=None)

    assert len(dataset) == 1

    image, target = dataset[0]

    # Verify image
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 80, 100)  # (C, H, W)

    # Verify image fields
    assert target["image_id"].item() == 1
    assert target["orig_size"].tolist() == [80, 100]  # (H, W)
    assert target["size"].tolist() == [80, 100]  # (H, W)

    # Verify boxes
    boxes = target["boxes"]
    assert boxes.shape == (2, 4)

    expected_box1 = torch.tensor([0.25, 0.50, 0.30, 0.50])
    expected_box2 = torch.tensor([0.60, 0.3125, 0.20, 0.375])

    # Find matching boxes
    is_box1_present = False
    is_box2_present = False

    for box in boxes:
        if torch.allclose(box, expected_box1, atol=1e-4):
            is_box1_present = True
        elif torch.allclose(box, expected_box2, atol=1e-4):
            is_box2_present = True

    assert is_box1_present, f"Expected box 1 {expected_box1} not found in {boxes}"
    assert is_box2_present, f"Expected box 2 {expected_box2} not found in {boxes}"

    # Verify labels
    labels = target["labels"]
    assert labels.shape == (2,)
    assert 0 in labels
    assert 1 in labels

    # Check correspondence
    for box, label in zip(boxes, labels):
        if torch.allclose(box, expected_box1, atol=1e-4):
            assert label.item() == 0
        elif torch.allclose(box, expected_box2, atol=1e-4):
            assert label.item() == 1


def test_coco_dataset_transforms(coco_path: Path) -> None:
    """
    Test functionality with transformations pipeline.
    Verifies image resizing and box coordinate normalization.
    """
    split = "val2017"
    resolution = 200

    # Create transformation pipeline
    transforms = make_transformations("val", resolution=resolution, square_resize=True)

    dataset = CocoDataset(str(coco_path), split, transforms=transforms)

    image, target = dataset[0]

    # Verify image shape after resize
    assert image.shape == (3, 200, 200)  # (C, H, W)

    # Verify size in target
    assert target["size"].tolist() == [200, 200]

    # Normalized coordinates should be preserved even after resizing
    expected_box1 = torch.tensor([0.25, 0.50, 0.30, 0.50])
    expected_box2 = torch.tensor([0.60, 0.3125, 0.20, 0.375])

    boxes = target["boxes"]

    is_box1_present = False
    is_box2_present = False

    for box in boxes:
        if torch.allclose(box, expected_box1, atol=1e-4):
            is_box1_present = True
        elif torch.allclose(box, expected_box2, atol=1e-4):
            is_box2_present = True

    assert is_box1_present, f"Expected box 1 {expected_box1} not found in transformed {boxes}"
    assert is_box2_present, f"Expected box 2 {expected_box2} not found in transformed {boxes}"
