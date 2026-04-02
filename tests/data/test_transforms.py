"""
Tests for data transformation utilities.
"""

from unittest.mock import patch

import pytest
import torch
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from data.transforms import DiscreteRandomResize, RandomErasingBoxAware, make_transformations


def test_random_erasing_box_aware_erasing() -> None:
    """
    Test RandomErasingBoxAware erasing logic and handling of bounding boxes.
    """
    image = torch.rand(3, 100, 100)
    # Box 1: [0, 0, 50, 50], Box 2: [60, 60, 100, 100], Box 3: [20, 20, 40, 40]
    boxes = BoundingBoxes(
        torch.tensor([[0, 0, 50, 50], [60, 60, 100, 100], [20, 20, 40, 40]], dtype=torch.float32),
        format=BoundingBoxFormat.XYXY,
        canvas_size=(100, 100),
    )
    targets = {"boxes": boxes, "labels": torch.tensor([1, 2, 3]), "area": torch.tensor([2500, 1600, 400])}

    transform = RandomErasingBoxAware(p=1.0)

    # Mock make_params to return a specific patch
    # Say we erase x=10, y=10, w=30, h=30 (x2=40, y2=40)
    # Box 1 [0,0,50,50] overlaps with erased region [10,10,40,40]
    # Box 2 [60,60,100,100] doesn't overlap
    # Box 3 [20,20,40,40] overlaps almost completely -> should be erased
    patch_params = {"v": torch.tensor([0.0]), "i": 10, "j": 10, "h": 30, "w": 30}

    with patch("torchvision.transforms.v2.RandomErasing.make_params", return_value=patch_params):
        out_img, out_targets = transform(image.clone(), targets.copy())

    out_boxes = out_targets["boxes"]
    assert len(out_boxes) == 2


def test_discrete_random_resize() -> None:
    """
    Test DiscreteRandomResize with a specific resolution.
    """
    images = torch.rand(2, 3, 100, 100)
    targets = [{"size": torch.tensor([100, 100])}, {"size": torch.tensor([100, 100])}]

    # Test same resolution (no-op)
    transform = DiscreteRandomResize([100])
    out_imgs, out_targets = transform(images.clone(), targets.copy())
    assert out_imgs.shape == (2, 3, 100, 100)

    # Test resize
    transform_resize = DiscreteRandomResize([50])
    out_imgs_resize, out_targets_resize = transform_resize(images.clone(), targets.copy())
    assert out_imgs_resize.shape == (2, 3, 50, 50)
    assert out_targets_resize[0]["size"].tolist() == [50, 50]


def test_make_transformations() -> None:
    """
    Test make_transformations factory.
    """
    train_transform = make_transformations("train", 100)
    assert train_transform is not None

    val_transform = make_transformations("val", 100)
    assert val_transform is not None

    image = Image.new("RGB", (200, 200))
    boxes = BoundingBoxes(torch.tensor([[10, 10, 50, 50]], dtype=torch.float32), format=BoundingBoxFormat.XYXY, canvas_size=(200, 200))
    targets = {"boxes": boxes, "labels": torch.tensor([1]), "area": torch.tensor([1600])}

    image, targets = val_transform(image, targets)
    assert image.shape == (3, 100, 100)
    assert targets["boxes"].shape == (1, 4)
    assert targets["labels"].shape == (1,)

    test_transform = make_transformations("test", 100, normalize=False)
    assert test_transform is not None

    with pytest.raises(ValueError):
        make_transformations("invalid", 100)
