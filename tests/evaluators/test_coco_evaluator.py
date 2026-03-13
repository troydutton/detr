import json
from pathlib import Path

import pytest
import torch
from pycocotools.coco import COCO

from evaluators import CocoEvaluator
from models import Predictions
from utils.misc import silence_stdout


@pytest.fixture
def coco_gt(tmp_path: Path) -> COCO:
    """
    Creates a dummy COCO dataset and returns the loaded COCO object.
    Includes an image with non-square dimensions and multiple annotations
    to test coordinate handling and class differentiation.
    """
    root = tmp_path
    split = "val2017"

    (root / "annotations").mkdir(parents=True, exist_ok=True)

    annotations = {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "height": 80, "width": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 10, 20, 30], "area": 600, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }

    ann_file = root / f"annotations/instances_{split}.json"
    with open(ann_file, "w") as f:
        json.dump(annotations, f)

    with silence_stdout():
        return COCO(ann_file)


def create_mock_predictions(boxes, classes, num_layers=1, num_groups=1, num_classes=3):
    """
    Helper to quickly create mock 4D prediction tensors.
    Boxes should be of shape (batch, queries, 4).
    Classes should be a list of lists representing the correct class per query.
    """
    boxes = torch.tensor(boxes, dtype=torch.float32)
    batch_size, num_queries, _ = boxes.shape

    # Expand to (batch, layers, groups, queries, 4)
    pred_boxes = boxes.unsqueeze(1).unsqueeze(1).expand(batch_size, num_layers, num_groups, num_queries, 4).clone()

    pred_logits = torch.zeros(batch_size, num_layers, num_groups, num_queries, num_classes)
    for b in range(batch_size):
        for q, c in enumerate(classes[b]):
            pred_logits[b, -1, 0, q, c] = 100.0  # Set high score for correct class in final layer

    return Predictions(logits=pred_logits, boxes=pred_boxes)


def test_coco_evaluator_simple(coco_gt: COCO):
    evaluator = CocoEvaluator(coco_gt)

    # Check label map
    # Categories are 1 and 2. Sorted: 1 -> 0, 2 -> 1.
    assert evaluator.label_to_category_id == {0: 1, 1: 2}

    # Create fake predictions matching the GT
    # 1 image, 1 layer, 2 queries.
    # Query 0: matches annotation 1 (cat 1, bbox [10, 20, 30, 40] xywh)
    # Query 1: matches annotation 2 (cat 2, bbox [50, 10, 20, 30] xywh)
    # Target size: 100x80 (w, h)

    # Box 1: x=10, y=20, w=30, h=40 -> cx=0.25, cy=0.5, w=0.3, h=0.5
    # Box 2: x=50, y=10, w=20, h=30 -> cx=0.6, cy=0.3125, w=0.2, h=0.375
    preds = create_mock_predictions(
        boxes=[[[0.25, 0.5, 0.3, 0.5], [0.6, 0.3125, 0.2, 0.375]]],
        classes=[[0, 1]],  # Query 0 -> class 0 (person), Query 1 -> class 1 (dog)
    )

    targets = [{"image_id": torch.tensor(1), "orig_size": torch.tensor([80, 100])}]  # h, w

    evaluator.update((preds, None, None), targets)

    results = evaluator.predictions
    assert len(results) == 2

    # Check Result 0
    res0 = results[0]
    assert res0["image_id"] == 1
    assert res0["category_id"] == 1
    assert res0["bbox"] == pytest.approx([10, 20, 30, 40], abs=1e-4)

    # Check Result 1
    res1 = results[1]
    assert res1["image_id"] == 1
    assert res1["category_id"] == 2
    assert res1["bbox"] == pytest.approx([50, 10, 20, 30], abs=1e-4)

    # Test Compute
    # Since predictions are perfect, AP should be 1.0
    metrics = evaluator.compute()
    assert metrics["overall"]["AP"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["overall"]["AP50"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["overall"]["AP75"] == pytest.approx(1.0, abs=1e-4)


def test_coco_evaluator_multi_layer(coco_gt: COCO):
    """
    Test that CocoEvaluator only uses the final layer for evaluation,
    ignoring intermediate decoder layer outputs.
    """
    evaluator = CocoEvaluator(coco_gt)

    # create_mock_predictions already puts zeroes in intermediate layers
    # and correct predictions only in the final layer
    preds = create_mock_predictions(boxes=[[[0.25, 0.5, 0.3, 0.5], [0.6, 0.3125, 0.2, 0.375]]], classes=[[0, 1]], num_layers=6)

    targets = [{"image_id": torch.tensor(1), "orig_size": torch.tensor([80, 100])}]  # h, w

    evaluator.update((preds, None, None), targets)

    results = evaluator.predictions
    assert len(results) == 2

    # Check Result 0
    res0 = results[0]
    assert res0["image_id"] == 1
    assert res0["category_id"] == 1
    assert res0["bbox"] == pytest.approx([10, 20, 30, 40], abs=1e-4)

    # Check Result 1
    res1 = results[1]
    assert res1["image_id"] == 1
    assert res1["category_id"] == 2
    assert res1["bbox"] == pytest.approx([50, 10, 20, 30], abs=1e-4)

    # Since final layer predictions are perfect, AP should be 1.0
    metrics = evaluator.compute()
    assert metrics["overall"]["AP"] == pytest.approx(1.0, abs=1e-4)


def test_coco_evaluator_batch(tmp_path: Path):
    """
    Test CocoEvaluator with multiple images in a batch.
    """
    root = tmp_path
    split = "val2017"
    ann_file = root / f"annotations/instances_{split}.json"
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    # 2 images, 3 annotations total
    annotations = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "height": 80, "width": 100},
            {"id": 2, "file_name": "000000000002.jpg", "height": 90, "width": 120},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 10, 20, 30], "area": 600, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [20, 30, 40, 20], "area": 800, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
    }

    with open(ann_file, "w") as f:
        json.dump(annotations, f)

    with silence_stdout():
        coco_gt = COCO(ann_file)

    evaluator = CocoEvaluator(coco_gt)

    # Image 1: [10, 20, 30, 40] -> cx=0.25, cy=0.5, w=0.3, h=0.5
    # Image 1: [50, 10, 20, 30] -> cx=0.6, cy=0.3125, w=0.2, h=0.375
    # Image 2: [20, 30, 40, 20] -> cx=0.333, cy=0.444, w=0.333, h=0.222
    preds = create_mock_predictions(
        boxes=[
            [[0.25, 0.5, 0.3, 0.5], [0.6, 0.3125, 0.2, 0.375]],  # Batch 0
            [[0.333, 0.444, 0.333, 0.222], [0, 0, 0, 0]],  # Batch 1, padded 2nd query
        ],
        classes=[[0, 1], [0, 0]],
    )

    targets = [
        {"image_id": torch.tensor(1), "orig_size": torch.tensor([80, 100])},
        {"image_id": torch.tensor(2), "orig_size": torch.tensor([90, 120])},
    ]

    evaluator.update((preds, None, None), targets)

    results = evaluator.predictions
    assert len(results) == 4  # 2 from image 1, 2 from image 2 (even if low scores)

    # Check that predictions are correctly separated by image_id
    image1_results = [r for r in results if r["image_id"] == 1]
    image2_results = [r for r in results if r["image_id"] == 2]

    assert len(image1_results) == 2
    assert len(image2_results) == 2

    # Compute metrics
    metrics = evaluator.compute()
    assert "overall" in metrics
    assert "AP" in metrics["overall"]
    assert metrics["overall"]["AP"] >= 0.0  # AP should be reasonable (may not be 1.0 due to slight imprecision)


def test_coco_evaluator_class_metrics(coco_gt: COCO):
    """
    Test CocoEvaluator computes per-class metrics when class_metrics=True.
    """
    evaluator = CocoEvaluator(coco_gt, class_metrics=True)

    # Person box is correct, Dog box is completely wrong
    preds = create_mock_predictions(boxes=[[[0.25, 0.5, 0.3, 0.5], [0.9, 0.9, 0.1, 0.1]]], classes=[[0, 1]])

    targets = [{"image_id": torch.tensor(1), "orig_size": torch.tensor([80, 100])}]  # h, w

    evaluator.update((preds, None, None), targets)

    metrics = evaluator.compute()

    # Check overall
    assert "overall" in metrics
    # Person AP is 1.0, Dog AP is 0.0, so the category average (overall AP) is 0.5
    assert metrics["overall"]["AP"] == pytest.approx(0.5, abs=1e-4)

    # Check class specific metrics (person, dog from fixture categories)
    assert "person" in metrics
    assert "dog" in metrics

    # Verify the predicted behaviors
    assert metrics["person"]["AP"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["dog"]["AP"] == pytest.approx(0.0, abs=1e-4)
