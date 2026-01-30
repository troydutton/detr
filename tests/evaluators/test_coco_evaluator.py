import json
from pathlib import Path

import pytest
import torch
from pycocotools.coco import COCO

from evaluators import CocoEvaluator
from utils.misc import silence_stdout


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


def test_coco_evaluator_update_and_compute(coco_path: Path):
    root = coco_path
    split = "val2017"
    ann_file = root / f"annotations/instances_{split}.json"

    # Initialize COCO GT
    with silence_stdout():
        coco_gt = COCO(ann_file)

    evaluator = CocoEvaluator(coco_gt)

    # Check label map
    # Categories are 1 and 2. Sorted: 1 -> 0, 2 -> 1.
    assert evaluator.label_to_category_id == {0: 1, 1: 2}

    # Create fake predictions matching the GT
    # 1 image, 2 queries.
    # Query 0: matches annotation 1 (cat 1, bbox [10, 20, 30, 40] xywh)
    # Query 1: matches annotation 2 (cat 2, bbox [50, 10, 20, 30] xywh)
    # Target size: 100x80 (w, h)

    # Box transformation: cx = x + w/2, cy = y + h/2. Scale by w=100, h=80.
    # Box 1: x=10, y=20, w=30, h=40
    # cx = 25, cy = 40. Normalized: cx=0.25, cy=0.5, w=0.3, h=0.5

    # Box 2: x=50, y=10, w=20, h=30
    # cx = 60, cy = 25. Normalized: cx=0.6, cy=0.3125, w=0.2, h=0.375

    pred_boxes = torch.tensor([[0.25, 0.5, 0.3, 0.5], [0.6, 0.3125, 0.2, 0.375]]).unsqueeze(0)  # [1, 2, 4]

    pred_logits = torch.zeros(1, 2, 3)  # 2 classes + 1 background.
    # Logits: high score for correct class.
    # Query 0: class 0 (cat ID 1).
    pred_logits[0, 0, 0] = 100.0  # High logit for softmax
    pred_logits[0, 0, 1] = 0.0
    pred_logits[0, 0, 2] = 0.0

    # Query 1: class 1 (cat ID 2).
    pred_logits[0, 1, 1] = 100.0
    pred_logits[0, 1, 0] = 0.0
    pred_logits[0, 1, 2] = 0.0

    predictions = {"logits": pred_logits, "boxes": pred_boxes}

    # Targets only need image_id and orig_size
    targets = [{"image_id": torch.tensor(1), "orig_size": torch.tensor([80, 100])}]  # h, w

    evaluator.update(predictions, targets)

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
    assert metrics["AP"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["AP50"] == pytest.approx(1.0, abs=1e-4)
    assert metrics["AP75"] == pytest.approx(1.0, abs=1e-4)
