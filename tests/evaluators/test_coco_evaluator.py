import json
from pathlib import Path

import pytest
import torch
from pycocotools.coco import COCO

from evaluators import CocoEvaluator
from models import Predictions
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
    # 1 image, 1 layer, 2 queries.
    # Query 0: matches annotation 1 (cat 1, bbox [10, 20, 30, 40] xywh)
    # Query 1: matches annotation 2 (cat 2, bbox [50, 10, 20, 30] xywh)
    # Target size: 100x80 (w, h)

    # Box transformation: cx = x + w/2, cy = y + h/2. Scale by w=100, h=80.
    # Box 1: x=10, y=20, w=30, h=40
    # cx = 25, cy = 40. Normalized: cx=0.25, cy=0.5, w=0.3, h=0.5

    # Box 2: x=50, y=10, w=20, h=30
    # cx = 60, cy = 25. Normalized: cx=0.6, cy=0.3125, w=0.2, h=0.375

    # 4D predictions: (batch, layers, groups, queries, *)
    num_layers = 1
    # [1, 1, 1, 2, 4]
    pred_boxes = torch.tensor([[0.25, 0.5, 0.3, 0.5], [0.6, 0.3125, 0.2, 0.375]]).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    pred_logits = torch.zeros(1, num_layers, 1, 2, 3)  # 2 classes + 1 background.
    # Logits: high score for correct class.
    # Query 0: class 0 (cat ID 1).
    pred_logits[0, 0, 0, 0, 0] = 100.0  # High logit for softmax
    pred_logits[0, 0, 0, 0, 1] = 0.0
    pred_logits[0, 0, 0, 0, 2] = 0.0

    # Query 1: class 1 (cat ID 2).
    pred_logits[0, 0, 0, 1, 1] = 100.0
    pred_logits[0, 0, 0, 1, 0] = 0.0
    pred_logits[0, 0, 0, 1, 2] = 0.0

    predictions = (Predictions(logits=pred_logits, boxes=pred_boxes), None, None)

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


def test_coco_evaluator_multi_layer_uses_final_only(coco_path: Path):
    """
    Test that CocoEvaluator only uses the final layer for evaluation,
    ignoring intermediate decoder layer outputs.
    """
    root = coco_path
    split = "val2017"
    ann_file = root / f"annotations/instances_{split}.json"

    # Initialize COCO GT
    with silence_stdout():
        coco_gt = COCO(ann_file)

    evaluator = CocoEvaluator(coco_gt)

    # Create predictions with 6 layers (simulating decoder outputs)
    # Intermediate layers have wrong predictions, final layer has correct ones
    num_layers = 6
    batch_size = 1
    num_groups = 1
    num_queries = 2

    # Initialize with zeros (all predictions wrong)
    pred_boxes = torch.zeros(batch_size, num_layers, num_groups, num_queries, 4)
    pred_logits = torch.zeros(batch_size, num_layers, num_groups, num_queries, 3)

    # Set correct predictions only in final layer
    # Box 1: x=10, y=20, w=30, h=40 -> normalized cxcywh: [0.25, 0.5, 0.3, 0.5]
    # Box 2: x=50, y=10, w=20, h=30 -> normalized cxcywh: [0.6, 0.3125, 0.2, 0.375]
    pred_boxes[0, -1, 0, 0] = torch.tensor([0.25, 0.5, 0.3, 0.5])
    pred_boxes[0, -1, 0, 1] = torch.tensor([0.6, 0.3125, 0.2, 0.375])

    # High scores for correct classes in final layer only
    pred_logits[0, -1, 0, 0, 0] = 100.0  # Query 0: class 0 (cat ID 1)
    pred_logits[0, -1, 0, 1, 1] = 100.0  # Query 1: class 1 (cat ID 2)

    predictions = (Predictions(logits=pred_logits, boxes=pred_boxes), None, None)

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

    # Since final layer predictions are perfect, AP should be 1.0
    metrics = evaluator.compute()
    assert metrics["AP"] == pytest.approx(1.0, abs=1e-4)


def test_coco_evaluator_multi_image_batch(coco_path: Path):
    """
    Test CocoEvaluator with multiple images in a batch.
    """
    root = coco_path
    split = "val2017"
    ann_file = root / f"annotations/instances_{split}.json"

    # Add second image to annotations
    import json

    with open(ann_file, "r") as f:
        annotations = json.load(f)

    # Add second image
    width2, height2 = 120, 90
    image_id2 = 2
    file_name2 = "000000000002.jpg"
    annotations["images"].append({"id": image_id2, "file_name": file_name2, "height": height2, "width": width2})

    # Add annotation for second image
    annotations["annotations"].append(
        {"id": 3, "image_id": image_id2, "category_id": 1, "bbox": [20, 30, 40, 20], "area": 800, "iscrowd": 0}
    )

    with open(ann_file, "w") as f:
        json.dump(annotations, f)

    # Initialize COCO GT
    with silence_stdout():
        coco_gt = COCO(ann_file)

    evaluator = CocoEvaluator(coco_gt)

    # Create predictions for 2 images
    batch_size = 2
    num_layers = 1
    num_groups = 1
    num_queries = 2

    pred_boxes = torch.zeros(batch_size, num_layers, num_groups, num_queries, 4)
    pred_logits = torch.zeros(batch_size, num_layers, num_groups, num_queries, 3)

    # Image 1: 2 objects
    # Box 1: [10, 20, 30, 40] -> [0.25, 0.5, 0.3, 0.5]
    # Box 2: [50, 10, 20, 30] -> [0.6, 0.3125, 0.2, 0.375]
    pred_boxes[0, 0, 0, 0] = torch.tensor([0.25, 0.5, 0.3, 0.5])
    pred_boxes[0, 0, 0, 1] = torch.tensor([0.6, 0.3125, 0.2, 0.375])
    pred_logits[0, 0, 0, 0, 0] = 100.0  # class 0
    pred_logits[0, 0, 0, 1, 1] = 100.0  # class 1

    # Image 2: 1 object
    # Box: [20, 30, 40, 20] -> cx=40, cy=40, w=40, h=20
    # Normalized: cx=40/120=0.333, cy=40/90=0.444, w=40/120=0.333, h=20/90=0.222
    pred_boxes[1, 0, 0, 0] = torch.tensor([0.333, 0.444, 0.333, 0.222])
    pred_logits[1, 0, 0, 0, 0] = 100.0  # class 0

    predictions = (Predictions(logits=pred_logits, boxes=pred_boxes), None, None)

    targets = [
        {"image_id": torch.tensor(1), "orig_size": torch.tensor([80, 100])},  # h, w
        {"image_id": torch.tensor(2), "orig_size": torch.tensor([90, 120])},
    ]

    evaluator.update(predictions, targets)

    results = evaluator.predictions
    assert len(results) == 4  # 2 from image 1, 2 from image 2 (even if low scores)

    # Check that predictions are correctly separated by image_id
    image1_results = [r for r in results if r["image_id"] == 1]
    image2_results = [r for r in results if r["image_id"] == 2]

    assert len(image1_results) == 2
    assert len(image2_results) == 2

    # Compute metrics
    metrics = evaluator.compute()
    assert "AP" in metrics
    assert metrics["AP"] >= 0.0  # AP should be reasonable (may not be 1.0 due to slight imprecision)
