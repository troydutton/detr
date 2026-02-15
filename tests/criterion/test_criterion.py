import torch

from criterion import SetCriterion
from models import ModelPredictions, Predictions


def test_set_criterion_single_layer() -> None:
    """
    Test the forward pass of the SetCriterion with single layer predictions.
    """
    # Test Parameters
    batch_size = 2
    num_layers = 1
    num_queries = 50
    num_classes = 10
    num_targets = [3, 5]
    loss_weights = {
        "class": 1.0,
        "box": 5.0,
        "giou": 2.0,
    }

    # Dummy predictions with 4D tensors (batch, layers, queries, *)
    pred_logits = torch.randn(batch_size, num_layers, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_layers, num_queries, 4)

    predictions = ModelPredictions(decoder=Predictions(logits=pred_logits, boxes=pred_boxes))

    # Dummy targets
    targets = []
    for i in range(batch_size):
        labels = torch.randint(0, num_classes, (num_targets[i],))
        boxes = torch.rand(num_targets[i], 4)

        targets.append({"labels": labels, "boxes": boxes})

    criterion = SetCriterion(loss_weights=loss_weights)

    losses = criterion(predictions, targets)

    assert "overall" in losses, "overall loss not computed."

    for loss_name in loss_weights.keys():
        assert loss_name in losses, f"{loss_name} loss not computed."


def test_set_criterion_multi_layer() -> None:
    """
    Test the forward pass of the SetCriterion with multi-layer predictions.
    Verifies that losses are computed across all decoder layers.
    """
    # Test Parameters
    batch_size = 2
    num_layers = 6
    num_queries = 50
    num_classes = 10
    num_targets = [3, 5]
    loss_weights = {
        "class": 1.0,
        "box": 5.0,
        "giou": 2.0,
    }

    # Dummy predictions with 4D tensors (batch, layers, queries, *)
    pred_logits = torch.randn(batch_size, num_layers, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_layers, num_queries, 4)

    predictions = ModelPredictions(decoder=Predictions(logits=pred_logits, boxes=pred_boxes))

    # Dummy targets
    targets = []
    for i in range(batch_size):
        labels = torch.randint(0, num_classes, (num_targets[i],))
        boxes = torch.rand(num_targets[i], 4)

        targets.append({"labels": labels, "boxes": boxes})

    criterion = SetCriterion(loss_weights=loss_weights)

    losses = criterion(predictions, targets)

    assert "overall" in losses, "overall loss not computed."

    for loss_name in loss_weights.keys():
        assert loss_name in losses, f"{loss_name} loss not computed."

    # Verify losses are finite (no NaN or Inf)
    for loss_name, loss_value in losses.items():
        assert torch.isfinite(loss_value), f"{loss_name} loss is not finite: {loss_value}"


def test_set_criterion_empty_targets() -> None:
    """
    Test the forward pass of the SetCriterion with empty targets.
    Verifies that division by zero is handled (currently results in NaN).
    """
    # Test Parameters
    batch_size = 2
    num_layers = 3
    num_queries = 50
    num_classes = 10
    loss_weights = {
        "class": 1.0,
        "box": 5.0,
        "giou": 2.0,
    }

    # Dummy predictions
    pred_logits = torch.randn(batch_size, num_layers, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_layers, num_queries, 4)

    predictions = ModelPredictions(decoder=Predictions(logits=pred_logits, boxes=pred_boxes))

    # Empty targets (no objects in images)
    targets = [
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
        {"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty(0, 4)},
    ]

    criterion = SetCriterion(loss_weights=loss_weights)

    losses = criterion(predictions, targets)

    assert "overall" in losses, "overall loss not computed."

    # With empty targets, losses are divided by 0, resulting in NaN or inf
    # This is expected behavior that should be handled during training
    for loss_name in ["box", "giou", "class", "overall"]:
        assert torch.isfinite(losses[loss_name]), f"{loss_name} loss should be finite."


def test_set_criterion_single_target() -> None:
    """
    Test the forward pass of the SetCriterion with a single target per image.
    """
    # Test Parameters
    batch_size = 2
    num_layers = 3
    num_queries = 50
    num_classes = 10
    loss_weights = {
        "class": 1.0,
        "box": 5.0,
        "giou": 2.0,
    }

    # Dummy predictions
    pred_logits = torch.randn(batch_size, num_layers, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_layers, num_queries, 4)

    predictions = ModelPredictions(decoder=Predictions(logits=pred_logits, boxes=pred_boxes))

    # Single target per image
    targets = []
    for i in range(batch_size):
        labels = torch.randint(0, num_classes, (1,))
        boxes = torch.rand(1, 4)

        targets.append({"labels": labels, "boxes": boxes})

    criterion = SetCriterion(loss_weights=loss_weights)

    losses = criterion(predictions, targets)

    assert "overall" in losses, "overall loss not computed."

    for loss_name in loss_weights.keys():
        assert loss_name in losses, f"{loss_name} loss not computed."

    # Verify losses are finite and positive
    for loss_name, loss_value in losses.items():
        assert torch.isfinite(loss_value), f"{loss_name} loss is not finite: {loss_value}"
        assert loss_value >= 0.0, f"{loss_name} loss should be non-negative."
