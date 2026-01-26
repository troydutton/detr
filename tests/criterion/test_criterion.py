import torch

from criterion.criterion import SetCriterion


def test_set_criterion_forward() -> None:
    """
    Test the forward pass of the SetCriterion.
    """
    # Test Parameters
    batch_size = 2
    num_queries = 50
    num_classes = 10
    num_targets = [3, 5]
    loss_weights = {
        "class": 1.0,
        "box": 5.0,
        "giou": 2.0,
    }

    # Dummy predictions
    pred_logits = torch.randn(batch_size, num_queries, num_classes)
    pred_boxes = torch.rand(batch_size, num_queries, 4)

    predictions = {"logits": pred_logits, "boxes": pred_boxes}

    # Dummy targets
    targets = []
    for i in range(batch_size):
        # Dummy targets
        labels = torch.randint(0, num_classes, (num_targets[i],))
        boxes = torch.rand(num_targets[i], 4)

        targets.append({"labels": labels, "boxes": boxes})

    criterion = SetCriterion(loss_weights=loss_weights)

    _ = criterion(predictions, targets)
