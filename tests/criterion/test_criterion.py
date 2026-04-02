from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from criterion.criterion import Criterion
from criterion.hungarian_matcher import HungarianMatcher
from models.detr import Predictions


class TestHungarianMatcher:
    """
    Tests for the HungarianMatcher class.
    """

    def test_matcher_call(self) -> None:
        """
        Tests the HungarianMatcher call method with standard inputs.
        """
        batch_size = 2
        num_layers = 2
        num_groups = 1
        num_queries = 10
        num_classes = 5

        predictions = Predictions(
            logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.tensor([0, 1]),
                "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.4, 0.4]]),
            },
            {
                "labels": torch.tensor([2]),
                "boxes": torch.tensor([[0.5, 0.5, 0.6, 0.6]]),
            },
        ]

        matcher = HungarianMatcher(cost_weights={"class": 1.0, "box": 5.0, "giou": 2.0})
        matched_indices, target_indices = matcher(predictions, targets)

        # Batch, Layer, Group, Query arrays
        assert len(matched_indices) == 4
        assert len(target_indices) == 2

        # 2 matches for image 0, 1 match for image 1. Total matches = 3 per layer x 2 layers = 6.
        assert matched_indices[0].shape == torch.Size([6])

    def test_matcher_empty_targets(self) -> None:
        """
        Tests matcher with no targets.
        """
        batch_size = 1
        num_layers = 1
        num_groups = 1
        num_queries = 5
        num_classes = 3

        predictions = Predictions(
            logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.empty(0, dtype=torch.int64),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
            }
        ]

        matcher = HungarianMatcher(cost_weights={"class": 1.0, "box": 5.0, "giou": 2.0})
        matched_indices, target_indices = matcher(predictions, targets)

        assert len(matched_indices[0]) == 0
        assert len(target_indices) == 1
        assert len(target_indices[0]) == 0

    def test_matcher_handling_cost_weights(self) -> None:
        """
        Tests that matcher uses default 1.0 if cost is not in weights.
        """
        matcher = HungarianMatcher(cost_weights={"class": 2.0})  # Missing Box and giou

        batch_size = 1
        num_layers = 1
        num_groups = 1
        num_queries = 2
        num_classes = 2

        predictions = Predictions(
            logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.1, 0.1, 0.2, 0.2]]),
            }
        ]

        # execution should not fail
        matched_indices, target_indices = matcher(predictions, targets)
        assert len(matched_indices[0]) == 1


class TestCriterion:
    """
    Tests for the Criterion class.
    """

    def test_criterion_call(self) -> None:
        """
        Tests standard Criterion call with decoder, encoder, and denoise preds.
        """
        batch_size = 2
        num_layers = 2
        num_groups = 1
        num_queries = 5
        num_classes = 4

        decoder_preds = Predictions(
            logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
        )
        encoder_preds = Predictions(
            logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
        )
        denoise_preds = Predictions(
            logits=torch.randn((batch_size, num_layers, 1, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, 1, num_queries, 4)),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            },
            {
                "labels": torch.empty(0, dtype=torch.int64),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
            },
        ]

        criterion = Criterion(loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0})

        losses = criterion((decoder_preds, encoder_preds, denoise_preds), targets)

        assert "box" in losses
        assert "class" in losses
        assert "giou" in losses
        assert "overall" in losses
        assert losses["overall"] > 0

    def test_criterion_no_encoder_no_denoise(self) -> None:
        """
        Tests Criterion call with only decoder predictions.
        """
        batch_size = 1
        num_layers = 1
        num_groups = 1
        num_queries = 5
        num_classes = 4

        decoder_preds = Predictions(
            logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            }
        ]

        criterion = Criterion(loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0}, cost_weights={"class": 2.0, "box": 5.0, "giou": 2.0})

        losses = criterion((decoder_preds, None, None), targets)

        assert losses["box"] >= 0
        assert losses["giou"] >= 0

    def test_criterion_empty_targets(self) -> None:
        """
        Tests Criterion call with empty targets.
        """
        batch_size = 1
        num_layers = 1
        num_groups = 1
        num_queries = 5
        num_classes = 4

        decoder_preds = Predictions(
            logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.empty(0, dtype=torch.int64),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
            }
        ]

        criterion = Criterion(loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0})

        losses = criterion((decoder_preds, None, None), targets)
        assert losses["box"] == 0
        assert losses["giou"] == 0
