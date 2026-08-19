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
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
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
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
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
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
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
        num_bins = 4

        decoder_preds = Predictions(
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            edge_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, 4 * (num_bins + 1))),
        )
        encoder_preds = Predictions(
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
        )
        denoise_preds = Predictions(
            boxes=torch.rand((batch_size, num_layers, 1, num_queries, 4)),
            class_logits=torch.randn((batch_size, num_layers, 1, num_queries, num_classes)),
            edge_logits=torch.randn((batch_size, num_layers, 1, num_queries, 4 * (num_bins + 1))),
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

        criterion = Criterion(loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0}, num_bins=num_bins)

        losses = criterion((decoder_preds, encoder_preds, denoise_preds), targets)

        assert "box" in losses
        assert "class" in losses
        assert "giou" in losses
        assert "localization" in losses
        assert "overall" in losses

        for key, value in losses.items():
            assert torch.isfinite(value), f"Loss {key} is not finite: {value.item()}"

    def test_criterion_no_encoder_no_denoise(self) -> None:
        """
        Tests Criterion call with only decoder predictions.
        """
        batch_size = 1
        num_layers = 1
        num_groups = 1
        num_queries = 5
        num_classes = 4
        num_bins = 4

        decoder_preds = Predictions(
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            edge_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, 4 * (num_bins + 1))),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            }
        ]

        criterion = Criterion(
            loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0}, cost_weights={"class": 2.0, "box": 5.0, "giou": 2.0}, num_bins=num_bins
        )

        losses = criterion((decoder_preds, None, None), targets)

        assert "box" in losses
        assert "class" in losses
        assert "giou" in losses
        assert "localization" in losses
        assert "overall" in losses

        for key, value in losses.items():
            assert torch.isfinite(value), f"Loss {key} is not finite: {value.item()}"

    def test_criterion_empty_targets(self) -> None:
        """
        Tests Criterion call with empty targets.
        """
        batch_size = 1
        num_layers = 1
        num_groups = 1
        num_queries = 5
        num_classes = 4
        num_bins = 4

        decoder_preds = Predictions(
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4)),
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes)),
            edge_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, 4 * (num_bins + 1))),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.empty(0, dtype=torch.int64),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
            }
        ]

        criterion = Criterion(loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0}, num_bins=num_bins)

        losses = criterion((decoder_preds, None, None), targets)

        assert "box" in losses
        assert "class" in losses
        assert "giou" in losses
        assert "localization" in losses
        assert "overall" in losses

        for key, value in losses.items():
            assert torch.isfinite(value), f"Loss {key} is not finite: {value.item()}"

    def test_empty_targets_keep_regression_losses_connected(self) -> None:
        """
        Empty local batches should still mark regression predictions as used.
        """

        batch_size = 1
        num_layers = 1
        num_groups = 1
        num_queries = 5
        num_classes = 4
        num_bins = 4

        decoder_preds = Predictions(
            boxes=torch.rand((batch_size, num_layers, num_groups, num_queries, 4), requires_grad=True),
            class_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, num_classes), requires_grad=True),
            edge_logits=torch.randn((batch_size, num_layers, num_groups, num_queries, 4 * (num_bins + 1)), requires_grad=True),
        )

        targets: List[Dict[str, Tensor]] = [
            {
                "labels": torch.empty(0, dtype=torch.int64),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
            }
        ]

        criterion = Criterion(loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0}, num_bins=num_bins)
        losses = criterion((decoder_preds, None, None), targets)

        assert losses["box"].requires_grad
        assert losses["giou"].requires_grad
        assert losses["localization"].requires_grad

        losses["overall"].backward()

        assert decoder_preds.boxes.grad is not None
        assert decoder_preds.edge_logits.grad is not None
        assert torch.count_nonzero(decoder_preds.boxes.grad) == 0
        assert torch.count_nonzero(decoder_preds.edge_logits.grad) == 0

    def test_criterion_invariant_to_micro_batch_split(self) -> None:
        """
        Losses summed over micro batches normalized by the whole batch should match the unsplit losses.
        """

        batch_size = 4
        num_layers = 2
        num_groups = 2
        num_queries = 10
        num_denoise_queries = 20
        num_classes = 4
        num_bins = 4

        def make_predictions(queries: int, groups: int) -> Predictions:
            return Predictions(
                boxes=torch.rand((batch_size, num_layers, groups, queries, 4)),
                class_logits=torch.randn((batch_size, num_layers, groups, queries, num_classes)),
                edge_logits=torch.randn((batch_size, num_layers, groups, queries, 4 * (num_bins + 1))),
            )

        def slice_predictions(predictions: Predictions, start: int, stop: int) -> Predictions:
            return Predictions(
                boxes=predictions.boxes[start:stop],
                class_logits=predictions.class_logits[start:stop],
                edge_logits=predictions.edge_logits[start:stop],
            )

        decoder_preds = make_predictions(num_queries, num_groups)
        encoder_preds = make_predictions(num_queries, num_groups)
        denoise_preds = make_predictions(num_denoise_queries, 1)

        # Uneven object counts, including an empty image, so the normalizers differ per micro batch
        targets: List[Dict[str, Tensor]] = []
        for num_objects in [2, 0, 3, 1]:
            targets.append(
                {
                    "labels": torch.randint(0, num_classes, (num_objects,)),
                    "boxes": torch.cat([torch.rand((num_objects, 2)) * 0.5 + 0.25, torch.rand((num_objects, 2)) * 0.2 + 0.05], dim=-1),
                }
            )

        criterion = Criterion(loss_weights={"class": 1.0, "box": 5.0, "giou": 2.0, "localization": 0.5}, num_bins=num_bins)

        expected = criterion((decoder_preds, encoder_preds, denoise_preds), targets)

        for micro_batch_size in [1, 2, 4]:
            accumulated: Dict[str, Tensor] = {}
            for start in range(0, batch_size, micro_batch_size):
                stop = start + micro_batch_size
                micro_losses = criterion(
                    (
                        slice_predictions(decoder_preds, start, stop),
                        slice_predictions(encoder_preds, start, stop),
                        slice_predictions(denoise_preds, start, stop),
                    ),
                    targets[start:stop],
                    normalizer_targets=targets,
                )
                accumulated = {name: accumulated.get(name, 0.0) + loss for name, loss in micro_losses.items()}

            for name, loss in expected.items():
                assert torch.allclose(
                    accumulated[name], loss, atol=1e-4
                ), f"Loss {name} differs at micro_batch_size={micro_batch_size}: {accumulated[name].item()} vs {loss.item()}"
