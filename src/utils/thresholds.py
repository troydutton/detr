import logging
from typing import List, Tuple

import torch
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from rich.live import Live
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms, box_iou
from torchvision.ops.boxes import box_convert
from tqdm import tqdm

from models import DETR


@torch.no_grad()
def optimize_thresholds(
    model: DETR,
    data: DataLoader,
    accelerator: Accelerator,
    num_classes: int,
    confidence_range: List[float],
    iou_range: List[float],
) -> Tuple[float, float]:
    """
    Find the confidence and NMS IoU thresholds that maximize F1@50 on the provided data.

    Args:
        model: Model to optimize for.
        data: Data to optimize on.
        accelerator: Distributed accelerator.
        num_classes: Number of classes.
        confidence_range: [min, max, step] for confidence threshold search.
        iou_range: [min, max, step] for IoU threshold search.

    Returns:
        confidence_threshold: Optimal confidence threshold.
        #### iou_threshold
        Optimal IoU threshold.
    """

    logging.info(f"Optimizing thresholds with {confidence_range=} and {iou_range=}.")

    # Set the model to evaluation mode
    model.eval()

    prediction_boxes, prediction_scores, prediction_labels = [], [], []
    target_boxes, target_labels = [], []

    # Gather predictions and targets
    data = tqdm(data, desc="Running Inference", dynamic_ncols=True, disable=not accelerator.is_main_process)
    for images, targets in data:
        predictions = model(images)
        decoder_predictions, _, _ = predictions
        boxes = decoder_predictions.boxes[:, -1, 0]
        logits = decoder_predictions.class_logits[:, -1, 0]
        scores, labels = logits.sigmoid().max(dim=-1)

        boxes = box_convert(boxes, "cxcywh", "xyxy")

        for i in range(len(targets)):
            prediction_boxes.append(boxes[i].cpu().float())
            prediction_scores.append(scores[i].cpu().float())
            prediction_labels.append(labels[i].cpu())

            image_target_boxes = box_convert(targets[i]["boxes"], "cxcywh", "xyxy")
            image_target_labels = targets[i]["labels"]
            target_boxes.append(image_target_boxes.cpu().float())
            target_labels.append(image_target_labels.cpu())

    accelerator.wait_for_everyone()

    # Define the threshold search space
    start, stop, step = confidence_range
    confidence_thresholds = torch.arange(start, stop + step / 2, step, device=accelerator.device)
    start, stop, step = iou_range
    iou_thresholds = torch.arange(start, stop + step / 2, step, device=accelerator.device)

    # Store the results in a matrix for visualization, and keep track of the best thresholds
    results_matrix = [[None for _ in range(len(iou_thresholds))] for _ in range(len(confidence_thresholds))]
    best_f1, best_confidence, best_iou = -1.0, 0.5, 0.5
    current_step = 0

    def _make_table() -> Table:
        """Helper function to visualize the search results."""
        total_steps = len(confidence_thresholds) * len(iou_thresholds)
        progress = f"[{current_step}/{total_steps} {current_step / total_steps * 100:.1f}%]"
        title = f"Macro F1@50 {progress}"

        table = Table(title=title, header_style="cyan")
        table.add_column("Conf \\ IoU", justify="center", style="cyan", header_style="cyan")
        for iou_threshold in iou_thresholds:
            table.add_column(f"{iou_threshold.item():.2f}", justify="center", style="magenta")

        for row_index, confidence_threshold in enumerate(confidence_thresholds):
            row = [f"{confidence_threshold.item():.2f}"]
            for col_index in range(len(iou_thresholds)):
                f1 = results_matrix[row_index][col_index]

                if f1 is None:
                    row.append("[dim]·[/dim]")
                else:
                    if f1 == best_f1:
                        color = "bold white on green"
                    elif f1 >= 0.70:
                        color = "bold green"
                    elif f1 >= 0.50:
                        color = "green"
                    elif f1 >= 0.30:
                        color = "yellow"
                    else:
                        color = "red"
                    row.append(f"[{color}]{f1 * 100:.1f}[/]")
            table.add_row(*row)
        return table

    if accelerator.is_main_process:
        live = Live(_make_table(), transient=False)
        live.start()

    # Search over all combinations of confidence and IoU thresholds
    for row, confidence_threshold in enumerate(confidence_thresholds):
        confidence_threshold = confidence_threshold.item()
        for col, iou_threshold in enumerate(iou_thresholds):
            current_step += 1
            iou_threshold = iou_threshold.item()

            tp_per_class = torch.zeros(num_classes)
            fp_per_class = torch.zeros(num_classes)
            fn_per_class = torch.zeros(num_classes)

            for i in range(len(prediction_boxes)):
                image_boxes = prediction_boxes[i]
                image_scores = prediction_scores[i]
                image_labels = prediction_labels[i]

                image_target_boxes = target_boxes[i]
                image_target_labels = target_labels[i]

                # Apply confidence thresholding
                keep = image_scores > confidence_threshold
                image_boxes, image_scores, image_labels = image_boxes[keep], image_scores[keep], image_labels[keep]

                # Apply non-maximum suppression
                if len(image_boxes) > 0:
                    keep = batched_nms(image_boxes, image_scores, image_labels, iou_threshold)
                    image_boxes, image_scores, image_labels = image_boxes[keep], image_scores[keep], image_labels[keep]

                # Calculate TP, FP, FN for each class
                for class_label in range(num_classes):
                    class_boxes = image_boxes[image_labels == class_label]
                    class_scores = image_scores[image_labels == class_label]
                    class_target_boxes = image_target_boxes[image_target_labels == class_label]

                    if len(class_boxes) == 0 and len(class_target_boxes) == 0:  # No predictions and no targets, ignore class
                        continue
                    if len(class_boxes) == 0:  # No predictions, all targets are false negatives
                        fn_per_class[class_label] += len(class_target_boxes)
                        continue
                    if len(class_target_boxes) == 0:  # No targets, all predictions are false positives
                        fp_per_class[class_label] += len(class_boxes)
                        continue

                    # Greedily match predictions to targets based on IoU (higher confidence predictions get priority)
                    class_boxes = class_boxes[torch.argsort(class_scores, descending=True)]
                    ious = box_iou(class_boxes, class_target_boxes)

                    # Keep track of matched targets to avoid double counting
                    matched_targets = set()

                    tp, fp = 0, 0
                    for j in range(len(class_boxes)):
                        match_iou, match_index = ious[j].max(dim=0)
                        if match_iou >= 0.5:
                            target_idx = match_index.item()
                            if target_idx not in matched_targets:  # Match is free, true positive
                                matched_targets.add(target_idx)
                                tp += 1
                            else:  # Target already matched, false positive
                                fp += 1
                        else:  # No matching target, false positive
                            fp += 1

                    fn = len(class_target_boxes) - len(matched_targets)

                    tp_per_class[class_label] += tp
                    fp_per_class[class_label] += fp
                    fn_per_class[class_label] += fn

            # Reduce across processes
            tp_per_class = tp_per_class.to(accelerator.device)
            fp_per_class = fp_per_class.to(accelerator.device)
            fn_per_class = fn_per_class.to(accelerator.device)

            tp_per_class: Tensor = accelerator.reduce(tp_per_class, reduction="sum")
            fp_per_class: Tensor = accelerator.reduce(fp_per_class, reduction="sum")
            fn_per_class: Tensor = accelerator.reduce(fn_per_class, reduction="sum")

            if accelerator.is_main_process:
                # Compute average F1 across classes (macro F1)
                valid_classes = (tp_per_class + fp_per_class + fn_per_class) > 0
                precision = tp_per_class[valid_classes] / (tp_per_class[valid_classes] + fp_per_class[valid_classes] + 1e-6)
                recall = tp_per_class[valid_classes] / (tp_per_class[valid_classes] + fn_per_class[valid_classes] + 1e-6)
                f1_per_class = 2 * (precision * recall) / (precision + recall + 1e-6)
                f1 = f1_per_class.mean().item() if len(f1_per_class) > 0 else 0.0

                # Update best metrics
                if f1 > best_f1:
                    best_f1 = f1
                    best_confidence = confidence_threshold
                    best_iou = iou_threshold

                results_matrix[row][col] = f1

                live.update(_make_table())

    if accelerator.is_main_process:
        live.stop()

    # Send the best thresholds to all processes
    best_confidence, best_iou = broadcast_object_list([best_confidence, best_iou], from_process=0)

    logging.info(f"Optimal confidence threshold: {best_confidence:.2f}, Optimal IoU threshold: {best_iou:.2f}")

    return best_confidence, best_iou
