import random
from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.ops.boxes import box_convert


def draw_annotations(
    image: Image.Image,
    boxes: Tensor,
    labels: List[str],
    scores: Tensor = None,
    color_map: Dict[str, Tuple[float, float, float]] = None,
    box_format: str = "cxcywh",
    *,
    normalized_boxes: bool = False,
) -> Image.Image:
    """
    Draws bounding boxes, labels, and scores on a PIL Image.

    Args:
        image: Image to draw on.
        boxes: Bounding boxes in `box_format` format.
        labels: List of class labels corresponding to the boxes.
        scores: List of confidence scores for each box.
        color_map: A dictionary mapping labels to specific colors. If None, random colors are assigned.
        box_format: Format of the bounding boxes ("xywh", "cxcywh", etc.).
        normalized_boxes: Whether the boxes are normalized to [0, 1].
    Returns:
        image: The image with drawn predictions.
    """

    # Begin drawing on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Convert boxes to xyxy
    boxes = box_convert(boxes, box_format, "xyxy")

    # Scale boxes by the image size
    if normalized_boxes:
        width, height = image.size

        boxes = boxes.clone() * torch.tensor([width, height, width, height])

    # Generate random colors if no color map is provided
    if color_map is None:
        color_map = {label: tuple(random.randint(0, 255) for _ in range(3)) for label in set(labels)}

    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i]

        # Get object color
        color = color_map[label]

        # Draw the bounding box
        box = box.tolist()

        draw.rectangle(box, outline=color, width=3)

        # Write label and confidence score
        text = label if scores is None else f"{label}: {scores[i].item():.2f}"
        left, top, right, bottom = font.getbbox(text)
        text_w = right - left
        text_h = bottom - top

        text_bg = [box[0], box[1] - text_h, box[0] + text_w, box[1]]

        draw.rectangle(text_bg, fill=color)
        draw.text((box[0], box[1] - text_h), text, fill=(255, 255, 255), font=font)

    return image
