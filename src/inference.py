import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import imageio.v2 as imageio
import onnxruntime as ort
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from data import ImageDataset, make_normalize_transform
from models import DETR
from utils.checkpoint import load_metadata
from utils.postprocess import Detections, postprocess
from utils.visualize import draw_annotations, make_color_map

Args = Dict[str, Union[Any, "Args"]]


class PyTorchWrapper:
    def __init__(self, model: DETR, accelerator: Accelerator):
        self.model: DETR = accelerator.unwrap_model(model)
        self.model.eval()

    def predict(self, image: Tensor, confidence_threshold: float, iou_threshold: float) -> Detections:
        return self.model.predict(image, confidence_threshold, iou_threshold)[0]


class ONNXWrapper:
    def __init__(self, pretrained_weights: str, device: str):
        self.device = device
        self.session = ort.InferenceSession(pretrained_weights, providers=["CUDAExecutionProvider"])
        self.categories = json.loads(self.session.get_modelmeta().custom_metadata_map["categories"])

    def predict(self, images: Tensor, confidence_threshold: float, iou_threshold: float) -> Detections:
        ort_inputs = {self.session.get_inputs()[0].name: images.cpu().numpy()}
        outputs = self.session.run(None, ort_inputs)

        boxes = torch.from_numpy(outputs[0]).to(self.device)
        logits = torch.from_numpy(outputs[1]).to(self.device)

        return postprocess(boxes, logits, confidence_threshold, iou_threshold, self.categories)[0]


@hydra.main(config_path="../configs", config_name="inference", version_base=None)
def main(args: DictConfig) -> None:
    # Distributed setup
    accelerator = Accelerator()

    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.ERROR)

    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Create transformations, separating normalization so that we can visualize the images
    args["transforms"]["test"]["normalize"] = False
    transforms = instantiate(args["transforms"]["test"])
    normalize = make_normalize_transform()

    # Create dataset
    input_dir = args.get("input_dir")
    if not input_dir:
        raise ValueError("Please provide an input directory via 'input_dir=/path/to/images'")

    dataset = ImageDataset(root=input_dir, transforms=transforms)

    data = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args["num_workers"],
        collate_fn=lambda x: x[0],
    )

    # Ensure pretrained weights are provided
    pretrained_weights = args["model"].get("pretrained_weights")
    if not pretrained_weights:
        raise ValueError("Please provide pretrained weights via 'model.pretrained_weights=/path/to/weights'")

    # Create model
    pretrained_weights = Path(pretrained_weights)

    if not pretrained_weights.exists():
        raise FileNotFoundError(f"Pretrained weights not found at '{pretrained_weights}'")

    if pretrained_weights.suffix == ".safetensors" or pretrained_weights.is_dir():
        metadata = load_metadata(pretrained_weights)
        categories = json.loads(metadata["categories"])
        num_classes = len(categories)

        args["model"]["pretrained_weights"] = pretrained_weights
        args["model"]["decoder"]["num_classes"] = num_classes
        args["model"]["categories"] = categories

        model = DETR(**args["model"])
        model, data = accelerator.prepare(model, data)
        wrapper = PyTorchWrapper(model, accelerator)
    elif pretrained_weights.suffix == ".onnx":
        wrapper = ONNXWrapper(pretrained_weights, accelerator.device)
        categories = wrapper.categories
        data = accelerator.prepare(data)
    else:
        raise ValueError(f"Unsupported model extension: {pretrained_weights.suffix}")

    # Create color map for consistent annotation colors across images
    color_map = make_color_map(categories)

    # Create output directory
    output_dir = Path(args["output_dir"])
    images_dir = output_dir / "images"

    if accelerator.is_main_process:
        images_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving inference images to '{images_dir}'")

    accelerator.wait_for_everyone()

    # Inference loop
    confidence_threshold = args["confidence_threshold"]
    iou_threshold = args["iou_threshold"]

    data = tqdm(data, desc="Running Inference", dynamic_ncols=True, disable=not accelerator.is_main_process)
    for image, image_info in data:
        image = torch.unsqueeze(image, 0)

        detections = wrapper.predict(normalize(image), confidence_threshold, iou_threshold)

        # Save annotations
        image_path = Path(image_info["file_name"])

        annotated_image = draw_annotations(
            to_pil_image(image.squeeze(0).cpu()),
            detections.boxes.cpu(),
            detections.categories,
            detections.scores.cpu(),
            box_format="cxcywh",
            normalized_boxes=True,
            color_map=color_map,
        )

        annotated_image.save(images_dir / f"{image_path.stem}.jpg")

    accelerator.wait_for_everyone()

    logging.info(f"Saved inference images to '{images_dir}'")

    # Video creation
    if accelerator.is_main_process and args["make_video"]:
        video_path = output_dir / "video.mp4"
        logging.info(f"Saving inference video to '{video_path}'.")

        image_paths = sorted(list(images_dir.glob("*.jpg")))

        writer = imageio.get_writer(str(video_path), fps=args["fps"])

        image_paths = tqdm(image_paths, desc="Creating Video", dynamic_ncols=True)
        for image_path in image_paths:
            writer.append_data(imageio.imread(str(image_path)))

        writer.close()

        logging.info(f"Saved inference video to '{video_path}'.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
