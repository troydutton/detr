import json
import logging
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import onnx
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import CocoDataset, collate_fn
from models import DETR
from utils.thresholds import optimize_thresholds

Args = Dict[str, Union[Any, "Args"]]


@hydra.main(config_path="../configs", config_name="export", version_base=None)
def main(args: DictConfig) -> None:
    # Distributed setup
    accelerator = Accelerator()

    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.ERROR)

    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Fetch an image from the validation set for tracing
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    image, _ = val_dataset[0]
    image = image.unsqueeze(0)

    # Warn the user if pretrained weights were not provided
    pretrained_weights = args["model"].get("pretrained_weights")
    if not pretrained_weights:
        logging.warning("Please provide pretrained weights via 'model.pretrained_weights=/path/to/pretrained_weights'")

    # Create model (config/model/*.yaml)
    args["model"]["pretrained_weights"] = pretrained_weights
    args["model"]["categories"] = val_dataset.get_categories()
    args["model"]["decoder"]["num_classes"] = val_dataset.num_classes
    args["model"]["decoder"]["num_groups"] = 1
    args["model"]["decoder"]["denoise_queries"] = False
    if "num_inference_queries" in args["model"]:
        args["model"]["decoder"]["num_queries"] = args["model"]["num_inference_queries"]
    model = DETR(**args["model"]).eval()

    if args.get("optimize_thresholds", False):
        # Maintain a clean copy of the model for export
        export_model = deepcopy(model)

        # Create dataloader
        batch_size, num_workers = args["batch_size"], args["num_workers"]
        val_data = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        model, val_data = accelerator.prepare(model, val_data)

        best_conf, best_iou = optimize_thresholds(
            model=model,
            data=val_data,
            accelerator=accelerator,
            num_classes=val_dataset.num_classes,
            confidence_range=args["confidence_thresholds"],
            iou_range=args["iou_thresholds"],
        )

        model = export_model
    else:
        best_conf = args.get("confidence_threshold", 0.5)
        best_iou = args.get("iou_threshold", 0.5)

    if accelerator.is_main_process:
        # Override forward with an export-friendly predict
        model.forward = lambda x: model.predict(x, export=True)

        # Create output directory and path
        output_dir = Path(args["output_dir"]) / "onnx"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_dir.parent.name}.onnx"

        logging.info(f"Exporting model to '{output_path}'.")

        # Suppress ONNX warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="copyreg")
        for logger_name in ["torch.onnx", "onnxscript", "onnx_ir"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
            for name in list(logging.Logger.manager.loggerDict.keys()):
                if name.startswith(logger_name):
                    logging.getLogger(name).setLevel(logging.ERROR)

        # Export the model
        torch.onnx.export(
            model,
            image,
            output_path,
            export_params=True,
            opset_version=args["opset_version"],
            do_constant_folding=True,
            input_names=["images"],
            output_names=["boxes", "logits"],
            external_data=False,
            verbose=False,
        )

        # Add categories and optimal thresholds as custom metadata
        onnx_model = onnx.load(output_path)
        metadata = {
            "categories": json.dumps(val_dataset.get_categories()),
            "confidence_threshold": str(best_conf),
            "iou_threshold": str(best_iou),
        }
        onnx.helper.set_model_props(onnx_model, metadata)
        onnx.save(onnx_model, output_path)

        logging.info(f"Exported model to '{output_path}'.")

        accelerator.end_training()


if __name__ == "__main__":
    main()
