import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import onnx
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import CocoDataset
from models import DETR

Args = Dict[str, Union[Any, "Args"]]


@hydra.main(config_path="../configs", config_name="export", version_base=None)
def main(args: DictConfig) -> None:
    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Fetch an image from the validation set for tracing
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    image, _ = val_dataset[0]
    image = image.unsqueeze(0)

    # Ensure pretrained weights are provided, notifying the user if not specified
    pretrained_weights = args["model"].get("pretrained_weights")
    if not pretrained_weights:
        logging.warning("Please provide pretrained weights via 'model.pretrained_weights=/path/to/pretrained_weights'")

    # Create model (config/model/*.yaml)
    args["model"]["pretrained_weights"] = pretrained_weights
    args["model"]["categories"] = val_dataset.get_categories()
    args["model"]["decoder"]["num_classes"] = val_dataset.num_classes
    args["model"]["decoder"]["num_groups"] = 1
    args["model"]["decoder"]["denoise_queries"] = False
    model = DETR(**args["model"]).eval()

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
        verbose=False,
    )

    # Add categories as custom metadata
    onnx_model = onnx.load(output_path)
    onnx.helper.set_model_props(onnx_model, {"categories": json.dumps(val_dataset.get_categories())})
    onnx.save(onnx_model, output_path)

    logging.info(f"Exported model to '{output_path}'.")


if __name__ == "__main__":
    main()
