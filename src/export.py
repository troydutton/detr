import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import CocoDataset
from models import DETR

Args = Dict[str, Union[Any, "Args"]]


@hydra.main(config_path="../configs", config_name="detr", version_base=None)
def main(args: DictConfig) -> None:
    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Fetch an image from the validation set for tracing
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    image, _ = val_dataset[0]
    image = image.unsqueeze(0)

    # Ensure checkpoint is provided, otherwise export an untrained model with a warning
    checkpoint = args["train"].get("checkpoint")
    if not checkpoint:
        logging.warning("No checkpoint provided, exporting untrained model. Specify with +train.checkpoint=<path>.")

    # Create the model
    args["model"]["pretrained_weights"] = checkpoint
    args["model"]["categories"] = val_dataset.get_categories()
    args["model"]["decoder"]["num_classes"] = val_dataset.num_classes
    model = DETR(**args["model"]).eval()

    # Override forward with an export-friendly predict
    model.forward = lambda x: model.predict(x, export=True)

    # Create output directory and path
    output_dir = Path(args["train"]["output_dir"]) / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_dir.parent.name}.onnx"

    logging.info(f"Exporting model to '{output_path}'.")

    # Determine opset version, notifying the user if not specified
    opset_version = args.get("opset_version", 18)

    if "opset_version" not in args:
        logging.info(f"No opset version specified, defaulting to {opset_version}. Specify with +opset_version=<version>.")

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
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["boxes", "logits"],
        verbose=False,
    )

    logging.info(f"Exported model to '{output_path}'.")


if __name__ == "__main__":
    main()
