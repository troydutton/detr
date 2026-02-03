import logging
from typing import Any, Dict, Union

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from criterion import Criterion
from data import CocoDataset, collate_fn
from engine import evaluate
from evaluators import CocoEvaluator
from models import Model

device = "cuda:0"

Args = Dict[str, Union[Any, "Args"]]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Create dataset (config/dataset/*.yaml)
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    # Create dataloader
    batch_size, num_workers = args["train"]["batch_size"], args["train"]["num_workers"]
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # Ensure checkpoint is provided
    checkpoint = args["train"].get("checkpoint")
    if not checkpoint:
        raise ValueError("Please provide a checkpoint path via 'train.checkpoint=/path/to/checkpoint.pt'")

    # Create model (config/model/*.yaml)
    model: Model = instantiate(args["model"], num_classes=val_dataset.num_classes, pretrained_weights=checkpoint).to(device)

    # Create criterion (config/criterion/*.yaml)
    class_weights = val_dataset.calculate_class_weights(beta=args["criterion"].pop("beta"))
    criterion: Criterion = instantiate(args["criterion"], class_weights=class_weights)

    # Create evaluator
    evaluator: CocoEvaluator = CocoEvaluator(coco_targets=val_dataset.coco)

    # Start evaluation
    losses, metrics = evaluate(
        model=model,
        criterion=criterion,
        evaluator=evaluator,
        data=val_data,
        epoch=0,
        device=device,
    )

    logging.info(f"Evaluation Results for '{checkpoint}' on '{args['dataset']['val']['dataset_root']}':")
    logging.info("Losses: " + ", ".join([f"{k}={v:,}" for k, v in losses.items()]) + ".")
    logging.info("Metrics: " + ", ".join([f"{k}={v:,}" for k, v in metrics.items()]) + ".")


if __name__ == "__main__":
    main()
