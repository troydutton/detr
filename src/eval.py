import logging
from typing import Any, Dict, Union

import hydra
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from criterion import Criterion
from data import CocoDataset, collate_fn
from engine import evaluate
from evaluators import CocoEvaluator
from models import DETR

Args = Dict[str, Union[Any, "Args"]]


@hydra.main(config_path="../configs", config_name="detr", version_base=None)
def main(args: DictConfig) -> None:
    # Distributed setup
    accelerator = Accelerator()

    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.ERROR)

    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Create dataset (config/dataset/*.yaml)
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    # Create dataloader
    batch_size, num_workers = args["train"]["batch_size"], args["train"]["num_workers"]
    val_data = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Ensure checkpoint is provided
    checkpoint = args["train"].get("checkpoint")
    if not checkpoint:
        raise ValueError("Please provide a checkpoint path via 'train.checkpoint=/path/to/checkpoint'")

    # Create model (config/model/*.yaml)
    args["model"]["pretrained_weights"] = checkpoint
    args["model"]["categories"] = val_dataset.get_categories()
    args["model"]["decoder"]["num_classes"] = val_dataset.num_classes
    model = DETR(**args["model"])

    # Distribute evaluation components
    model, val_data = accelerator.prepare(model, val_data)

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
        accelerator=accelerator,
    )

    # Log results
    if accelerator.is_main_process:
        header = " Evaluation Results "
        width = max(len(header), 80)

        print(f"\n{header:=^{width}}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Dataset: {val_dataset.root / val_dataset.split}")
        print(f"{' Losses ':-^{width}}")
        print(", ".join([f"{k}: {v:.2f}" for k, v in losses.items()]))
        print(f"{' Metrics ':-^{width}}")
        print(", ".join([f"{k}: {v:.2f}" for k, v in metrics.items()]))
        print("=" * width)


if __name__ == "__main__":
    main()
