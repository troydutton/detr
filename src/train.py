import logging
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import wandb
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from criterion import Criterion
from data import CocoDataset, collate_fn
from engine import train
from evaluators import CocoEvaluator
from models import Model
from utils.lr import prepare_scheduler_arguments
from utils.optimizer import build_parameter_groups

Args = Dict[str, Union[Any, "Args"]]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    # Distributed setup
    accelerator = Accelerator()
    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.ERROR)

    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Save the arguments to the output directory
    output_dir = Path(args["train"]["output_dir"])

    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=args, f=output_dir / "config.yaml")

        # Initialize Weights & Biases
        wandb.init(project="detr", name=output_dir.name, config=args)

    accelerator.wait_for_everyone()

    # Now that we've saved the arguments, we can remove unecessary keys
    del args["train"]["output_dir"]

    # Create datasets (config/dataset/*.yaml)
    train_dataset: CocoDataset = instantiate(args["dataset"]["train"])
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    # Create dataloaders
    batch_size, num_workers = args["train"].pop("batch_size"), args["train"].pop("num_workers")
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # Create model (config/model/*.yaml)
    model: Model = instantiate(args["model"], num_classes=train_dataset.num_classes)

    # Create optimizer (config/optimizer/*.yaml)
    lr, lr_backbone = args["optimizer"]["lr"], args["optimizer"].pop("lr_backbone")
    args["optimizer"]["params"] = build_parameter_groups(model, lr=lr, lr_backbone=lr_backbone)
    optimizer: Optimizer = instantiate(args["optimizer"], _convert_="all")

    # Create learning rate scheduler (config/scheduler/*.yaml)
    logging.info(f"Preparing scheduler with {len(train_data)=} and {accelerator.num_processes=}")
    args["scheduler"] = prepare_scheduler_arguments(args["scheduler"], steps_per_epoch=len(train_data) * accelerator.num_processes)
    scheduler: _LRScheduler = instantiate(args["scheduler"], optimizer=optimizer)

    # Distribute the components
    model, optimizer, train_data, val_data, scheduler = accelerator.prepare(model, optimizer, train_data, val_data, scheduler)

    # Create criterion (config/criterion/*.yaml)
    class_weights = train_dataset.calculate_class_weights(beta=args["criterion"].pop("beta"))
    class_weights = class_weights.to(accelerator.device)
    criterion: Criterion = instantiate(args["criterion"], class_weights=class_weights)

    # Create evaluator
    evaluator: CocoEvaluator = CocoEvaluator(coco_targets=val_dataset.coco)

    # Load from checkpoint if provided
    checkpoint = args["train"].pop("checkpoint", None)

    if checkpoint is not None:
        accelerator.load_state(checkpoint)

    # Start training
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        evaluator=evaluator,
        train_data=train_data,
        val_data=val_data,
        accelerator=accelerator,
        output_dir=output_dir,
        **args["train"],
    )


if __name__ == "__main__":
    main()
