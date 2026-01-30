from pathlib import Path

import hydra
import wandb
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
from utils.optimizer import get_parameter_groups

device = "cuda:0"


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    # Resolve arguments
    args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)

    # Initialize Weights & Biases
    output_dir = Path(args["train"]["output_dir"])
    wandb.init(project="detr", name=output_dir.name, config=args)

    # Create datasets (config/dataset/*.yaml)
    train_dataset: CocoDataset = instantiate(args["dataset"]["train"])
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    # Create dataloaders
    batch_size, num_workers = args["train"]["batch_size"], args["train"]["num_workers"]
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    # Create model (config/model/*.yaml)
    model: Model = instantiate(args["model"], num_classes=train_dataset.num_classes).to(device)

    # Create optimizer (config/optimizer/*.yaml)
    lr, lr_backbone = args["optimizer"].pop("lr"), args["optimizer"].pop("lr_backbone")
    args["optimizer"]["params"] = get_parameter_groups(model, lr=lr, lr_backbone=lr_backbone)
    optimizer: Optimizer = instantiate(args["optimizer"], _convert_="all")

    # Create learning rate scheduler (config/scheduler/*.yaml)
    scheduler: _LRScheduler = instantiate(args["scheduler"], optimizer=optimizer)

    # Create criterion (config/criterion/*.yaml)
    class_weights = train_dataset.calculate_class_weights(beta=args["criterion"].pop("beta"))
    criterion: Criterion = instantiate(args["criterion"], class_weights=class_weights)

    # Create evaluator
    evaluator: CocoEvaluator = CocoEvaluator(coco_targets=val_dataset.coco)

    # Start training
    num_epochs, save_period = args["train"]["num_epochs"], args["train"]["save_period"]
    train(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        evaluator=evaluator,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        output_dir=output_dir,
        device=device,
        save_period=save_period,
    )


if __name__ == "__main__":
    main()
