import logging
from typing import Any, Dict, Union

import hydra
import numpy as np
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

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
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
    batch_size, num_workers = args["batch_size"], args["num_workers"]
    val_data = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Ensure pretrained weights are provided
    pretrained_weights = args["model"].get("pretrained_weights")
    if not pretrained_weights:
        raise ValueError("Please provide pretrained weights via 'model.pretrained_weights=/path/to/pretrained_weights'")

    # Create model (config/model/*.yaml)
    args["model"]["pretrained_weights"] = pretrained_weights
    args["model"]["categories"] = val_dataset.get_categories()
    args["model"]["decoder"]["num_classes"] = val_dataset.num_classes
    model = DETR(**args["model"])

    # Distribute evaluation components
    model, val_data = accelerator.prepare(model, val_data)

    # Create criterion (config/criterion/*.yaml)
    criterion: Criterion = instantiate(args["criterion"])

    # Create evaluator
    evaluator: CocoEvaluator = CocoEvaluator(coco_targets=val_dataset.coco, class_metrics=True)

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
        print(f"\n{' Evaluation Results ':=^80}")
        print(f"Weights: {pretrained_weights}")
        print(f"{' Losses ':-^80}")
        print(", ".join([f"{k}: {v:.2f}" for k, v in losses.items()]))

        overall_metrics = metrics.pop("overall")
        col_names = list(overall_metrics.keys())

        # Compute standard deviation for each metric across valid classes
        stds = {k: np.std([m[k] for m in metrics.values() if m[k] != -1]) if metrics else 0 for k in col_names}

        # Header
        print(f"{' Metrics ':-^80}")
        print(f"{'Class':<20} " + " ".join(f"{k:>6}" for k in col_names))
        print("-" * 80)

        # Overall metrics
        print(f"{'overall':<20} " + " ".join(f"{overall_metrics[k] * 100:>6.1f}" for k in col_names))

        # Class metrics
        for name, values in metrics.items():
            row = [f"{name:<20}"]
            for k in col_names:
                val = values[k]
                mean_val = overall_metrics[k]

                val_str = f"{val:>6.1f}" if val == -1 else f"{val * 100:>6.1f}"

                # Apply color logic: green if > mean + std, red if < mean - std
                if val != -1 and mean_val != 0 and stds[k] != 0:
                    if val > mean_val + stds[k]:
                        val_str = f"{GREEN}{val_str}{RESET}"
                    elif val < mean_val - stds[k]:
                        val_str = f"{RED}{val_str}{RESET}"

                row.append(val_str)

            print(" ".join(row))

        # Footer
        print("=" * 80)

    accelerator.end_training()


if __name__ == "__main__":
    main()
