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
        header = " Evaluation Results "
        width = max(len(header), 80)

        print(f"\n{header:=^{width}}")
        print(f"Weights: {pretrained_weights}")
        print(f"{' Losses ':-^{width}}")
        print(", ".join([f"{k}: {v:.2f}" for k, v in losses.items()]))

        overall_metrics = metrics.pop("overall")
        col_names = list(overall_metrics.keys())

        # Compute standard deviation for each metric across valid classes
        stds = {}
        for k in col_names:
            valid_vals = [m[k] for m in metrics.values() if m[k] != -1]
            stds[k] = np.std(valid_vals) if valid_vals else 0

        col_widths = [max(len(name), 6) for name in col_names]
        header_fmt = "{:<20} " + " ".join([f"{{:>{w}}}" for w in col_widths])
        row_fmt = "{:<20} " + " ".join([f"{{:>{w}.2f}}" for w in col_widths])

        # Header
        print(f"{' Metrics ':-^{width}}")
        print(header_fmt.format("Class", *col_names))
        print("-" * width)

        # Overall metrics
        print(row_fmt.format("overall", *[overall_metrics[k] for k in col_names]))

        # Class metrics
        for name, values in metrics.items():
            row = [f"{name:<20}"]
            for k, w in zip(col_names, col_widths):
                val = values[k]
                mean_val = overall_metrics[k]
                val_str = f"{val:>{w}.2f}"

                if mean_val == 0 or stds[k] == 0 or val == -1:
                    row.append(val_str)
                else:
                    if val > mean_val + stds[k]:
                        color = GREEN
                    elif val < mean_val - stds[k]:
                        color = RED
                    else:
                        color = ""

                    color_end = RESET if color else ""
                    row.append(f"{color}{val_str}{color_end}")

            print(" ".join(row))

        # Footer
        print("=" * width)

    accelerator.end_training()


if __name__ == "__main__":
    main()
