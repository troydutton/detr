from typing import Any, Dict, Union

import hydra
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from data import CocoDataset
from models import DETR

Args = Dict[str, Union[Any, "Args"]]

WARMUP_STEPS = 100
ACTIVE_STEPS = 1000


@hydra.main(config_path="../configs", config_name="detr", version_base=None)
def main(args: DictConfig) -> None:
    # Resolve arguments
    args: Args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    device = torch.device("cuda")

    # Create dataset (config/dataset/*.yaml)
    val_dataset: CocoDataset = instantiate(args["dataset"]["val"])

    # Get a sample image
    image = val_dataset[0][0].unsqueeze(0).to(device)
    _, _, height, width = image.shape

    # Create model (config/model/*.yaml)
    args["model"]["categories"] = val_dataset.get_categories()
    args["model"]["decoder"]["num_classes"] = val_dataset.num_classes
    args["model"]["decoder"]["num_groups"] = 1
    args["model"]["decoder"]["denoise_queries"] = False
    model = DETR(**args["model"]).eval().to(device)
    model.forward = lambda x: model.predict(x, export=True)

    print("=" * 60)
    print(f"Benchmarking DETR with resolution {width}x{height}")
    print(f"Device: {device}")
    print("=" * 60)

    # Compute FLOPs
    print("\nComputing FLOPs...")
    flops_analyzer = FlopCountAnalysis(model, image)
    flops = flops_analyzer.total()

    print("\nFLOPs Table:")
    print(flop_count_table(flops_analyzer, show_param_shapes=False))

    # Compute Latency
    print(f"\nComputing average latency ({WARMUP_STEPS} warmup steps, {ACTIVE_STEPS} active steps)...")

    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(image)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        starter.record()
        for _ in range(ACTIVE_STEPS):
            _ = model(image)
        ender.record()

        torch.cuda.synchronize()
        total_time_ms = starter.elapsed_time(ender)

    avg_latency_ms = total_time_ms / ACTIVE_STEPS
    fps = 1000.0 / avg_latency_ms
    num_params = sum(p.numel() for p in model.parameters()) / 1e6

    print("=" * 60)
    print(f"Parameters: {num_params:.2f} M")
    print(f"Operations: {flops / 1e9:.2f} GFLOPs")
    print(f"Latency:    {avg_latency_ms:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    print("=" * 60)


if __name__ == "__main__":
    main()
