from pathlib import Path

from accelerate import Accelerator
from torch import nn

from utils.checkpoint import load_checkpoint, save_checkpoint


def test_training_state_round_trip(tmp_path: Path) -> None:
    accelerator = Accelerator()
    accelerator.prepare(nn.Linear(2, 2))

    save_checkpoint(accelerator, tmp_path, epoch=12, step=8844, images=141504)

    assert load_checkpoint(accelerator, tmp_path) == {"epoch": 12, "step": 8844, "images": 141504}
