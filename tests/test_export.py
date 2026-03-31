import json
from pathlib import Path

import pytest
from hydra import compose, initialize
from PIL import Image

from export import main as export_main


@pytest.fixture
def dummy_coco(tmp_path: Path) -> Path:
    """Creates a dummy COCO dataset for testing purposes."""
    root = tmp_path / "coco"
    root.mkdir()
    (root / "images").mkdir(parents=True, exist_ok=True)

    width, height = 100, 80
    image_id = 1
    file_name = "000000000001.jpg"

    image = Image.new("RGB", (width, height), color="red")
    image.save(root / "images" / file_name)

    annotations = {
        "images": [{"id": image_id, "file_name": file_name, "height": height, "width": width}],
        "annotations": [
            {"id": 1, "image_id": image_id, "category_id": 1, "bbox": [10, 20, 30, 40], "area": 1200, "iscrowd": 0},
            {"id": 2, "image_id": image_id, "category_id": 2, "bbox": [50, 10, 20, 30], "area": 600, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "dog"}],
        "info": {},
        "licenses": [],
    }
    with open(root / "_annotations.coco.json", "w") as f:
        json.dump(annotations, f)

    return root


@pytest.mark.parametrize("config_name", ["detr"])
def test_export(tmp_path: Path, dummy_coco: Path, config_name: str) -> None:
    output_dir = tmp_path / "output"

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"dataset.train.roots={dummy_coco}",
                f"dataset.val.roots={dummy_coco}",
                f"train.output_dir={output_dir}",
                "train.num_workers=0",
                "model.pretrained_weights=null",
                "transforms.val.resolution=64",
            ],
        )

    # Run the export script
    export_main(cfg)

    # Verify that the ONNX model was created at the expected location
    onnx_path = output_dir / "onnx" / f"{output_dir.name}.onnx"
    assert onnx_path.exists(), f"Expected ONNX file at {onnx_path}"
