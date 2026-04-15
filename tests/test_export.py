import json
from pathlib import Path

import pytest
from omegaconf import DictConfig
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


def test_export(tmp_path: Path, dummy_coco: Path) -> None:
    output_dir = tmp_path / "output"

    embed_dim = 64
    num_classes = 10
    num_decoder_layers = 1
    num_queries = 5
    num_groups = 1

    cfg = DictConfig(
        {
            "dataset": {
                "val": {
                    "_target_": "data.CocoDataset",
                    "roots": [str(dummy_coco)],
                    "annotation_name": "_annotations.coco.json",
                    "image_directory": "images",
                    "transforms": {"_target_": "data.make_transformations", "split": "val", "resolution": 64},
                }
            },
            "model": {
                "embed_dim": embed_dim,
                "backbone": {
                    "embed_dim": embed_dim,
                    "feature_extractor": {
                        "name": "facebook/dinov2-with-registers-small",
                        "image_size": 512,
                        "patch_size": 16,
                        "out_feature_indices": [2, 5, 8, 11],
                        "window_layer_indices": [0, 1, 3, 4, 6, 7, 9, 10],
                        "num_windows": 2,
                    },
                    "projector": {"embed_dim": embed_dim, "multi_scale": True, "out_strides": [16], "num_blocks": 3},
                    "pretrained": False,
                },
                "encoder": {
                    "num_layers": 1,
                    "embed_dim": embed_dim,
                    "layer": {
                        "_target_": "models.layers.encoder.EncoderLayer",
                        "embed_dim": embed_dim,
                        "ffn_dim": 128,
                        "num_heads": 4,
                        "dropout": 0.0,
                    },
                },
                "decoder": {
                    "num_layers": num_decoder_layers,
                    "embed_dim": embed_dim,
                    "num_queries": num_queries,
                    "num_classes": num_classes,
                    "num_groups": num_groups,
                    "denoise_queries": False,
                    "layer": {
                        "_target_": "models.layers.decoder.DecoderLayer",
                        "embed_dim": embed_dim,
                        "ffn_dim": 128,
                        "num_heads": 4,
                        "dropout": 0.0,
                    },
                },
                "pretrained_weights": None,
            },
            "train": {"output_dir": str(output_dir)},
            "opset_version": 18,
        }
    )

    # Run the export script
    export_main(cfg)

    # Verify that the ONNX model was created at the expected location
    onnx_path = output_dir / "onnx" / f"{output_dir.name}.onnx"
    assert onnx_path.exists(), f"Expected ONNX file at {onnx_path}"
