import torch

from models import DETR
from models.detr import Detections


def test_detr_forward() -> None:
    """
    Test the forward pass of the DETR model.

    Instantiates a mock DETR architecture and tests the shapes and formats
    of the boxes and logits outputted in a forward pass.
    """
    # Test parameters
    batch_size = 2

    # Model arguments
    num_classes = 10
    embed_dim = 64
    num_decoder_layers = 2
    num_queries = 5
    num_groups = 1

    kwargs = {
        "embed_dim": embed_dim,
        "backbone": {
            "feature_extractor": {
                "name": "facebook/dinov2-with-registers-small",
                "out_feature_indices": [2, 5, 8, 11],
                "window_layer_indices": [0, 1, 3, 4, 6, 7, 9, 10],
                "num_windows": 2,
            },
            "projector": {"embed_dim": embed_dim, "multi_scale": True, "out_strides": [16], "num_blocks": 3},
            "embed_dim": embed_dim,
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
            "layer": {
                "_target_": "models.layers.decoder.DecoderLayer",
                "embed_dim": embed_dim,
                "ffn_dim": 128,
                "num_heads": 4,
                "dropout": 0.0,
            },
        },
    }

    model = DETR(**kwargs)

    # Create dummy input
    images = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    output, encoder_output, denoise_output = model(images)

    # Check output content
    assert output.boxes is not None
    assert output.logits is not None

    # Check output shapes
    expected_logits_shape = (batch_size, num_decoder_layers, num_groups, num_queries, num_classes)
    assert output.logits.shape == expected_logits_shape, f"Expected logits shape {expected_logits_shape}, got {output.logits.shape}"

    expected_boxes_shape = (batch_size, num_decoder_layers, num_groups, num_queries, 4)
    assert output.boxes.shape == expected_boxes_shape, f"Expected boxes shape {expected_boxes_shape}, got {output.boxes.shape}"

    # Check that box coordinates are in [0, 1]
    assert output.boxes.min() >= 0.0
    assert output.boxes.max() <= 1.0


def test_detr_predict() -> None:
    """
    Test the predict method of the DETR model.

    Instantiates a mock DETR architecture and tests the shapes and formats
    of the predicted Detections objects.
    """
    # Test parameters
    batch_size = 2

    # Model arguments
    num_classes = 10
    embed_dim = 64
    num_decoder_layers = 2
    num_queries = 5
    num_groups = 1

    kwargs = {
        "embed_dim": embed_dim,
        "backbone": {
            "feature_extractor": {
                "name": "facebook/dinov2-with-registers-small",
                "out_feature_indices": [2, 5, 8, 11],
                "window_layer_indices": [0, 1, 3, 4, 6, 7, 9, 10],
                "num_windows": 2,
            },
            "projector": {"embed_dim": embed_dim, "multi_scale": True, "out_strides": [16], "num_blocks": 3},
            "embed_dim": embed_dim,
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
            "layer": {
                "_target_": "models.layers.decoder.DecoderLayer",
                "embed_dim": embed_dim,
                "ffn_dim": 128,
                "num_heads": 4,
                "dropout": 0.0,
            },
        },
    }

    model = DETR(**kwargs)

    # Test batched image input
    batched_images = torch.randn(batch_size, 3, 224, 224)
    batched_output = model.predict(batched_images, confidence_threshold=0.0)

    assert isinstance(batched_output, list)
    assert len(batched_output) == batch_size
    for output in batched_output:
        assert isinstance(output, Detections)
        assert isinstance(output.boxes, torch.Tensor)
        assert isinstance(output.labels, torch.Tensor)
        assert isinstance(output.scores, torch.Tensor)
        assert output.boxes.ndim == 2 and output.boxes.shape[1] == 4
        assert output.labels.ndim == 1
        assert output.scores.ndim == 1
        assert output.categories is None
