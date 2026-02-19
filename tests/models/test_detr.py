import torch

from models import DETR


def test_detr_forward() -> None:
    """
    Test the forward pass of the DETR model.
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
            "name": "resnet50s",
            "embed_dim": embed_dim,
            "pretrained": False,
        },
        "encoder": {
            "num_layers": 1,
            "embed_dim": embed_dim,
            "layer": {
                "_target_": "models.encoder.EncoderLayer",
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
                "_target_": "models.decoder.DecoderLayer",
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
    output = model(images)

    # Check output content
    assert output.decoder.boxes is not None
    assert output.decoder.logits is not None

    # Check output shapes
    expected_logits_shape = (batch_size, num_decoder_layers, num_groups, num_queries, num_classes)
    assert (
        output.decoder.logits.shape == expected_logits_shape
    ), f"Expected logits shape {expected_logits_shape}, got {output.decoder.logits.shape}"

    expected_boxes_shape = (batch_size, num_decoder_layers, num_groups, num_queries, 4)
    assert (
        output.decoder.boxes.shape == expected_boxes_shape
    ), f"Expected boxes shape {expected_boxes_shape}, got {output.decoder.boxes.shape}"

    # Check that box coordinates are in [0, 1]
    assert output.decoder.boxes.min() >= 0.0
    assert output.decoder.boxes.max() <= 1.0
