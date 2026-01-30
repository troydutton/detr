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
    num_queries = 5

    kwargs = {
        "backbone": {
            "name": "resnet50s",
            "pretrained": False,
        },
        "transformer": {
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "embed_dim": embed_dim,
            "ffn_dim": 128,
            "num_heads": 4,
            "num_queries": num_queries,
            "dropout": 0.0,
        },
        "num_classes": num_classes,
    }

    model = DETR(**kwargs)

    # Create dummy input
    images = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    output = model(images)

    # Check output keys
    assert "boxes" in output
    assert "logits" in output

    # Check output shapes
    expected_logits_shape = (batch_size, num_queries, num_classes)
    assert output["logits"].shape == expected_logits_shape, f"Expected logits shape {expected_logits_shape}, got {output['logits'].shape}"

    expected_boxes_shape = (batch_size, num_queries, 4)
    assert output["boxes"].shape == expected_boxes_shape, f"Expected boxes shape {expected_boxes_shape}, got {output['boxes'].shape}"

    # Check that box coordinates are in [0, 1]
    assert output["boxes"].min() >= 0.0
    assert output["boxes"].max() <= 1.0
