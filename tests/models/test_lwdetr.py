import torch

from models import DETR


def test_lwdetr_forward() -> None:
    """
    Test the forward pass of the LW-DETR model.
    """
    # Test parameters
    batch_size = 2

    # Model arguments
    num_classes = 10
    embed_dim = 64
    num_decoder_layers = 2
    num_queries = 5
    num_groups = 1
    num_denoise_queries = 10

    kwargs = {
        "embed_dim": embed_dim,
        "backbone": {
            "name": "convnext_tiny",
            "embed_dim": embed_dim,
            "num_levels": 4,
            "enable_projector": True,
            "projector": {"embed_dim": embed_dim, "out_strides": [8], "num_blocks": 1},
            "pretrained": False,
        },
        "encoder": {
            "num_layers": 0,
            "embed_dim": embed_dim,
            "layer": {
                "_target_": "models.layers.encoder.DeformableEncoderLayer",
                "embed_dim": embed_dim,
                "ffn_dim": 128,
                "num_heads": 4,
                "num_points": 4,
                "num_levels": 4,
                "dropout": 0.0,
            },
        },
        "decoder": {
            "num_layers": num_decoder_layers,
            "embed_dim": embed_dim,
            "num_queries": num_queries,
            "num_classes": num_classes,
            "num_groups": num_groups,
            "two_stage": True,
            "refine_boxes": True,
            "denoise_queries": True,
            "num_denoise_queries": num_denoise_queries,
            "layer": {
                "_target_": "models.layers.decoder.DeformableDecoderLayer",
                "embed_dim": embed_dim,
                "ffn_dim": 128,
                "num_heads": 4,
                "num_deformable_heads": 4,
                "num_points": 4,
                "num_levels": 4,
                "dropout": 0.0,
            },
        },
    }

    model = DETR(**kwargs)

    # Create dummy input
    images = torch.randn(batch_size, 3, 224, 224)

    targets = [{"labels": torch.zeros(3, dtype=torch.long), "boxes": torch.randn(3, 4).clip(0, 1)} for _ in range(batch_size)]

    # Forward pass
    output, encoder_output, denoise_output = model(images, targets)

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
