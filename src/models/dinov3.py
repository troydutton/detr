import logging
from math import log, pi
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import DINOv3ViTConfig
from transformers import DINOv3ViTModel as HFDINOv3ViTModel
from transformers.activations import ACT2FN

from utils.misc import take_annotation_from

from .dinov2 import Dinov2WithRegistersDropPath, Dinov2WithRegistersLayerScale

transformers.utils.logging.disable_progress_bar()


class DINOv3ViTEmbeddings(nn.Module):
    """
    Constructs the CLS token, register tokens, and patch embeddings.

    Args:
        config: Configuration object containing model hyperparameters.
    """

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)

    def forward(self, images: Tensor) -> Tensor:
        # Get batch information
        batch_size, num_channels, height, width = images.shape
        patch_height, patch_width = self.patch_embeddings.kernel_size

        # Patchify the images (as a linear layer, which is faster than a strided convolution in TensorRT)
        patches = images.view(batch_size, num_channels, height // patch_height, patch_height, width // patch_width, patch_width)
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, -1, num_channels * patch_height * patch_width)

        patch_embeddings = F.linear(
            patches.to(dtype=self.patch_embeddings.weight.dtype), self.patch_embeddings.weight.flatten(1), self.patch_embeddings.bias
        )

        # Prepend the CLS and register tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        return torch.cat((cls_tokens, register_tokens, patch_embeddings), dim=1)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DINOv3ViTRopePositionEmbedding(nn.Module):
    """
    2D rotary positional embeddings for the patch tokens.

    Args:
        config: Configuration object containing model hyperparameters.
    """

    inv_freq: Tensor

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()

        self.pos_embed_rescale = config.pos_embed_rescale

        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1 / config.rope_theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, num_patches_height: int, num_patches_width: int) -> Tuple[Tensor, Tensor]:
        """Compute the cosine and sine of the rotation angles for each patch, with shape (num_patches, head_dim)."""

        device = self.inv_freq.device

        # Patch centers in YX normalized to [-1, 1]
        y = torch.arange(0.5, num_patches_height, device=device) / num_patches_height
        x = torch.arange(0.5, num_patches_width, device=device) / num_patches_width
        patch_coords = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1).flatten(0, 1)
        patch_coords = 2.0 * patch_coords - 1.0

        # Randomly rescale the coordinates during training
        if self.training and self.pos_embed_rescale is not None:
            log_rescale = log(self.pos_embed_rescale)
            patch_coords = patch_coords * torch.empty(1, device=device).uniform_(-log_rescale, log_rescale).exp()

        angles = 2 * pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]
        angles = angles.flatten(1, 2).tile(2)

        return torch.cos(angles), torch.sin(angles)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DINOv3ViTAttention(nn.Module):
    """
    Multi-head self-attention with rotary positional embeddings.

    Args:
        config: Configuration object containing model hyperparameters.
    """

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout_prob = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.query_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.key_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.value_bias)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.proj_bias)

    def forward(self, embeddings: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        # Get batch information
        batch_size, seq_length, embed_dim = embeddings.shape

        # Project the queries, keys, and values with a single matrix multiplication (scaling the queries here keeps TensorRT's attention fused)
        scale = self.head_dim**-0.5
        query_bias, key_bias, value_bias = [
            projection.bias if projection.bias is not None else torch.zeros_like(projection.weight[:, 0])
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        qkv_weight = torch.cat((self.q_proj.weight * scale, self.k_proj.weight, self.v_proj.weight))
        qkv_bias = torch.cat((query_bias * scale, key_bias, value_bias))

        # Reshape for multi-head attention (3, batch_size, num_heads, seq_length, head_dim)
        qkv = F.linear(embeddings, qkv_weight, qkv_bias)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # Rotate the queries and keys together in their own dtype (special tokens have an identity rotation)
        qk, v = qkv.split((2, 1))
        qk = qk * rope_cos.to(qk.dtype)[:, None] + _rotate_half(qk) * rope_sin.to(qk.dtype)[:, None]

        # Perform attention
        attention_output = F.scaled_dot_product_attention(
            qk[0], qk[1], v[0], dropout_p=self.dropout_prob if self.training else 0.0, scale=1.0
        )

        # Restore original shape
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_length, embed_dim)

        return self.o_proj(attention_output)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


def _rotate_half(features: Tensor) -> Tensor:
    first_half, second_half = features.chunk(2, dim=-1)

    return torch.cat((-second_half, first_half), dim=-1)


class DINOv3ViTMLP(nn.Module):
    """
    Standard feedforward network with one hidden layer.

    Args:
        config: Configuration object containing model hyperparameters.
    """

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()

        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.up_proj(embeddings)))

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DINOv3ViTGatedMLP(nn.Module):
    """
    Gated feedforward network with one hidden layer.

    Args:
        config: Configuration object containing model hyperparameters.
    """

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()

        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.down_proj(self.activation(self.gate_proj(embeddings)) * self.up_proj(embeddings))

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DINOv3ViTLayer(nn.Module):
    """
    Single layer of the backbone, consisting of self-attention and a feedforward network.

    Args:
        config: Configuration object containing model hyperparameters.
    """

    def __init__(self, config: DINOv3ViTConfig) -> None:
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DINOv3ViTAttention(config)
        self.layer_scale1 = Dinov2WithRegistersLayerScale(config)
        self.drop_path = Dinov2WithRegistersDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        # Feedforward network
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = DINOv3ViTGatedMLP(config) if config.use_gated_mlp else DINOv3ViTMLP(config)
        self.layer_scale2 = Dinov2WithRegistersLayerScale(config)

    def forward(self, embeddings: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        # Self-attention
        embeddings = embeddings + self.drop_path(self.layer_scale1(self.attention(self.norm1(embeddings), rope_cos, rope_sin)))

        # Feedforward network
        embeddings = embeddings + self.drop_path(self.layer_scale2(self.mlp(self.norm2(embeddings))))

        return embeddings

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class DINOv3ViTEncoder(nn.Module):
    """
    Sequential vision encoder.

    Args:
        config: Configuration object containing model hyperparameters.
        out_feature_indices: Layer indices to output features from.
        window_layer_indices: Layer indices to apply windowed attention to (if num_windows > 1).
        num_windows: Number of windows to use for windowed attention, optional.
    """

    def __init__(
        self,
        config: DINOv3ViTConfig,
        out_feature_indices: List[int],
        window_layer_indices: List[int],
        num_windows: int = 1,
    ) -> None:
        super().__init__()

        self.out_feature_indices = out_feature_indices
        self.window_layer_indices = window_layer_indices
        self.num_windows = num_windows

        self.layer = nn.ModuleList([DINOv3ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, embeddings: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> List[Tensor]:
        # Get batch information
        windowed_batch_size, _, embed_dim = embeddings.shape
        batch_size = windowed_batch_size // (self.num_windows**2)
        head_dim = rope_cos.shape[-1]

        # The rotations share the token layout, so they are reshaped for global attention the same way
        global_rope_cos, global_rope_sin = rope_cos.view(batch_size, -1, head_dim), rope_sin.view(batch_size, -1, head_dim)

        features = []

        for i, layer_module in enumerate(self.layer):
            # If windowed attention is enabled and this layer is designated for global attention,
            # restore the original batch size by bringing the windows out of the batch dimension
            reshape_for_global_attention = self.num_windows > 1 and i not in self.window_layer_indices

            if reshape_for_global_attention:
                embeddings = embeddings.view(batch_size, -1, embed_dim)
                embeddings = layer_module(embeddings, global_rope_cos, global_rope_sin)
                embeddings = embeddings.view(windowed_batch_size, -1, embed_dim)
            else:
                embeddings = layer_module(embeddings, rope_cos, rope_sin)

            if i in self.out_feature_indices:
                features.append(embeddings)

        return features


class DINOv3ViTModel(nn.Module):
    """
    DINOv3 backbone model with optional windowed attention.

    Args:
        name: Name of the model on HuggingFace.
        patch_size: Size of the patches.
        out_feature_indices: Layer indices to output features from, optional.
        window_layer_indices: Layer indices to apply windowed attention to (if num_windows > 1), optional.
        num_windows: Number of windows to use for windowed attention, optional.
        pos_embed_rescale: Maximum factor to randomly rescale the rotary coordinates by during training, optional.
        pretrained: Whether to load pretrained weights, optional.
    """

    def __init__(
        self,
        name: str,
        patch_size: int,
        out_feature_indices: Optional[List[int]] = None,
        window_layer_indices: Optional[List[int]] = None,
        num_windows: int = 1,
        pos_embed_rescale: Optional[float] = None,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        config = DINOv3ViTConfig.from_pretrained(name)
        config.patch_size = patch_size
        config.pos_embed_rescale = pos_embed_rescale

        self.config = config
        self.out_feature_indices = out_feature_indices if out_feature_indices is not None else [config.num_hidden_layers - 1]
        self.window_layer_indices = window_layer_indices if window_layer_indices is not None else []
        self.num_windows = num_windows

        self.embeddings = DINOv3ViTEmbeddings(config)
        self.rope_embeddings = DINOv3ViTRopePositionEmbedding(config)
        self.model = DINOv3ViTEncoder(config, self.out_feature_indices, self.window_layer_indices, self.num_windows)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self._initialize_weights(name, pretrained=pretrained)

    def forward(self, images: Tensor) -> List[Tensor]:
        # Get batch information
        batch_size, _, height, width = images.shape
        num_patches_height = height // self.config.patch_size
        num_patches_width = width // self.config.patch_size
        num_special_tokens = 1 + self.config.num_register_tokens

        # Create initial embeddings
        embeddings = self.embeddings(images)

        # Create rotations for every token, using the identity rotation for the special tokens
        rope_cos, rope_sin = self.rope_embeddings(num_patches_height, num_patches_width)
        rope_cos = F.pad(rope_cos, (0, 0, num_special_tokens, 0), value=1.0)
        rope_sin = F.pad(rope_sin, (0, 0, num_special_tokens, 0), value=0.0)
        rope_cos = rope_cos.to(embeddings.dtype).expand(batch_size, -1, -1)
        rope_sin = rope_sin.to(embeddings.dtype).expand(batch_size, -1, -1)

        # Optionally, window the embeddings and rotations
        embeddings = self._window_embeddings(embeddings, num_patches_height, num_patches_width)
        rope_cos = self._window_embeddings(rope_cos, num_patches_height, num_patches_width)
        rope_sin = self._window_embeddings(rope_sin, num_patches_height, num_patches_width)

        # Encoder forward pass
        encoder_features = self.model(embeddings, rope_cos, rope_sin)

        # Apply layer norm and restore spatial dimensions, optionally unwindowing the features
        output_features = []
        for encoder_feature in encoder_features:
            # Remove CLS and register tokens
            patch_tokens = encoder_feature[:, num_special_tokens:]

            # Normalize the features
            patch_tokens = self.norm(patch_tokens)

            # Unwindow the features if they were windowed
            patch_tokens = self._unwindow_patch_tokens(patch_tokens, num_patches_height, num_patches_width)

            # Restore the spatial dimensions and move channels to the correct position
            patch_tokens = patch_tokens.view(batch_size, num_patches_height, num_patches_width, self.config.hidden_size)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2)

            output_features.append(patch_tokens)

        return output_features

    def _window_embeddings(self, embeddings: Tensor, num_patches_height: int, num_patches_width: int) -> Tensor:
        """Apply windowing to the embeddings for efficient attention."""

        # No-op if windowing is disabled
        if self.num_windows == 1:
            return embeddings

        # Get batch information
        batch_size, _, embed_dim = embeddings.shape
        num_special_tokens = 1 + self.config.num_register_tokens
        num_windows_height = num_patches_height // self.num_windows
        num_windows_width = num_patches_width // self.num_windows

        # Separate special and patch tokens
        special_tokens, patch_tokens = embeddings[:, :num_special_tokens], embeddings[:, num_special_tokens:]

        # Restore patch dimensions
        patch_tokens = patch_tokens.view(batch_size, num_patches_height, num_patches_width, embed_dim)

        # Separate windows
        patch_tokens = patch_tokens.view(batch_size, self.num_windows, num_windows_height, self.num_windows, num_windows_width, embed_dim)

        # Move windows into the batch dimension and flatten spatial dimensions
        patch_tokens = patch_tokens.permute(0, 1, 3, 2, 4, 5).contiguous()
        patch_tokens = patch_tokens.view(batch_size * self.num_windows**2, num_windows_height * num_windows_width, embed_dim)

        # Repeat special tokens for each window
        special_tokens = special_tokens.unsqueeze(1).expand(batch_size, self.num_windows**2, num_special_tokens, embed_dim)
        special_tokens = special_tokens.reshape(batch_size * self.num_windows**2, num_special_tokens, embed_dim)

        embeddings = torch.cat([special_tokens, patch_tokens], dim=1)

        return embeddings

    def _unwindow_patch_tokens(self, patch_tokens: Tensor, num_patches_height: int, num_patches_width: int) -> Tensor:
        """Restore the original batch size and spatial dimensions for the patch tokens after windowed attention."""

        # No-op if windowing is disabled
        if self.num_windows == 1:
            return patch_tokens

        # Get batch information
        windowed_batch_size, _, embed_dim = patch_tokens.shape
        batch_size = windowed_batch_size // (self.num_windows**2)
        num_windows_height = num_patches_height // self.num_windows
        num_windows_width = num_patches_width // self.num_windows

        # Bring the windows out of the batch dimension
        patch_tokens = patch_tokens.view(batch_size, self.num_windows, self.num_windows, num_windows_height, num_windows_width, embed_dim)
        patch_tokens = patch_tokens.permute(0, 1, 3, 2, 4, 5).contiguous()

        # Restore original spatial dimensions
        patch_tokens = patch_tokens.view(batch_size, -1, embed_dim)

        return patch_tokens

    @torch.no_grad()
    def _initialize_weights(self, name: str, *, pretrained: bool = True) -> None:
        """Initialize model weights, optionally loading from a pretrained huggingface checkpoint."""

        if not pretrained:
            return

        logging.info(f"Loading backbone pretrained weights from '{name}'.")

        # Load the state dict from the pretrained HuggingFace model
        state_dict: Dict[str, Tensor] = HFDINOv3ViTModel.from_pretrained(name).state_dict()

        # We don't mask tokens
        del state_dict["embeddings.mask_token"]

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logging.warning(f"Missing keys when loading backbone pretrained weights: {incompatible.missing_keys}")

        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys when loading backbone pretrained weights: {incompatible.unexpected_keys}")

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


__all__ = [
    "DINOv3ViTModel",
]
