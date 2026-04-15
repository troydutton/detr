import logging
from collections.abc import Iterable
from math import sqrt
from typing import Dict, List, Tuple

import torch
import transformers
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import Dinov2WithRegistersConfig
from transformers import Dinov2WithRegistersModel as HFDinov2WithRegistersModel

from utils.misc import take_annotation_from

transformers.utils.logging.disable_progress_bar()


class Dinov2WithRegistersPatchEmbeddings(nn.Module):
    """
    Initial convolutional layer responsible for converting the input images into patch embeddings.

    Args:
        image_size: Size of the input images, interpreted as (image_size, image_size) if a single integer is provided.
        patch_size: Size of the patches, interpreted as (patch_size, patch_size) if a single integer is provided.
        num_channels: Number of channels in the input images.
        hidden_size: Embedding dimension of the patch embeddings.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        # Assume square images and patches if only a single integer is provided
        self.image_size = config.image_size if isinstance(config.image_size, Iterable) else (config.image_size, config.image_size)
        self.patch_size = config.patch_size if isinstance(config.patch_size, Iterable) else (config.patch_size, config.patch_size)
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size

        # Calculate the number of patches for the target resolution
        height, width = self.image_size
        patch_height, patch_width = self.patch_size

        self.num_patches = (height // patch_height) * (width // patch_width)

        # Initialize the projection layer
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, images: Tensor) -> Tensor:
        """
        Convert input images to patch embeddings.

        Args:
            images: Images with shape (batch_size, num_channels, height, width).

        Returns:
            embeddings: Patch embeddings with shape (batch_size, num_patches, embed_dim).
        """

        # Patchify the images
        embeddings: Tensor = self.projection(images)

        # Collapse the spatial dimensions and move the embedding dimension to the end
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersEmbeddings(nn.Module):
    """
    Constructs the CLS token, register tokens, patch embeddings, and adds positional embeddings.

    Args:
        hidden_size: Embedding dimension.
        patch_size: Size of the patches, interpreted as (patch_size, patch_size) if a single integer is provided.
        num_register_tokens: Number of register tokens to prepend.
        hidden_dropout_prob: The dropout probability for the embeddings.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.num_register_tokens = config.num_register_tokens
        self.dropout_prob = config.hidden_dropout_prob

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, self.embed_dim))
        self.patch_embeddings = Dinov2WithRegistersPatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, self.embed_dim))

        self.dropout = nn.Dropout(self.dropout_prob)

    def interpolate_pos_encoding(self, height: int, width: int) -> Tensor:
        """
        Interpolate the pre-trained position encodings to match the number of patches for the given image size.

        Args:
            height: Height of the input image.
            width: Width of the input image.

        Returns:
            position_embeddings: Interpolated position embeddings with shape (1, num_patches + 1, embed_dim).
        """

        # Skip interpolation for matching dimensions (unless tracing)
        if not torch.jit.is_tracing() and self.patch_embeddings.image_size == (height, width):
            return self.position_embeddings

        # Separate the class and patch positional embeddings
        class_pos_embed, patch_pos_embed = self.position_embeddings[:, :1], self.position_embeddings[:, 1:]

        # Calculate the current number of patches
        _, current_num_patches, embed_dim = patch_pos_embed.shape
        current_size = int(sqrt(current_num_patches))

        # Calculate desired number of patches
        num_patches_height = int(height // self.patch_size)
        num_patches_width = int(width // self.patch_size)
        new_num_patches = num_patches_height * num_patches_width

        # Restore spatial dimensions, interpolate, and flatten again
        patch_pos_embed = patch_pos_embed.reshape(1, current_size, current_size, embed_dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = F.interpolate(
            patch_pos_embed.float(),
            size=(num_patches_height, num_patches_width),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=patch_pos_embed.dtype)

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        patch_pos_embed = patch_pos_embed.view(1, new_num_patches, embed_dim)

        # Combine class and patch embeddings
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, images: Tensor) -> Tensor:
        """
        Create patch embeddings and add class and register tokens.

        Args:
            images: Images with shape (batch_size, num_channels, height, width).

        Returns:
            embeddings: Patch embeddings with class and register tokens with shape (batch_size, seq_length, embed_dim).
        """

        # Get batch information
        batch_size, _, height, width = images.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype

        # Create patch embeddings
        embeddings = self.patch_embeddings(images.to(dtype=target_dtype))

        # Add the CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # Add positional embeddings
        embeddings = embeddings + self.interpolate_pos_encoding(height, width)

        # Add register tokens
        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat((embeddings[:, :1], register_tokens, embeddings[:, 1:]), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Args:
        hidden_size: Embedding dimension.
        num_attention_heads: Number of attention heads.
        attention_probs_dropout_prob: The dropout probability for the attention probabilities.
        qkv_bias: Whether to include bias terms in the query, key, and value projections.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f"{config.hidden_size=} is not a multiple of {config.num_attention_heads=}.")

        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = int(self.embed_dim / self.num_heads)
        self.dropout_prob = config.attention_probs_dropout_prob

        self.query = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.key = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.value = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Performs multi-head self-attention on the input embeddings.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            attention_output: Attention output with shape (batch_size, seq_length, embed_dim).

        """

        # Get batch information
        batch_size, seq_length, embed_dim = embeddings.shape

        # Project the embeddings and reshape for multi-head attention
        q: Tensor = self.query(embeddings)
        k: Tensor = self.key(embeddings)
        v: Tensor = self.value(embeddings)

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform attention
        attention_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_prob if self.training else 0.0)

        # Restore original shape
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_length, embed_dim)

        return attention_output

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersSelfOutput(nn.Module):
    """
    Output projection for self-attention.

    Args:
        hidden_size: Embedding dimension.
        hidden_dropout_prob: Dropout probability for the output.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.dropout_prob = config.hidden_dropout_prob

        self.dense = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Project the output of self-attention and apply dropout.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            output: Output embeddings with shape (batch_size, seq_length, embed_dim).
        """

        embeddings = self.dropout(self.dense(embeddings))

        return embeddings

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersAttention(nn.Module):
    """
    Wrapper for the attention mechanism.

    This is really dumb but it's how huggingface implemented it ¯\(ツ)/¯

    Args:
        config: Configuration object containing model hyperparameters.
    """

    def __init__(self, config: Dinov2WithRegistersConfig):
        super().__init__()

        self.attention = Dinov2WithRegistersSelfAttention(config)
        self.output = Dinov2WithRegistersSelfOutput(config)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Perform attention and project the output.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            output: Output embeddings with shape (batch_size, seq_length, embed_dim).
        """

        output = self.output(self.attention(embeddings))

        return output

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersLayerScale(nn.Module):
    """
    Layer scale module for scaling the output of attention and MLP layers.

    Args:
        hidden_size: Embedding dimension.
        layerscale_value: Initial value for the layer scale parameters.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size
        self.value = config.layerscale_value

        self.lambda1 = nn.Parameter(self.value * torch.ones(self.embed_dim))

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Scale the input embeddings.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            output: Scaled embeddings with shape (batch_size, seq_length, embed_dim).
        """

        return embeddings * self.lambda1

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Args:
        drop_prob: The probability of dropping paths, optional.
    """

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()

        self.drop_prob = drop_prob if drop_prob is not None else 0.0
        self.keep_prob = 1 - self.drop_prob

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Apply drop path to the embeddings.

        Args:
            embeddings: Embeddings with shape (batch_size, ...).

        Returns:
            embeddings: Embeddings after applying drop path, with shape (batch_size, ...).
        """

        if self.drop_prob == 0.0 or not self.training:
            return embeddings

        # Generate a mask for the batch elements (efficient version of torch.bernoulli(self.keep_prob))
        shape = (embeddings.shape[0],) + (1,) * (embeddings.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(shape, dtype=embeddings.dtype, device=embeddings.device)
        random_tensor.floor_()

        # Apply the mask and scale the embeddings to maintain the expected value
        embeddings = embeddings.div(self.keep_prob) * random_tensor

        return embeddings

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersMLP(nn.Module):
    """
    Standard feedforward network with one hidden layer and an activation function.

    Args:
        hidden_size: Embedding dimension.
        mlp_ratio: Ratio of the hidden layer dimension to the embedding dimension.
        hidden_act: Activation function to use, either as a string or a callable.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.activation = nn.GELU() if isinstance(config.hidden_act, str) else config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Apply the feedforward network to the input embeddings.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            embeddings: Output embeddings with shape (batch_size, seq_length, embed_dim).
        """

        embeddings = self.fc2(self.activation(self.fc1(embeddings)))

        return embeddings

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersSwiGLUFFN(nn.Module):
    """
    SwiGLU feedforward network with one hidden layer.

    Args:
        hidden_size: Embedding dimension.
        mlp_ratio: Ratio of the hidden layer dimension to the embedding dimension.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Apply the SwiGLU feedforward network to the input embeddings.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            embeddings: Output embeddings with shape (batch_size, seq_length, embed_dim).
        """
        embeddings = self.weights_in(embeddings)

        gate, signal = embeddings.chunk(2, dim=-1)

        embeddings = F.silu(gate) * signal

        return self.weights_out(embeddings)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Dinov2WithRegistersLayer(nn.Module):
    """
    Single layer of the backbone, consisting of self-attention and a feedforward network.

    Args:
        config: Configuration object containing model hyperparameters.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2WithRegistersAttention(config)
        self.layer_scale1 = Dinov2WithRegistersLayerScale(config)
        self.drop_path = Dinov2WithRegistersDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        # Feedforward network
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Dinov2WithRegistersSwiGLUFFN(config) if config.use_swiglu_ffn else Dinov2WithRegistersMLP(config)
        self.layer_scale2 = Dinov2WithRegistersLayerScale(config)

    def forward(self, embeddings: Tensor) -> Tensor:
        """
        Forward pass for a single layer of the backbone.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            embeddings: Output embeddings with shape (batch_size, seq_length, embed_dim).
        """

        # Self-attention
        embeddings = embeddings + self.drop_path(self.layer_scale1(self.attention(self.norm1(embeddings))))

        # Feedforward network
        embeddings = embeddings + self.drop_path(self.layer_scale2(self.mlp(self.norm2(embeddings))))

        return embeddings


class Dinov2WithRegistersPreTrainedModel(nn.Module):
    config: Dinov2WithRegistersConfig
    base_model_prefix = "dinov2_with_registers"
    main_input_name = "images"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2WithRegistersLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "embeddings": Dinov2WithRegistersLayer,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize the module weights.

        Args:
            module: Module to initialize.
        """

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, Dinov2WithRegistersEmbeddings):
            nn.init.trunc_normal_(module.position_embeddings, mean=0.0, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.cls_token, mean=0.0, std=self.config.initializer_range)
            nn.init.zeros_(module.register_tokens)
        elif isinstance(module, Dinov2WithRegistersLayerScale):  # noqa: F821
            nn.init.constant_(module.lambda1, self.config.layerscale_value)


class Dinov2WithRegistersEncoder(Dinov2WithRegistersPreTrainedModel):
    """
    Sequential encoder made up of multiple layers.

    Args:
        config: Configuration object containing model hyperparameters.
        out_feature_indices: Layer indices to output features from.
        window_layer_indices: Layer indices to apply windowed attention to (if num_windows > 1).
        num_windows: Number of windows to use for windowed attention, optional.
    """

    def __init__(
        self,
        config: Dinov2WithRegistersConfig,
        out_feature_indices: List[int],
        window_layer_indices: List[int],
        num_windows: int = 1,
    ) -> None:
        super().__init__()

        self.out_feature_indices = out_feature_indices
        self.window_layer_indices = window_layer_indices
        self.num_windows = num_windows

        self.layer = nn.ModuleList([Dinov2WithRegistersLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, embeddings: Tensor) -> List[Tensor]:
        """
        Forward pass through the encoder layers.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).

        Returns:
            features: Features from the specified layers, each with shape (batch_size, seq_length, embed_dim).
        """

        # Get batch information
        windowed_batch_size, _, embed_dim = embeddings.shape
        batch_size = windowed_batch_size // (self.num_windows**2)

        features = []

        for i, layer_module in enumerate(self.layer):
            # If windowed attention is enabled and this layer is designated for global attention,
            # restore the original batch size by bringing the windows out of the batch dimension
            reshape_for_global_attention = self.num_windows > 1 and i not in self.window_layer_indices

            if reshape_for_global_attention:
                embeddings = embeddings.view(batch_size, -1, embed_dim)

            embeddings = layer_module(embeddings)

            # Restore windowed batch size if we reshaped for global attention at the start of the layer
            if reshape_for_global_attention:
                embeddings = embeddings.view(windowed_batch_size, -1, embed_dim)

            if i in self.out_feature_indices:
                features.append(embeddings)

        return features


class Dinov2WithRegistersModel(Dinov2WithRegistersPreTrainedModel):
    """
    DINOv2 backbone model with register tokens and optional windowed attention.

    Args:
        name: Name of the model on HuggingFace.
        image_size: Size of the input images, interpreted as (image_size, image_size) if a single integer is provided.
        patch_size: Size of the patches, interpreted as (patch_size, patch_size) if a single integer is provided.
        out_feature_indices: Layer indices to output features from, optional.
        window_layer_indices: Layer indices to apply windowed attention to (if num_windows > 1), optional.
        num_windows: Number of windows to use for windowed attention, optional.
    """

    def __init__(
        self,
        name: str,
        image_size: int | Tuple[int, int],
        patch_size: int | Tuple[int, int],
        out_feature_indices: List[int] | None = None,
        window_layer_indices: List[int] | None = None,
        num_windows: int = 1,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        config = Dinov2WithRegistersConfig.from_pretrained(name)
        config.image_size = image_size
        config.patch_size = patch_size

        self.config = config
        self.out_feature_indices = out_feature_indices if out_feature_indices is not None else [config.num_hidden_layers - 1]
        self.window_layer_indices = window_layer_indices if window_layer_indices is not None else []
        self.num_windows = num_windows

        self.embeddings = Dinov2WithRegistersEmbeddings(config)
        self.encoder = Dinov2WithRegistersEncoder(config, self.out_feature_indices, self.window_layer_indices, self.num_windows)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self._initialize_weights(name, pretrained=pretrained)

    def forward(self, images: Tensor) -> List[Tensor]:
        # Get batch information
        batch_size, _, height, width = images.shape
        num_patches_height = height // self.config.patch_size
        num_patches_width = width // self.config.patch_size

        # Create initial embeddings
        embeddings = self.embeddings(images)

        # Optionally, window the embeddings
        embeddings = self._window_embeddings(embeddings, num_patches_height, num_patches_width)

        # Encoder forward pass
        encoder_features = self.encoder(embeddings)

        # Apply layer norm and restore spatial dimensions, optionally unwindowing the features
        output_features = []
        for encoder_feature in encoder_features:
            # Remove CLS and register tokens
            patch_tokens = encoder_feature[:, 1 + self.config.num_register_tokens :]

            # Normalize the features
            patch_tokens = self.layernorm(patch_tokens)

            # Unwindow the features if they were windowed
            patch_tokens = self._unwindow_patch_tokens(patch_tokens, num_patches_height, num_patches_width)

            # Restore the spatial dimensions and move channels to the correct position
            patch_tokens = patch_tokens.view(batch_size, num_patches_height, num_patches_width, self.config.hidden_size)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2)

            output_features.append(patch_tokens)

        return output_features

    def _window_embeddings(self, embeddings: Tensor, num_patches_height: int, num_patches_width: int) -> Tensor:
        """
        Apply windowing to the embeddings for efficient attention.

        Args:
            embeddings: Input embeddings with shape (batch_size, seq_length, embed_dim).
            num_patches_height: Number of patches along the height dimension.
            num_patches_width: Number of patches along the width dimension.

        Returns:
            embeddings: Windowed embeddings with shape (batch_size * num_windows^2, window_seq_length, embed_dim).
        """

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
        patch_tokens = patch_tokens.view(batch_size * self.num_windows**2, num_windows_height * num_windows_width, self.config.hidden_size)

        # Repeat special tokens for each window
        special_tokens = special_tokens.unsqueeze(1).expand(batch_size, self.num_windows**2, num_special_tokens, self.config.hidden_size)
        special_tokens = special_tokens.reshape(batch_size * self.num_windows**2, num_special_tokens, self.config.hidden_size)

        embeddings = torch.cat([special_tokens, patch_tokens], dim=1)

        return embeddings

    def _unwindow_patch_tokens(self, patch_tokens: Tensor, num_patches_height: int, num_patches_width: int) -> Tensor:
        """
        Restore the original batch size and spatial dimensions for the patch tokens after windowed attention.

        Args:
            patch_tokens: Windowed patch tokens with shape (batch_size * num_windows^2, window_seq_length, embed_dim).
            num_patches_height: Number of patches along the height dimension.
            num_patches_width: Number of patches along the width dimension.

        Returns:
            patch_tokens: Unwindowed patch tokens with shape (batch_size, num_patches, embed_dim).
        """

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
        patch_tokens = patch_tokens.view(batch_size, -1, self.config.hidden_size)

        return patch_tokens

    @torch.no_grad()
    def _initialize_weights(self, name: str, *, pretrained: bool = True) -> None:
        """
        Initialize model weights, optionally loading from a pretrained huggingface checkpoint.

        Args:
            name: Name of the pretrained model to load from HuggingFace.
            pretrained: Whether to load pretrained weights, optional.
        """

        if not pretrained:
            return

        logging.info(f"Loading backbone pretrained weights from '{name}'.")

        # Load the state dict from the pretrained HuggingFace model
        state_dict: Dict[str, Tensor] = HFDinov2WithRegistersModel.from_pretrained(name).state_dict()

        # We don't mask tokens
        del state_dict["embeddings.mask_token"]

        # Interpolate positional embeddings (i.e. if the image or patch size changes)
        old_pos_embed = state_dict["embeddings.position_embeddings"]
        new_pos_embed = self.embeddings.position_embeddings

        if old_pos_embed.shape != new_pos_embed.shape:
            # Separate the class and patch tokens
            old_cls_pos_embed, old_patch_pos_embed = old_pos_embed[:, :1], old_pos_embed[:, 1:]
            new_patch_pos_embed = new_pos_embed[:, 1:]

            # Calculate the side lengths
            _, old_num_patches, embed_dim = old_patch_pos_embed.shape
            _, new_num_patches, _ = new_patch_pos_embed.shape

            old_size = int(sqrt(old_num_patches))
            new_size = int(sqrt(new_num_patches))

            logging.info(f"Interpolating positional embeddings: ({old_size}, {old_size}) -> ({new_size}, {new_size}).")

            # Restore spatial dimensions, interpolate, and flatten again
            old_patch_pos_embed = old_patch_pos_embed.reshape(1, old_size, old_size, embed_dim)
            old_patch_pos_embed = old_patch_pos_embed.permute(0, 3, 1, 2)

            patch_pos_embed = F.interpolate(
                old_patch_pos_embed.float(),
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            ).to(old_patch_pos_embed.dtype)

            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
            patch_pos_embed = patch_pos_embed.reshape(1, new_num_patches, embed_dim)

            pos_embed = torch.cat((old_cls_pos_embed, patch_pos_embed), dim=1)

            state_dict["embeddings.position_embeddings"] = pos_embed

        # Interpolate patch embeddings (i.e. if the patch size changes)
        old_patch_embeddings = state_dict["embeddings.patch_embeddings.projection.weight"]
        new_patch_embeddings = self.embeddings.patch_embeddings.projection.weight

        if old_patch_embeddings.shape != new_patch_embeddings.shape:
            old_patch_size = tuple(old_patch_embeddings.shape[-2:])
            new_patch_size = self.embeddings.patch_embeddings.patch_size

            logging.info(f"Interpolating patch embeddings: {old_patch_size} -> {new_patch_size}.")

            patch_embeddings = F.interpolate(
                old_patch_embeddings.float(),
                size=new_patch_size,
                mode="bicubic",
                align_corners=False,
            ).to(old_patch_embeddings.dtype)

            state_dict["embeddings.patch_embeddings.projection.weight"] = patch_embeddings

        incompatible = self.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logging.warning(f"Missing keys when loading backbone pretrained weights: {incompatible.missing_keys}")

        if incompatible.unexpected_keys:
            logging.warning(f"Unexpected keys when loading backbone pretrained weights: {incompatible.unexpected_keys}")

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


__all__ = [
    "Dinov2WithRegistersPreTrainedModel",
    "Dinov2WithRegistersModel",
]
