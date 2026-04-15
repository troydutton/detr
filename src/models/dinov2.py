import logging
from collections.abc import Iterable
from math import sqrt
from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import Dinov2WithRegistersConfig
from transformers import Dinov2WithRegistersModel as HFDinov2WithRegistersModel

# TODO: Clean up DINOv2 w/ windowed attention, this is heavily slopped :)


class Dinov2WithRegistersPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: Dinov2WithRegistersConfig):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class Dinov2WithRegistersEmbeddings(nn.Module):
    """
    Construct the CLS token, register tokens, position and patch embeddings.
    """

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        self.config = config
        self.patch_size = config.patch_size

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.register_tokens = nn.Parameter(torch.zeros(1, config.num_register_tokens, config.hidden_size))
        self.patch_embeddings = Dinov2WithRegistersPatchEmbeddings(config)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config.hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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


class Dinov2WithRegistersSelfAttention(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention " f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def forward(self, hidden_states: Tensor, **kwargs) -> Tensor:
        batch_size = hidden_states.shape[0]
        new_shape = batch_size, -1, self.num_attention_heads, self.attention_head_size

        key_layer = self.key(hidden_states).view(*new_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(*new_shape).transpose(1, 2)
        query_layer = self.query(hidden_states).view(*new_shape).transpose(1, 2)

        context_layer = nn.functional.scaled_dot_product_attention(
            query=query_layer,
            key=key_layer,
            value=value_layer,
            dropout_p=self.dropout_prob if self.training else 0.0,
        )
        context_layer = context_layer.transpose(1, 2).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        return context_layer


class Dinov2WithRegistersSelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov2WithRegistersLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Dinov2WithRegistersConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class Dinov2WithRegistersAttention(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig):
        super().__init__()
        self.attention = Dinov2WithRegistersSelfAttention(config)
        self.output = Dinov2WithRegistersSelfOutput(config)

    def forward(self, hidden_states: Tensor, **kwargs) -> Tensor:
        self_attn_output = self.attention(hidden_states, **kwargs)
        output = self.output(self_attn_output, hidden_states)
        return output


class Dinov2WithRegistersLayerScale(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    def forward(self, hidden_state: Tensor) -> Tensor:
        return hidden_state * self.lambda1


def drop_path(input: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class Dinov2WithRegistersDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float | None = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: Tensor) -> Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


class Dinov2WithRegistersMLP(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        if isinstance(config.hidden_act, str):
            self.activation = nn.GELU()
        else:
            self.activation = config.hidden_act
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2WithRegistersSwiGLUFFN(nn.Module):
    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: Tensor) -> Tensor:
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


class Dinov2WithRegistersLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config: Dinov2WithRegistersConfig) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Dinov2WithRegistersAttention(config)
        self.layer_scale1 = Dinov2WithRegistersLayerScale(config)
        self.drop_path = Dinov2WithRegistersDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.use_swiglu_ffn:
            self.mlp = Dinov2WithRegistersSwiGLUFFN(config)
        else:
            self.mlp = Dinov2WithRegistersMLP(config)
        self.layer_scale2 = Dinov2WithRegistersLayerScale(config)

    def forward(
        self,
        hidden_states: Tensor,
    ) -> Tensor:
        hidden_states_norm = self.norm1(hidden_states)
        self_attention_output = self.attention(hidden_states_norm)
        self_attention_output = self.layer_scale1(self_attention_output)

        # first residual connection
        hidden_states = self.drop_path(self_attention_output) + hidden_states

        # in Dinov2WithRegisters, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        return layer_output


class Dinov2WithRegistersPreTrainedModel(nn.Module):
    config: Dinov2WithRegistersConfig
    base_model_prefix = "dinov2_with_registers"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["Dinov2WithRegistersLayer"]
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Dinov2WithRegistersLayer,
    }

    @torch.no_grad()
    def _init_weights(self, module: nn.Linear | nn.Conv2d | nn.LayerNorm) -> None:
        """Initialize the weights"""
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
    def __init__(
        self,
        config: Dinov2WithRegistersConfig,
        out_feature_indices: List[int],
        window_layer_indices: List[int],
        num_windows: int = 1,
    ):
        super().__init__()
        self.layer = nn.ModuleList([Dinov2WithRegistersLayer(config) for _ in range(config.num_hidden_layers)])
        self.out_feature_indices = out_feature_indices
        self.window_layer_indices = window_layer_indices
        self.num_windows = num_windows

    def forward(self, hidden_states: Tensor, **kwargs) -> List[Tensor]:
        features = []
        for i, layer_module in enumerate(self.layer):
            if self.num_windows > 1 and i not in self.window_layer_indices:
                B_win, L_win, D = hidden_states.shape
                B = B_win // (self.num_windows**2)
                hidden_states = hidden_states.view(B, -1, D)

            hidden_states = layer_module(hidden_states)

            if self.num_windows > 1 and i not in self.window_layer_indices:
                hidden_states = hidden_states.view(B_win, L_win, D)

            if i in self.out_feature_indices:
                features.append(hidden_states)

        return features


class Dinov2WithRegistersModel(Dinov2WithRegistersPreTrainedModel):
    def __init__(
        self,
        name: str,
        out_feature_indices: List[int] | None = None,
        window_layer_indices: List[int] | None = None,
        num_windows: int = 1,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        config = Dinov2WithRegistersConfig.from_pretrained(name)
        config.image_size = 512
        config.patch_size = 16

        self.config = config
        self.out_feature_indices = out_feature_indices if out_feature_indices is not None else [config.num_hidden_layers - 1]
        self.window_layer_indices = window_layer_indices if window_layer_indices is not None else []
        self.num_windows = num_windows

        self.embeddings = Dinov2WithRegistersEmbeddings(config)
        self.encoder = Dinov2WithRegistersEncoder(
            config,
            out_feature_indices=self.out_feature_indices,
            window_layer_indices=self.window_layer_indices,
            num_windows=self.num_windows,
        )

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self._initialize_weights(name, pretrained=pretrained)

    def forward(self, images: Tensor, **kwargs) -> List[Tensor]:

        embedding_output = self.embeddings(images)

        batch_size, _, height, width = images.shape
        num_patches_height = height // self.config.patch_size
        num_patches_width = width // self.config.patch_size

        if self.num_windows > 1:
            num_special_tokens = 1 + self.config.num_register_tokens
            special_tokens = embedding_output[:, :num_special_tokens, :]
            patch_tokens = embedding_output[:, num_special_tokens:, :]

            patch_tokens = patch_tokens.view(batch_size, num_patches_height, num_patches_width, self.config.hidden_size)
            wh, ww = num_patches_height // self.num_windows, num_patches_width // self.num_windows

            patch_tokens = patch_tokens.view(batch_size, self.num_windows, wh, self.num_windows, ww, self.config.hidden_size)
            patch_tokens = patch_tokens.permute(0, 1, 3, 2, 4, 5).contiguous()
            patch_tokens = patch_tokens.view(batch_size * self.num_windows**2, wh * ww, self.config.hidden_size)

            special_tokens = special_tokens.unsqueeze(1).expand(
                batch_size, self.num_windows**2, num_special_tokens, self.config.hidden_size
            )
            special_tokens = special_tokens.reshape(batch_size * self.num_windows**2, num_special_tokens, self.config.hidden_size)

            embedding_output = torch.cat([special_tokens, patch_tokens], dim=1)

        encoder_outputs = self.encoder(embedding_output, **kwargs)

        output_features = []

        for sequence_output in encoder_outputs:
            sequence_output = self.layernorm(sequence_output)

            if self.num_windows > 1:
                num_special_tokens = 1 + self.config.num_register_tokens
                patch_tokens = sequence_output[:, num_special_tokens:, :]
                wh, ww = num_patches_height // self.num_windows, num_patches_width // self.num_windows

                patch_tokens = patch_tokens.view(batch_size, self.num_windows, self.num_windows, wh, ww, self.config.hidden_size)
                patch_tokens = patch_tokens.permute(0, 1, 3, 2, 4, 5).contiguous()
                feature = patch_tokens.view(batch_size, num_patches_height, num_patches_width, self.config.hidden_size)
                feature = feature.permute(0, 3, 1, 2)
                output_features.append(feature)
            else:
                # Remove CLS (0) and registers (1:1+num_register_tokens)
                start_idx = 1 + self.config.num_register_tokens
                patch_tokens = sequence_output[:, start_idx:, :]
                # Reshape to (batch_size, channels, height, width)
                feature = patch_tokens.transpose(1, 2).reshape(batch_size, self.config.hidden_size, num_patches_height, num_patches_width)
                output_features.append(feature)

        return output_features

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


__all__ = [
    "Dinov2WithRegistersPreTrainedModel",
    "Dinov2WithRegistersModel",
]
