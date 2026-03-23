from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from utils.misc import take_annotation_from


class Projector(nn.Module):
    """
    Multi-scale feature projector that scales feature maps to unified spatial resolutions,
    concatenates them, and fuses them efficiently using C2f blocks.

    Args:
        embed_dim: The channel dimension for all input and output feature maps.
        in_strides: List of strides map relative to original image size.
        out_strides: List of target spatial strides defining final feature sizes.
        num_blocks: Number of bottleneck blocks per target resolution fusion block.
        activation: Activation function module type.
    """

    def __init__(
        self,
        embed_dim: int,
        in_strides: List[int],
        out_strides: List[int],
        *,
        num_blocks: int = 3,
        activation: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        self.in_strides = in_strides
        self.out_strides = out_strides
        self.num_blocks = num_blocks
        self.activation = activation
        self.num_output_levels = len(out_strides)

        # Create the spatial resampling stages
        self.resampling_stages = nn.ModuleList()
        for target_stride in out_strides:
            resampling_layers = nn.ModuleList()
            for in_stride in in_strides:
                resampling_layers.append(self._build_resampling_layer(embed_dim, in_stride, target_stride))
            self.resampling_stages.append(resampling_layers)

        # Create the multi-scale feature fusion stages
        self.fusion_stages = nn.ModuleList()
        for _ in out_strides:
            self.fusion_stages.append(C2f(embed_dim * len(in_strides), embed_dim, embed_dim, num_blocks, activation=activation))

    def forward(self, backbone_features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: List of backbone features with shape (batch_size, channels, height, width).

        Returns:
            projected_features: List of projected features with shape (batch_size, channels, height, width).
        """

        projected_features = []

        for resampling_stage, fusion_stage in zip(self.resampling_stages, self.fusion_stages):
            # Resample all backbone features to the target spatial resolution
            features = [resampler(feature) for resampler, feature in zip(resampling_stage, backbone_features)]

            # Concatenate along the channel dimension
            features = torch.cat(features, dim=1)

            # Fuse the concatenated features with a C2f block
            features = fusion_stage(features)

            projected_features.append(features)

        return projected_features

    def _build_resampling_layer(self, embed_dim: int, in_stride: int, out_stride: int) -> nn.Module:
        """
        Build a resampling layer to scale a feature map from `in_stride` to `out_stride`.

        Args:
            embed_dim: The dimension of the input and output feature maps.
            in_stride: The spatial stride of the input feature map.
            out_stride: The desired spatial stride of the output feature map.

        Returns:
            resampling_layer: A module that performs the desired resampling.
        """

        ratio = in_stride / out_stride

        if ratio == 4.0:  # Upsample 4x
            return nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                LayerNorm2d(embed_dim),
                self.activation(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                LayerNorm2d(embed_dim),
            )
        elif ratio == 2.0:  # Upsample 2x
            return nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                LayerNorm2d(embed_dim),
            )
        elif ratio == 1.0:  # Do nothing
            return nn.Identity()
        elif ratio == 0.5:  # Downsample 2x
            return nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                LayerNorm2d(embed_dim),
            )
        elif ratio == 0.25:  # Downsample 4x
            return nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                LayerNorm2d(embed_dim),
                self.activation(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                LayerNorm2d(embed_dim),
            )
        else:
            raise NotImplementedError(f"Unsupported stride ratio: {ratio} ({in_stride=}, {out_stride=})")

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class C2f(nn.Module):
    """
    Lightweight implementation of a multi-scale fusion block, inspired by the C2f block in YOLOv8.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        out_channels: Number of output channels.
        num_blocks: Number of bottleneck blocks.
        activation: Activation function, optional.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        *,
        activation: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        # Doubles the embedding dimension to allow for splitting into base and branch
        self.conv1 = nn.Conv2d(in_channels, 2 * hidden_channels, kernel_size=1, bias=False)
        self.norm1 = LayerNorm2d(2 * hidden_channels)
        self.act1 = activation()

        # Bottleneck convolutional blocks
        self.bottlenecks = nn.ModuleList()
        for _ in range(num_blocks):
            self.bottlenecks.append(Bottleneck(hidden_channels, hidden_channels, activation=activation))

        # Projects from the intermediate outputs to the desired output channels
        self.conv2 = nn.Conv2d((2 + num_blocks) * hidden_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = LayerNorm2d(out_channels)
        self.act2 = activation()

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: Features with shape (batch_size, in_channels, height, width).

        Returns:
            features: Features with shape (batch_size, out_channels, height, width).
        """

        # Double the channels and split into base and branch
        features = self.conv1(features)
        features = self.norm1(features)
        features = self.act1(features)

        base, branch = features.split(self.hidden_channels, dim=1)

        # Pass through bottlenecks, appending intermediates
        intermediates = [base, branch]

        for bottleneck in self.bottlenecks:
            branch = bottleneck(branch)

            intermediates.append(branch)

        # Concatenate along the channel dimension
        features = torch.cat(intermediates, dim=1)

        # Project to the output channels
        features = self.conv2(features)
        features = self.norm2(features)
        features = self.act2(features)

        return features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class Bottleneck(nn.Module):
    """
    Bottleneck convolutional block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        expansion: Expansion factor for the hidden channels, optional.
        activation: Activation function, optional.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        expansion: float = 1.0,
        activation: Type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = int(in_channels * expansion)
        self.out_channels = out_channels
        self.shortcut = in_channels == out_channels

        # First convolution projects from in_channels to hidden_channels
        self.conv1 = nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = LayerNorm2d(self.hidden_channels)
        self.act1 = activation()

        # Second convolution projects from hidden_channels to out_channels
        self.conv2 = nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = LayerNorm2d(self.out_channels)
        self.act2 = activation()

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            features: Transformed tensor with shape (batch_size, out_channels, height, width).
        """

        residual = features

        features = self.conv1(features)
        features = self.norm1(features)
        features = self.act1(features)

        features = self.conv2(features)
        features = self.norm2(features)
        features = self.act2(features)

        return features + residual if self.shortcut else features

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class LayerNorm2d(nn.Module):
    """
    2D LayerNorm that performs point-wise mean and variance normalization over the channel dimension.

    Args:
        in_channels: The number of channels to normalize over.
        eps: A value added to the denominator for numerical stability, optional.
    """

    def __init__(self, in_channels: int, eps: float = 1e-4) -> None:
        super().__init__()

        self.in_channels = (in_channels,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, channels, height, width).

        Returns:
            x: Normalized tensor with shape (batch_size, channels, height, width).
        """

        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.in_channels, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)

        return x

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
