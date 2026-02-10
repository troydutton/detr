from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from utils.misc import take_annotation_from


class MultiHeadDeformableAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_levels = num_levels
        self.num_points = num_points

        self.sampling_offsets: LinearType = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights: LinearType = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj: LinearType = nn.Linear(embed_dim, embed_dim)
        self.output_proj: LinearType = nn.Linear(embed_dim, embed_dim)

        self._initialize_weights()

    def forward(self, queries: Tensor, query_reference: Tensor, features: Tensor, dimensions: Tensor) -> Tensor:
        """
        Args:
            queries: Query features with shape (batch_size, num_queries, embed_dim).
            query_reference: Reference points/boxes for the queries with shape (batch_size, num_queries, 2 or 4).
            features: Multi-level features with shape (batch_size, num_features, embed_dim).
            dimensions: Width and height of each feature level with shape (num_levels, 2).
        """
        batch_size, num_queries, embed_dim = queries.shape

        # Compute sampling locations: (x + (Δx * w), y + (Δy * h))
        query_reference = query_reference.view(batch_size, num_queries, 1, 1, 1, -1)
        offsets = self.sampling_offsets(queries).view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)

        if query_reference.shape[-1] == 2:  # Points -> (x + Δx, y + Δy)
            offsets = offsets / dimensions.view(1, 1, 1, self.num_levels, 1, 2)
            points = query_reference + offsets
        elif query_reference.shape[-1] == 4:  # Boxes -> (x + (Δx * w), y + (Δy * h))
            offsets = offsets / (2 * self.num_points)
            points = query_reference[..., :2] + offsets * query_reference[..., 2:]
        else:
            raise ValueError(f"Expected reference points/boxes, got {query_reference.shape=}")

        # Convert sampling locations from [0, 1] to [-1, 1] for grid sampling
        points = (points * 2) - 1

        # Project features to value space
        values = self.value_proj(features)

        # Split values and points by level
        values = values.split([w * h for w, h in dimensions], dim=1)
        points = points.permute(3, 0, 2, 1, 4, 5).flatten(1, 2)  # (num_levels, batch_size * num_heads, num_queries, num_points, 2)

        # Sample values from each level
        sampled_values = []
        for level_values, level_points, (w, h) in zip(values, points, dimensions):
            level_values = level_values.view(batch_size, h, w, self.num_heads, self.head_dim)
            level_values = level_values.permute(0, 3, 4, 1, 2).flatten(0, 1)  # (batch_size * num_heads, head_dim, height, width)

            sampled_level_values = F.grid_sample(
                level_values,
                level_points,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )

            sampled_values.append(sampled_level_values.view(batch_size, self.num_heads, self.head_dim, num_queries, self.num_points))

        sampled_values = torch.cat(sampled_values, dim=-1)

        # Calculate attention weights (normalized over the points across all
        attention_weights = self.attention_weights(queries).view(batch_size, num_queries, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = attention_weights.transpose(1, 2).unsqueeze(2)

        # Perform attention
        output = (sampled_values * attention_weights).sum(dim=-1)

        # Restore original shape
        output = output.view(batch_size, embed_dim, num_queries).transpose(1, 2)

        # Output projection
        output = self.output_proj(output)

        return output

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        """
        Initialize the weights of the deformable attention module.
        """

        # Initialize sampling offsets to form a circular grid around the reference points
        nn.init.zeros_(self.sampling_offsets.weight)

        # Assign a unique angle to each head and project onto the unit square
        angles = torch.arange(self.num_heads) * (2 * torch.pi / self.num_heads)
        points = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        points = points / points.abs().max(dim=-1, keepdim=True)[0]

        # Repeat across levels and scale each point by its position in the sequence
        points = points.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            points[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.copy_(points.flatten())

        # Initialize attention weights for uniform attention
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

        # Initialize projections to maintain variance
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class LinearType(Protocol):
    weight: Tensor
    bias: Tensor

    def __call__(self, x: Tensor) -> Tensor: ...
