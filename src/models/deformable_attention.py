# from typing import Protocol

# import torch
# import torch.nn.functional as F
# from torch import Tensor, nn


# class MultiHeadDeformableAttention(nn.Module):
#     def __init__(self, embed_dim: int, num_heads: int, num_levels: int, num_points: int):
#         super().__init__()

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.num_levels = num_levels
#         self.num_points = num_points

#         self.sampling_offsets: ProjectionType = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
#         self.attention_weights: ProjectionType = nn.Linear(embed_dim, num_heads * num_levels * num_points)
#         self.value_proj: ProjectionType = nn.Linear(embed_dim, embed_dim)
#         self.output_proj: ProjectionType = nn.Linear(embed_dim, embed_dim)

#     def forward(self, query: Tensor, query_reference: Tensor, features: Tensor, spatial_shapes: Tensor) -> Tensor:
#         """
#         Args:
#             query: Query features with shape (batch_size, num_queries, embed_dim).
#             query_reference: Reference points/boxes for the queries with shape (batch_size, num_queries, 2 | 4).
#             features: Multi-level features with shape (batch_size, num_features, embed_dim).
#             spatial_shapes: Height and width of each feature level with shape (num_levels, 2).
#         """
#         batch_size, num_queries, _ = query.shape

#         # Compute sampling locations: (x + (Δx * w), y + (Δy * h))
#         query_reference = query_reference.view(batch_size, num_queries, 1, 1, -1)
#         offsets = self.sampling_offsets(query).view(batch_size, num_queries, self.num_heads, self.num_levels, self.num_points, 2)

#         if query_reference.shape[-1] == 2:  # Points -> no scaling
#             points = query_reference[..., :2] + offsets
#         elif query_reference.shape[-1] == 4:  # Boxes -> scale by w/h
#             points = query_reference[..., :2] + offsets * query_reference[..., 2:]
#         else:
#             raise ValueError(f"Expected reference points/boxes, got {query_reference.shape=}")

#         # Normalize attention weights over the sampling points across all levels
#         attention_weights = self.attention_weights(query).view(batch_size, num_queries, self.num_heads, self.num_levels * self.num_points)
#         attention_weights = attention_weights.softmax(dim=-1)

#         # Project features to value space
#         values = self.value_proj(features)
#         values = values.split([h * w for h, w in spatial_shapes], dim=1)

#         # Sample values
#         sampled_values = []
#         points = points.permute(3, 0, 2, 1, 4, 5).flatten(0, 1)  # (num_levels, batch_size * num_heads, num_queries, num_points, 2)
#         for level_values, level_points, (h, w) in zip(values, points, spatial_shapes):
#             level_values = level_values.view(batch_size, h, w, self.num_heads, self.head_dim)
#             level_values = level_values.permute(0, 3, 4, 1, 2).flatten(0, 1)  # (batch_size * num_heads, head_dim, height, width)

#             level_grid = (level_points * 2) - 1  # Convert from [0, 1] -> [-1, 1]

#             sampled_level_values = F.grid_sample(
#                 level_values,
#                 level_grid,
#                 mode="bilinear",
#                 padding_mode="zeros",
#                 align_corners=False,
#             )

#             sampled_values.append(sampled_level_values)

#         sampled_values = torch.cat(sampled_values, dim=-1)  # (batch_size * num_heads, head_dim, num_queries, num_points * num_levels)

#         attention_weights = attention_weights.view(batch_size * num_queries, self.num_heads, self.num_levels, self.num_points)

#         # Project output
#         output = self.output_proj(output)

#         return output


# class ProjectionType(Protocol):
#     def __call__(self, x: Tensor) -> Tensor: ...
