import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils.misc import take_annotation_from


class MultiheadAttention(nn.Module):
    """
    Implementation of multihead attention originally introduced in
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

    We prefer this implementation over `torch.nn.MultiheadAttention` because it allows
    us to pass a single attention mask that broadcasts over the heads directly. This
    avoids the need to materialize a copy of the mask for every head, saving memory.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate applied to the attention weights, optional.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(f"{embed_dim=} is not a multiple of {num_heads=}.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # Query, key, and value projections are packed into a single parameter
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._initialize_weights()

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, attention_mask: Tensor = None) -> Tensor:
        # Get batch information
        batch_size, num_queries, _ = queries.shape

        # Project the inputs and separate the heads
        query_weight, key_weight, value_weight = self.in_proj_weight.chunk(3, dim=0)
        query_bias, key_bias, value_bias = self.in_proj_bias.chunk(3, dim=0)

        queries = F.linear(queries, query_weight, query_bias).view(batch_size, num_queries, self.num_heads, self.head_dim)
        keys = F.linear(keys, key_weight, key_bias).view(batch_size, -1, self.num_heads, self.head_dim)
        values = F.linear(values, value_weight, value_bias).view(batch_size, -1, self.num_heads, self.head_dim)

        # Perform attention over the head dimension
        attended = F.scaled_dot_product_attention(
            queries.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # Restore the original shape and project the output
        attended = attended.transpose(1, 2).reshape(batch_size, num_queries, self.embed_dim)

        return self.out_proj(attended)

    @torch.no_grad()
    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj.bias)

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
