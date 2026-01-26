from torch import Tensor, nn

from models.layers import MultiLayerPerceptron
from utils.misc import take_annotation_from


class TransformerDecoder(nn.Module):
    """
    Transformer decoder composed of a stack of N decoder layers.

    Args:
        num_layers: Number of decoder layers.
        embed_dim: Embedding dimension.
        ffn_dim: Feedforward network dimension.
        num_heads: Number of attention heads.
        num_queries: Number of object queries.
        dropout: Dropout rate, optional.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.queries = nn.Embedding(num_queries, embed_dim)  # Learnable content for initializing object queries
        self.query_pos = nn.Embedding(num_queries, embed_dim)  # Learnable positional embeddings for object queries

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    embed_dim=embed_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: Tensor, feature_pos: Tensor = None) -> Tensor:
        """
        Forward pass for the transformer decoder.

        Args:
            features: Features with shape (batch_size, num_features, embed_dim).
            feature_pos: Feature positional embeddings with shape (batch_size, num_features, embed_dim).

        Returns:
            queries: Object queries with shape (batch_size, num_queries, embed_dim).
        """

        # Learnable content and positional embeddings for object queries
        queries = self.queries.weight.unsqueeze(0)
        query_pos = self.query_pos.weight.unsqueeze(0)

        # Expand the queries across the batch size
        queries = queries.expand(features.shape[0], -1, -1)
        query_pos = query_pos.expand(features.shape[0], -1, -1)

        for layer in self.layers:
            queries = layer(
                queries,
                features,
                query_pos=query_pos,
                feature_pos=feature_pos,
            )

        queries = self.norm(queries)

        return queries

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


class TransformerDecoderLayer(nn.Module):
    """
    Single layer of the transformer decoder.

    Args:
        embed_dim: Embedding dimension.
        ffn_dim: Feedforward network dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate, optional.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        # Feedforward Network
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = MultiLayerPerceptron(
            input_dim=embed_dim,
            hidden_dim=ffn_dim,
            output_dim=embed_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        features: Tensor,
        query_pos: Tensor = None,
        feature_pos: Tensor = None,
    ) -> Tensor:
        """
        Forward pass for a single transformer decoder layer.

        Args:
            queries: Object queries with shape (batch_size, num_queries, embed_dim).
            features: Features with shape (batch_size, num_features, embed_dim).
            feature_pos: Feature positional embeddings with shape (batch_size, num_features, embed_dim).
            query_pos: Query positional embeddings with shape (batch_size, num_queries, embed_dim).

        Returns:
            queries: Object queries with the same shape as the input.
        """

        assert features.ndim == 3, f"Expected features of shape (batch_size, num_features, embed_dim), got {features.shape = }"
        assert queries.ndim == 3, f"Expected queries of shape (batch_size, num_queries, embed_dim), got {queries.shape = }"

        # Self-attention
        v = self.norm1(queries)
        q = k = v if query_pos is None else v + query_pos
        queries = queries + self.dropout1(self.self_attention(q, k, v, need_weights=False)[0])

        # Cross-attention
        q = self.norm2(queries)
        q = q if query_pos is None else q + query_pos
        k = features if feature_pos is None else features + feature_pos
        v = features
        queries = queries + self.dropout2(self.cross_attention(q, k, v, need_weights=False)[0])

        # Feedforward Network
        queries = queries + self.dropout3(self.ffn(self.norm3(queries)))

        return queries

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
