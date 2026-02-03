from torch import Tensor, nn

from models.decoder import TransformerDecoder
from models.encoder import TransformerEncoder
from utils.misc import take_annotation_from


class Transformer(nn.Module):
    """
    Transformer consisting of an encoder and decoder.

    Args:
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        embed_dim: Embedding dimension.
        ffn_dim: Feedforward network dimension.
        num_heads: Number of attention heads.
        num_queries: Number of object queries.
        dropout: Dropout rate, optional.
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_queries=num_queries,
            dropout=dropout,
        )

    def forward(self, features: Tensor, feature_pos: Tensor = None) -> Tensor:
        """
        Args:
            features: Features with shape (batch_size, height, width, embed_dim).
            feature_pos: Feature positional embeddings with shape (batch_size, height, width, embed_dim).

        Returns:
            queries: Object queries with shape (batch_size, num_queries, embed_dim).
        """

        assert features.ndim == 4, f"Expected features of shape (batch_size, height, width, embed_dim), got {features.shape=}"

        # Collapse the spatial dimensions
        batch_size, height, width, embed_dim = features.shape
        features = features.view(batch_size, height * width, embed_dim)
        feature_pos = None if feature_pos is None else feature_pos.view(batch_size, height * width, embed_dim)

        # Encode the features
        features = self.encoder(features, feature_pos)

        # Decode object queries
        queries = self.decoder(features, feature_pos)

        return queries

    @take_annotation_from(forward)
    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)
