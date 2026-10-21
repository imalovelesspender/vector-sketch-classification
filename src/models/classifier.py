"""Sketch classifier: sequence encoder -> embedding z -> linear head."""

import torch
import torch.nn as nn


class SketchClassifier(nn.Module):
    """Encoder-agnostic classifier.

    ``encoder`` must implement ``forward(strokes, lengths) -> z`` and expose
    ``embedding_dim`` (see :mod:`src.models.rnn`). Swapping the LSTM for a
    Transformer encoder only requires passing a different encoder here.
    """

    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, num_classes)

    def embed(self, strokes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Return sketch embeddings ``z`` (B, embedding_dim), e.g. for retrieval."""
        return self.encoder(strokes, lengths)

    def forward(
        self, strokes: torch.Tensor, lengths: torch.Tensor, return_embedding: bool = False
    ):
        z = self.embed(strokes, lengths)
        logits = self.head(z)
        return (logits, z) if return_embedding else logits


def build_model(model_cfg: dict, input_dim: int, num_classes: int) -> SketchClassifier:
    """Build a classifier from a config dict ``{"encoder": "lstm", ...encoder kwargs}``."""
    cfg = dict(model_cfg)
    name = cfg.pop("encoder", "lstm")
    if name == "lstm":
        from .rnn import LSTMEncoder

        encoder = LSTMEncoder(input_dim=input_dim, **cfg)
    else:
        # TODO: register the Transformer encoder here once src/models/transformer.py exists.
        raise ValueError(f"unknown encoder {name!r}")
    return SketchClassifier(encoder, num_classes)
