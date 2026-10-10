"""RNN-based sequence classifier (baseline).

Guided by the TensorFlow QuickDraw recurrent classification tutorial.

Adapts the Sketch-RNN idea of encoding a stroke sequence with a (bidirectional)
LSTM into a fixed-size vector, but uses that vector as a sketch embedding ``z``
for classification instead of feeding a VAE / generative decoder.

Any encoder used with :class:`src.models.classifier.SketchClassifier` must
follow the same interface so it can be swapped (e.g. for a Transformer):

* ``forward(strokes, lengths) -> z`` with ``strokes`` (B, T, F) zero-padded,
  ``lengths`` (B,) and ``z`` (B, embedding_dim)
* attribute ``embedding_dim``

Framework: PyTorch.
TODO: Decide final architecture and hyperparameters (current values are untuned defaults).
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """LSTM encoder mapping a padded stroke sequence to a sketch embedding ``z``.

    Padding is handled with ``pack_padded_sequence``, so padded time steps never
    influence the LSTM state or the pooled embedding.

    Args:
        input_dim: features per point (3 for stroke-3, 5 for stroke-5).
        hidden_dim: LSTM hidden size per direction.
        num_layers: stacked LSTM layers.
        bidirectional: use a bidirectional LSTM.
        dropout: dropout between LSTM layers (only if ``num_layers > 1``) and
            on the pooled output.
        pooling: ``"last"`` concatenates the final hidden state of each direction
            of the top layer; ``"mean"`` averages LSTM outputs over valid steps.
        embedding_dim: if set, a linear projection maps the pooled vector to this
            size; otherwise ``z`` is the pooled vector itself.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
        pooling: str = "last",
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.pooling = pooling
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        pooled_dim = hidden_dim * self.num_directions
        self.proj = nn.Linear(pooled_dim, embedding_dim) if embedding_dim else None
        self.embedding_dim = embedding_dim or pooled_dim

    def forward(self, strokes: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            strokes, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (h_n, _) = self.lstm(packed)
        if self.pooling == "last":
            # h_n: (num_layers * num_directions, B, H); take the top layer.
            top = h_n[-self.num_directions:]
            pooled = top.transpose(0, 1).reshape(strokes.size(0), -1)
        else:
            padded, _ = pad_packed_sequence(
                outputs, batch_first=True, total_length=strokes.size(1)
            )
            mask = (
                torch.arange(strokes.size(1), device=strokes.device)[None, :]
                < lengths.to(strokes.device)[:, None]
            ).unsqueeze(-1)
            pooled = (padded * mask).sum(1) / lengths.to(strokes.device)[:, None]
        z = self.dropout(pooled)
        if self.proj is not None:
            z = self.proj(z)
        return z
