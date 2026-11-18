"""``stroke_embed`` encoder: stroke tokens (shape + location + order) -> baseline LSTM.

Token construction follows Sihui's stroke-level Transformer notebook
(``transformer-baseline``, ``StrokeLevelTransformer``), so the only difference
between that model and this one is the encoder that reads the tokens:
Transformer + CLS there, the baseline :class:`LSTMEncoder` here.

    stroke_pts (N, P, 4) --BiLSTM(4 -> 64x2), final states--> Linear(128 -> d)   shape
    stroke_xy  (N, 2)    --Linear(2 -> d)                                        location
    stroke index          --Embedding(max_strokes, d)                            order
    token = shape + location + order, dropout
    (B, S, d) --LSTMEncoder (baseline BiLSTM-256, last state)--> z

Input: the ``stroke_based`` representation (see ``src.data.stroke_split``).  If a
batch arrives as ``point_based`` the strokes are grouped here, so both collates work.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from ..data.stroke_split import group_strokes
from .rnn import LSTMEncoder


class StrokeLSTMEncoder(nn.Module):
    """Register in ``build_model`` as ``"encoder": "stroke_embed"``.

    Top-LSTM kwargs as :class:`LSTMEncoder` (hidden_dim, num_layers, bidirectional,
    dropout, pooling, embedding_dim).  Embedding kwargs (notebook names/defaults):
        dim            token width d (128)
        stroke_hidden  shape BiLSTM units per direction (64)
        max_strokes    order-embedding table (250)
        coordinate_scale  manifest sigma; only needed when a point_based batch
                       must be grouped inside the model
        use_shape / use_location / use_order   ablation switches
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
        dim: int = 128,
        stroke_hidden: int = 64,
        max_strokes: int = 250,
        coordinate_scale: Optional[float] = None,
        use_shape: bool = True,
        use_location: bool = True,
        use_order: bool = True,
    ):
        super().__init__()
        if input_dim != 3:
            raise ValueError("stroke_embed expects stroke-3 input (input_dim=3)")
        self.dim, self.max_strokes, self.coordinate_scale = dim, max_strokes, coordinate_scale
        if use_shape:
            self.shape_lstm = nn.LSTM(4, stroke_hidden, bidirectional=True, batch_first=True)
            self.shape_projection = nn.Linear(2 * stroke_hidden, dim)
        else:
            self.shape_lstm = None
        self.location_embedding = nn.Linear(2, dim) if use_location else None
        self.order_embedding = nn.Embedding(max_strokes, dim) if use_order else None
        self.embedding_dropout = nn.Dropout(dropout)
        self.top = LSTMEncoder(
            input_dim=dim, hidden_dim=hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, dropout=dropout, pooling=pooling,
            embedding_dim=embedding_dim,
        )
        self.embedding_dim = self.top.embedding_dim

    def tokens(
        self,
        strokes: torch.Tensor,
        lengths: torch.Tensor,
        stroke_pts: Optional[torch.Tensor] = None,
        stroke_lens: Optional[torch.Tensor] = None,
        stroke_xy: Optional[torch.Tensor] = None,
        n_strokes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stroke tokens ``(B, S, dim)`` and ``n_strokes (B,)``."""
        if stroke_pts is None:  # point_based batch: group here
            if self.coordinate_scale is None:
                raise ValueError("point_based batch needs model coordinate_scale to group strokes")
            stroke_pts, stroke_lens, stroke_xy, n_strokes = group_strokes(strokes, lengths, self.coordinate_scale)
        dev = strokes.device
        stroke_pts, stroke_xy, n_strokes = stroke_pts.to(dev), stroke_xy.to(dev), n_strokes.to(dev)
        N = stroke_pts.size(0)
        tok = stroke_pts.new_zeros(N, self.dim)
        if self.shape_lstm is not None:
            with torch.autocast(device_type=dev.type, enabled=False):  # packed LSTM stays float32
                packed = pack_padded_sequence(stroke_pts.float(), stroke_lens.cpu(), batch_first=True, enforce_sorted=False)
                _, (h_n, _) = self.shape_lstm(packed)
                shape = torch.cat([h_n[-2], h_n[-1]], dim=1)          # final fwd + bwd states
            tok = tok + self.shape_projection(shape)
        if self.location_embedding is not None:
            tok = tok + self.location_embedding(stroke_xy)
        if self.order_embedding is not None:
            orders = torch.cat([torch.arange(int(n), device=dev) for n in n_strokes]).clamp(max=self.max_strokes - 1)
            tok = tok + self.order_embedding(orders)
        tok = pad_sequence(torch.split(tok, n_strokes.tolist()), batch_first=True)  # (B, S, dim)
        return self.embedding_dropout(tok), n_strokes

    def forward(self, strokes: torch.Tensor, lengths: torch.Tensor, **stroke_batch) -> torch.Tensor:
        tok, n_strokes = self.tokens(strokes, lengths, **stroke_batch)
        return self.top(tok, n_strokes)
