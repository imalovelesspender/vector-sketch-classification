"""Collate functions for the two input representations.

    point_based   -> the original ``collate_sketches`` (strokes, lengths, labels)
    stroke_based  -> ``collate_stroke_based``: the same three keys plus the stroke
                     tensors from :func:`src.data.stroke_split.group_strokes`
                     (stroke_pts, stroke_lens, stroke_xy, sketch_ids, orders, n_strokes)

Select with ``data.representation`` in the config.  ``coordinate_scale`` must be the
manifest's ``normalization.scale`` (sigma) so centred coordinates are in bbox units,
as in Sihui's stroke-level notebook.
"""

from __future__ import annotations

from functools import partial
from typing import Dict

import torch
from torch.nn.utils.rnn import pad_sequence

from .stroke_split import REPRESENTATIONS, group_strokes


def collate_stroke_based(batch, coordinate_scale: float) -> Dict[str, torch.Tensor]:
    seqs, labels = zip(*batch)
    seqs = [torch.as_tensor(s, dtype=torch.float32) for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    strokes = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    stroke_pts, stroke_lens, stroke_xy, sketch_ids, orders, n_strokes = group_strokes(strokes, lengths, coordinate_scale)
    return {
        "strokes": strokes,
        "lengths": lengths,
        "labels": torch.tensor(labels, dtype=torch.long),
        "stroke_pts": stroke_pts,
        "stroke_lens": stroke_lens,
        "stroke_xy": stroke_xy,
        "sketch_ids": sketch_ids,
        "orders": orders,
        "n_strokes": n_strokes,
    }


def get_collate(representation: str = "point_based", coordinate_scale: float = None):
    if representation == "point_based":
        from .quickdraw import collate_sketches

        return collate_sketches
    if representation == "stroke_based":
        if coordinate_scale is None:
            raise ValueError("stroke_based needs coordinate_scale (manifest normalization.scale)")
        return partial(collate_stroke_based, coordinate_scale=float(coordinate_scale))
    raise ValueError(f"unknown representation {representation!r}; choose from {REPRESENTATIONS}")
