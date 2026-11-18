"""Point -> stroke representation (no learnable parameters).

Follows Sihui's stroke-level Transformer notebook (``transformer-baseline`` branch,
``notebooks/Transformer Stroke Level.ipynb``, cells 8 and 10) so that the LSTM and
Transformer stroke models see identical stroke tokens.

Representations (config ``data.representation``):

    point_based   (B, T, 3) = (dx, dy, pen) as stored by Sihui's preprocessing.
                  Nothing is added; this is the baseline input.

    stroke_based  Points are grouped into strokes.  Absolute coordinates are the
                  cumulative sum of (dx, dy) multiplied by ``coordinate_scale``
                  (the manifest's sigma), which puts them back in bounding-box
                  units (longest side = 1), then centred on the sketch's mean point.
                  For each stroke:
                      stroke_pts  (N, P, 4) = (x - x0, y - y0, draw, end) per point,
                                              local to the stroke's start point
                      stroke_xy   (N, 2)    = centred (x0, y0) of the start point
                      stroke_lens (N,), n_strokes (B,)
                  Pen-up jumps position the next stroke (through stroke_xy) but are
                  excluded from that stroke's local shape, exactly as in the notebook.
"""

from __future__ import annotations

from typing import Tuple

import torch

REPRESENTATIONS = ("point_based", "stroke_based")
STROKE_KEYS = ("stroke_pts", "stroke_lens", "stroke_xy", "sketch_ids", "orders", "n_strokes")


def centred_coordinates(strokes: torch.Tensor, lengths: torch.Tensor, coordinate_scale: float) -> torch.Tensor:
    """(B, T, 3) -> (B, T, 2) absolute xy in bbox units, centred per sketch (notebook cell 8)."""
    B, T, _ = strokes.shape
    lengths = lengths.to(strokes.device)
    valid = torch.arange(T, device=strokes.device)[None, :] < lengths[:, None]
    clean = strokes.masked_fill(~valid[:, :, None], 0)
    xy = clean[:, :, :2].cumsum(1) * coordinate_scale
    center = (xy * valid[:, :, None]).sum(1, keepdim=True) / lengths[:, None, None]
    return (xy - center).masked_fill(~valid[:, :, None], 0)


def group_strokes(
    strokes: torch.Tensor, lengths: torch.Tensor, coordinate_scale: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split (B, T, 3) into strokes``.

    Returns (notebook names in parentheses)
        stroke_pts  (N, P, 4)  (padded)          per-stroke local shape, zero-padded
        stroke_lens (N,)       (stroke_lengths)  points per stroke
        stroke_xy   (N, 2)     (locations)       centred start point of each stroke
        sketch_ids  (N,)       (sketch_ids)      which sketch each stroke belongs to
        orders      (N,)       (orders)          0-based stroke index within its sketch
        n_strokes   (B,)       (counts)          strokes per sketch
    Strokes are sketch-major in drawing order.
    """
    B, T, _ = strokes.shape
    dev = strokes.device
    lengths = lengths.to(dev)
    if torch.any(lengths < 1) or torch.any(lengths > T):
        raise ValueError("lengths must lie in [1, T]")
    valid = torch.arange(T, device=dev)[None, :] < lengths[:, None]
    xy_all = centred_coordinates(strokes, lengths, coordinate_scale)

    starts = torch.ones(B, T, dtype=torch.bool, device=dev)
    starts[:, 1:] = strokes[:, :-1, 2] > 0.5          # a new stroke follows every pen-up
    starts &= valid

    xy = xy_all[valid]                                  # (total_points, 2)
    flat_starts = starts[valid]
    point_index = torch.arange(len(xy), device=dev)
    first = point_index[flat_starts]                    # index of each stroke's first point
    stroke_ids = flat_starts.long().cumsum(0) - 1
    ranks = point_index - first[stroke_ids]
    stroke_lens = torch.bincount(stroke_ids, minlength=len(first))
    sketch_ids = torch.arange(B, device=dev)[:, None].expand(B, T)[valid][first]
    n_strokes = torch.bincount(sketch_ids, minlength=B)

    stroke_xy = xy[first]
    local_xy = xy - stroke_xy[stroke_ids]
    ends = ranks == stroke_lens[stroke_ids] - 1
    shape = torch.cat([local_xy, (~ends).to(xy.dtype)[:, None], ends.to(xy.dtype)[:, None]], dim=1)
    stroke_pts = strokes.new_zeros(len(first), int(stroke_lens.max()), 4)
    stroke_pts[stroke_ids, ranks] = shape
    offsets = n_strokes.cumsum(0) - n_strokes
    orders = torch.arange(len(first), device=dev) - offsets[sketch_ids]
    return stroke_pts, stroke_lens, stroke_xy, sketch_ids, orders, n_strokes
