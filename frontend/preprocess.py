"""Turn canvas strokes into the model's input format.

The training data comes from Sketch-RNN archives (QuickDraw "simplified"
drawings stored as stroke-3) and is then normalized by
`data_preprocessing/preprocess.ipynb`. A sketch drawn in the frontend goes
through the same stages so that the backend sees in-distribution input:

1. QuickDraw simplification: align to the top-left corner, scale the longest
   side to 255, simplify every stroke with Ramer-Douglas-Peucker (epsilon 2).
2. Stroke-3 encoding: rows of [dx, dy, pen_up]; pen_up=1 on the last point of
   each stroke. As in Sketch-RNN the first point is the origin and has no row.
3. Notebook normalization: divide dx/dy by the longest bounding-box dimension,
   then by the scale fitted on the training split (`manifest.json`).
"""

import numpy as np


def rdp(points, epsilon):
    """Ramer-Douglas-Peucker simplification of one [N, 2] polyline."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        start, chord = points[first], points[last] - points[first]
        offsets = points[first + 1:last] - start
        norm = np.hypot(*chord)
        if norm == 0:
            distances = np.hypot(offsets[:, 0], offsets[:, 1])
        else:
            distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / norm
        farthest = int(np.argmax(distances))
        if distances[farthest] > epsilon:
            split = first + 1 + farthest
            keep[split] = True
            stack.extend([(first, split), (split, last)])
    return points[keep]


def strokes_to_stroke3(strokes, box=255.0, rdp_epsilon=2.0):
    """Canvas strokes (list of [N, 2] arrays) -> raw stroke-3 [T, 3] float32.

    Returns an empty [0, 3] array when there is nothing to encode.
    """
    strokes = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) > 0]
    if not strokes:
        return np.zeros((0, 3), dtype=np.float32)
    every = np.concatenate(strokes)
    lower = every.min(axis=0)
    extent = (every.max(axis=0) - lower).max()
    factor = box / extent if extent > 0 else 1.0

    rows = []
    for stroke in strokes:
        simplified = np.round(rdp((stroke - lower) * factor, rdp_epsilon))
        # Rounding can merge neighbours; drop repeated points.
        distinct = np.ones(len(simplified), dtype=bool)
        distinct[1:] = np.any(np.diff(simplified, axis=0) != 0, axis=1)
        simplified = simplified[distinct]
        pen_up = np.zeros(len(simplified))
        pen_up[-1] = 1
        rows.append(np.column_stack([simplified, pen_up]))
    absolute = np.concatenate(rows)
    stroke3 = absolute[1:].copy()
    stroke3[:, :2] -= absolute[:-1, :2]
    return stroke3.astype(np.float32)


def normalize(stroke3, scale):
    """Per-sketch unit scaling followed by the training scale (see notebook)."""
    points = np.array(stroke3, dtype=np.float32)
    if len(points) == 0:
        return points
    positions = np.cumsum(points[:, :2], axis=0, dtype=np.float64)
    dimension = (positions.max(axis=0) - positions.min(axis=0)).max()
    if dimension > 0:
        points[:, :2] /= np.float32(dimension)
    points[:, :2] /= scale
    return points


def preprocess(strokes, scale, box=255.0, rdp_epsilon=2.0):
    """Canvas strokes -> normalized stroke-3 [T, 3] float32, ready for the model."""
    return normalize(strokes_to_stroke3(strokes, box, rdp_epsilon), scale)


def stroke3_to_lines(stroke3):
    """Stroke-3 -> list of [N, 2] absolute polylines (for drawing thumbnails)."""
    stroke3 = np.asarray(stroke3, dtype=np.float64).reshape(-1, 3)
    positions = np.concatenate([[[0.0, 0.0]], np.cumsum(stroke3[:, :2], axis=0)])
    # Point i+1 ends a stroke when row i has pen_up=1.
    ends = np.flatnonzero(stroke3[:, 2] == 1) + 2
    return [line for line in np.split(positions, ends) if len(line) > 0]
