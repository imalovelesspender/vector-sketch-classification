"""QuickDraw Sketch-RNN ``.npz`` dataset and dataloaders.

Expected file layout (one file per category, as released by Sketch-RNN at
``https://storage.googleapis.com/quickdraw_dataset/sketchrnn/<category>.npz``)::

    data/sketchrnn/
        cat.npz
        apple.npz
        ...

Inspected structure of each ``.npz`` file (e.g. ``cat.npz``):

* keys ``"train"`` (70000), ``"valid"`` (2500), ``"test"`` (2500)
* each key holds a 1-D ``object`` array of sketches
* each sketch is an ``int16`` array of shape ``(seq_len, 3)`` in **stroke-3**
  format: ``(dx, dy, pen_lift)`` where ``dx, dy`` are offsets from the previous
  point and ``pen_lift == 1`` means the pen is lifted after this point
  (end of a stroke)
* sequence lengths vary per sketch (e.g. 37 to 129 points for ``cat``)

Files are loaded with ``allow_pickle=True`` and ``encoding="latin1"`` because the
arrays are pickled object arrays written by Python 2.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "valid", "test")


def load_category(path: str, split: str) -> np.ndarray:
    """Load one split of one category ``.npz`` file as an object array of sketches."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    with np.load(path, encoding="latin1", allow_pickle=True) as data:
        if split not in data.files:
            raise KeyError(f"{path} has no {split!r} split (found {data.files})")
        return data[split]


def to_stroke5(stroke3: np.ndarray) -> np.ndarray:
    """Convert stroke-3 ``(dx, dy, pen_lift)`` to stroke-5 ``(dx, dy, p1, p2, p3)``.

    ``p1``: pen down, ``p2``: pen lifted after this point, ``p3``: end of sketch
    (set on the last point). Unlike Sketch-RNN we do not pad with ``p3`` here;
    padding is handled in :func:`collate_sketches` with a lengths tensor.
    """
    n = len(stroke3)
    out = np.zeros((n, 5), dtype=np.float32)
    out[:, :2] = stroke3[:, :2]
    out[:, 3] = stroke3[:, 2]
    out[:, 2] = 1 - stroke3[:, 2]
    out[-1, 2:] = (0, 0, 1)
    return out


def compute_scale_factor(sketches: Sequence[np.ndarray]) -> float:
    """Standard deviation of all ``dx, dy`` offsets (Sketch-RNN normalization)."""
    offsets = np.concatenate([s[:, :2].reshape(-1) for s in sketches]).astype(np.float64)
    return float(offsets.std())


class QuickDrawStrokeDataset(Dataset):
    """Multi-category QuickDraw stroke-sequence dataset for classification.

    Args:
        data_dir: directory containing ``<category>.npz`` files.
        categories: category names; label ``i`` is ``categories[i]``.
        split: ``"train"``, ``"valid"`` or ``"test"`` (the splits shipped in the files).
        samples_per_class: if set, randomly sample at most this many sketches per
            category (configurable subset size). ``None`` uses all sketches.
        max_seq_len: if set, drop sketches longer than this many points.
        scale_factor: divide ``dx, dy`` by this value. If ``None`` it is computed
            from this dataset; for valid/test pass the training set's value.
        stroke_format: ``"stroke3"`` (3 features) or ``"stroke5"`` (5 features).
        seed: seed for subset sampling.
    """

    def __init__(
        self,
        data_dir: str,
        categories: Sequence[str],
        split: str = "train",
        samples_per_class: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        scale_factor: Optional[float] = None,
        stroke_format: str = "stroke3",
        seed: int = 0,
    ):
        if stroke_format not in ("stroke3", "stroke5"):
            raise ValueError(f"unknown stroke_format {stroke_format!r}")
        self.categories = list(categories)
        self.split = split
        self.stroke_format = stroke_format
        rng = np.random.default_rng(seed)

        sketches: List[np.ndarray] = []
        labels: List[int] = []
        for label, name in enumerate(self.categories):
            path = os.path.join(data_dir, f"{name}.npz")
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} not found. Download it manually, see data/README.md."
                )
            cat_sketches = list(load_category(path, split))
            if max_seq_len is not None:
                cat_sketches = [s for s in cat_sketches if len(s) <= max_seq_len]
            if samples_per_class is not None and samples_per_class < len(cat_sketches):
                idx = rng.choice(len(cat_sketches), samples_per_class, replace=False)
                cat_sketches = [cat_sketches[i] for i in sorted(idx)]
            sketches.extend(cat_sketches)
            labels.extend([label] * len(cat_sketches))

        if not sketches:
            raise ValueError("dataset is empty; check categories / max_seq_len")
        self.scale_factor = (
            compute_scale_factor(sketches) if scale_factor is None else float(scale_factor)
        )
        self.sketches = [self._preprocess(s) for s in sketches]
        self.labels = np.asarray(labels, dtype=np.int64)

    @property
    def num_classes(self) -> int:
        return len(self.categories)

    @property
    def input_dim(self) -> int:
        return 3 if self.stroke_format == "stroke3" else 5

    def _preprocess(self, s: np.ndarray) -> torch.Tensor:
        s = s.astype(np.float32)
        s[:, :2] /= self.scale_factor
        if self.stroke_format == "stroke5":
            s = to_stroke5(s)
        return torch.from_numpy(s)

    def __len__(self) -> int:
        return len(self.sketches)

    def __getitem__(self, idx: int):
        return self.sketches[idx], int(self.labels[idx])


def collate_sketches(batch) -> Dict[str, torch.Tensor]:
    """Pad variable-length sketches to the longest in the batch.

    Returns a dict with
    ``strokes`` (B, T_max, F) zero-padded, ``lengths`` (B,) and ``labels`` (B,).
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    strokes = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    return {
        "strokes": strokes,
        "lengths": lengths,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_datasets(
    data_dir: str,
    categories: Sequence[str],
    train_per_class: Optional[int] = None,
    val_per_class: Optional[int] = None,
    test_per_class: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    stroke_format: str = "stroke3",
    seed: int = 0,
    scale_factor: Optional[float] = None,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, QuickDrawStrokeDataset]:
    """Build datasets for the requested splits.

    The normalization scale factor is computed on the training split (unless
    given) and reused for valid/test so no statistics leak from held-out data.
    """
    sizes = {"train": train_per_class, "valid": val_per_class, "test": test_per_class}
    kwargs = dict(max_seq_len=max_seq_len, stroke_format=stroke_format, seed=seed)
    datasets: Dict[str, QuickDrawStrokeDataset] = {}
    if scale_factor is None:
        train = QuickDrawStrokeDataset(
            data_dir, categories, "train", sizes["train"], **kwargs
        )
        scale_factor = train.scale_factor
        if "train" in splits:
            datasets["train"] = train
    for split in splits:
        if split not in datasets:
            datasets[split] = QuickDrawStrokeDataset(
                data_dir, categories, split, sizes[split],
                scale_factor=scale_factor, **kwargs,
            )
    return datasets


def build_dataloaders(
    datasets: Dict[str, QuickDrawStrokeDataset],
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 0,
) -> Dict[str, DataLoader]:
    """Wrap datasets in DataLoaders; only the train loader is shuffled."""
    loaders = {}
    for split, ds in datasets.items():
        generator = torch.Generator().manual_seed(seed) if split == "train" else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_sketches,
            num_workers=num_workers,
            generator=generator,
        )
    return loaders
