"""Test the processed-format reader (Sihui's ``preprocess.ipynb`` output).

By default it writes a small temporary synthetic fixture in the processed
layout (``<category>/{train,valid,test}.npz`` with ``points``/``offsets``/
``labels``/``ids`` + ``manifest.json``) and deletes it afterwards. With
``--data-dir`` it also runs on real processed shards::

    python -m tests.processed_format_test
    python -m tests.processed_format_test --data-dir data/processed_full --categories cat apple bicycle

Checks: sketches are rebuilt exactly from points + offsets, global labels are
remapped to 0..N-1 in config order, coordinates are not scaled again,
DataLoader padding + lengths, the baseline BiLSTM forward pass, loss/backprop,
and a 1-epoch train + evaluate run through ``src.train`` / ``src.evaluate``.
Accuracies printed here come from synthetic data and are NOT results.
"""

import argparse
import json
import os
import tempfile

import numpy as np
import torch

from src.data import build_dataloaders, build_datasets
from src.evaluate import evaluate
from src.models.classifier import build_model
from src.train import train
from src.utils import set_seed
from tests.smoke_test import check, make_synthetic_sketch

# Fake global indices, deliberately not 0..2, to check the remapping.
FIXTURE_CLASSES = {"apple": 3, "bicycle": 26, "cat": 57}
BASELINE_MODEL = {
    "encoder": "lstm", "hidden_dim": 256, "num_layers": 1, "bidirectional": True,
    "dropout": 0.0, "pooling": "last", "embedding_dim": None,
}


def write_processed_fixture(data_dir: str, sizes=(30, 10, 10), seed=0) -> dict:
    """Write a tiny processed-format dataset; return the sketches written per category/split."""
    rng = np.random.default_rng(seed)
    written = {}
    for kind, (name, global_label) in enumerate(sorted(FIXTURE_CLASSES.items())):
        os.makedirs(os.path.join(data_dir, name), exist_ok=True)
        for split, n in zip(("train", "valid", "test"), sizes):
            # Float values standing in for already-normalized dx, dy.
            seqs = []
            for _ in range(n):
                s = make_synthetic_sketch(kind, rng).astype(np.float32)
                s[:, :2] /= 37.0
                seqs.append(s)
            lengths = np.array([len(s) for s in seqs], dtype=np.int64)
            np.savez_compressed(
                os.path.join(data_dir, name, f"{split}.npz"),
                points=np.concatenate(seqs).astype(np.float32),
                offsets=np.concatenate(([0], np.cumsum(lengths))),
                labels=np.full(n, global_label, dtype=np.int64),
                ids=np.asarray([f"{name}/full/{split}/{i}" for i in range(n)]),
            )
            written[(name, split)] = seqs
    manifest = {
        "archive_version": "full", "class_to_index": FIXTURE_CLASSES,
        "representation": "stroke-3", "normalization": {"fit_split": "train", "scale": 0.1},
        "note": "synthetic test fixture, not real data",
    }
    with open(os.path.join(data_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    return written


def run(data_dir, categories, workdir, written=None):
    set_seed(0)
    C = len(categories)
    kw = dict(data_format="processed", normalize=False, seed=0)

    print("[1] normalization switch")
    try:
        build_datasets(data_dir, categories, 5, 5, 5, data_format="processed", splits=("valid",))
        check(False, "normalize=True on processed data raises")
    except ValueError:
        check(True, "normalize=True on processed data raises")
    ds = build_datasets(data_dir, categories, None, None, None, **kw)
    for split, d in ds.items():
        check(d.scale_factor == 1.0, f"{split}: scale factor is 1.0 (no second normalization)")

    print("[2] reconstruction from points + offsets")
    if written is not None:
        for split, d in ds.items():
            expected = [s for name in categories for s in written[(name, split)]]
            same = len(expected) == len(d) and all(
                torch.equal(d.sketches[i], torch.from_numpy(e)) for i, e in enumerate(expected)
            )
            check(same, f"{split}: {len(d)} sketches identical to the fixture (values, lengths, pen)")
    for split, d in ds.items():
        shapes_ok = all(s.ndim == 2 and s.shape[1] == 3 and len(s) > 0 for s in d.sketches)
        check(shapes_ok, f"{split}: every sketch is (L, 3)")
        pen = torch.cat([s[:, 2] for s in d.sketches])
        check(bool(((pen == 0) | (pen == 1)).all()), f"{split}: pen column is 0/1")

    print("[3] label remapping")
    for split, d in ds.items():
        check(set(d.labels.tolist()) == set(range(C)), f"{split}: labels are 0..{C - 1}")
    if written is not None:
        counts = [len(written[(name, "train")]) for name in categories]
        expected = np.repeat(np.arange(C), counts)
        check(np.array_equal(ds["train"].labels, expected), "sketches of categories[i] get label i")

    print("[4] DataLoader padding + lengths")
    ds = build_datasets(data_dir, categories, 20, 5, 5, **kw)
    batch = next(iter(build_dataloaders(ds, batch_size=16)["train"]))
    B, T, F = batch["strokes"].shape
    check((B, F) == (16, 3), f"strokes batch shape {tuple(batch['strokes'].shape)} is (16, T_max, 3)")
    check(int(batch["lengths"].max()) == T, "T_max equals the longest length in the batch")
    i = int(batch["lengths"].argmin())
    check(batch["strokes"][i, batch["lengths"][i]:].abs().sum() == 0, "zero padding past each length")

    print("[5] baseline BiLSTM forward + loss + backprop")
    model = build_model(BASELINE_MODEL, ds["train"].input_dim, C)
    logits = model(batch["strokes"], batch["lengths"])
    check(logits.shape == (B, C), f"logits shape {tuple(logits.shape)}")
    loss = torch.nn.CrossEntropyLoss()(logits, batch["labels"])
    check(bool(torch.isfinite(loss)), f"cross-entropy is finite ({loss.item():.3f})")
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    check(all(g is not None and torch.isfinite(g).all() for g in grads), "all parameters get finite gradients")

    print("[6] src.train + src.evaluate on processed data (1 epoch, plumbing only)")
    out = os.path.join(workdir, "run")
    cfg = {
        "seed": 0,
        "output_dir": out,
        "data": {"data_dir": data_dir, "data_format": "processed", "normalize": False,
                 "categories": list(categories), "train_per_class": 20, "val_per_class": 5,
                 "test_per_class": 5, "max_seq_len": 250, "batch_size": 16},
        "model": dict(BASELINE_MODEL),
        "train": {"epochs": 1, "lr": 1e-3, "grad_clip": 1.0, "device": "cpu"},
    }
    train(cfg)
    check(os.path.isfile(os.path.join(out, "best.pt")), "wrote best.pt")
    ckpt = torch.load(os.path.join(out, "best.pt"), weights_only=False)
    check(ckpt["scale_factor"] == 1.0, "checkpoint stores scale factor 1.0")
    metrics = evaluate(os.path.join(out, "best.pt"), "test", device="cpu")
    check(metrics["num_samples"] == 5 * C, f"evaluate ran on {metrics['num_samples']} test sketches")
    print("\nPROCESSED-FORMAT TEST PASSED (numbers above are plumbing checks, not results)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="real processed directory (default: temporary synthetic fixture)")
    p.add_argument("--categories", nargs="+")
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if a.data_dir:
            if not a.categories:
                p.error("--categories is required with --data-dir")
            run(a.data_dir, a.categories, tmp)
        else:
            data_dir = os.path.join(tmp, "processed_fixture")
            written = write_processed_fixture(data_dir)
            print(f"using temporary synthetic processed fixture in {data_dir}")
            # Config order differs from both alphabetical and global-index order.
            run(data_dir, ["cat", "apple", "bicycle"], tmp, written)


if __name__ == "__main__":
    main()
