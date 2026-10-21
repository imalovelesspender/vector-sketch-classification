"""Smoke test for the LSTM classification pipeline.

Runs on synthetic data in the Sketch-RNN stroke-3 ``.npz`` format by default,
or on a tiny subset of real files with ``--data-dir``::

    python -m tests.smoke_test
    python -m tests.smoke_test --data-dir data/sketchrnn --categories cat apple

Checks: dataset/padding shapes, forward pass, padding invariance, backprop,
loss decreasing when overfitting one tiny batch, checkpoint save/load, and a
2-epoch end-to-end run of ``src.train`` + ``src.evaluate``.
The numbers it prints are sanity checks, not experimental results.
"""

import argparse
import os
import tempfile

import numpy as np
import torch

from src.data import build_dataloaders, build_datasets, collate_sketches
from src.evaluate import evaluate
from src.models.classifier import build_model
from src.train import train
from src.utils import load_checkpoint, save_checkpoint, set_seed

SYNTH_CATEGORIES = ["synth_circle", "synth_zigzag", "synth_line"]


def make_synthetic_sketch(kind: int, rng: np.random.Generator) -> np.ndarray:
    """One stroke-3 sketch (int16 ``(n, 3)``) whose shape depends on ``kind``."""
    n = int(rng.integers(20, 80))
    t = np.linspace(0, 2 * np.pi, n)
    if kind == 0:
        pts = np.stack([np.cos(t), np.sin(t)], 1) * 100
    elif kind == 1:
        pts = np.stack([t * 20, np.sign(np.sin(6 * t)) * 30], 1)
    else:
        pts = np.stack([t * 30, t * 10], 1)
    pts += rng.normal(0, 3, pts.shape)
    d = np.diff(np.vstack([[0, 0], pts]), axis=0)
    pen = np.zeros((n, 1))
    pen[rng.choice(n - 1, size=int(rng.integers(0, 3)), replace=False)] = 1
    pen[-1] = 1
    return np.hstack([d, pen]).round().astype(np.int16)


def write_synthetic_npz(data_dir: str, sizes=(60, 20, 20), seed=0) -> None:
    rng = np.random.default_rng(seed)
    for kind, name in enumerate(SYNTH_CATEGORIES):
        splits = {}
        for split, n in zip(("train", "valid", "test"), sizes):
            arr = np.empty(n, dtype=object)
            for i in range(n):
                arr[i] = make_synthetic_sketch(kind, rng)
            splits[split] = arr
        np.savez(os.path.join(data_dir, f"{name}.npz"), **splits)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def run(data_dir, categories, workdir):
    set_seed(0)
    print("[1] dataset and padding")
    ds = build_datasets(data_dir, categories, 40, 10, 10, seed=0)
    batch = next(iter(build_dataloaders(ds, batch_size=8)["train"]))
    B, T, F = batch["strokes"].shape
    check((B, F) == (8, 3), f"strokes batch shape {tuple(batch['strokes'].shape)} is (8, T_max, 3)")
    check(int(batch["lengths"].max()) == T, "T_max equals the longest length in the batch")
    i = int(batch["lengths"].argmin())
    check(
        batch["strokes"][i, batch["lengths"][i]:].abs().sum() == 0,
        "positions past each sequence length are zero-padded",
    )
    check(len(ds["train"]) == 40 * len(categories), f"subset size {len(ds['train'])} = 40 per class")
    check(ds["valid"].scale_factor == ds["train"].scale_factor, "valid reuses the train scale factor")

    print("[2] forward pass")
    C = len(categories)
    for pooling in ("last", "mean"):
        model = build_model({"hidden_dim": 32, "pooling": pooling}, F, C)
        logits, z = model(batch["strokes"], batch["lengths"], return_embedding=True)
        check(logits.shape == (B, C), f"[{pooling}] logits shape {tuple(logits.shape)}")
        check(z.shape == (B, model.encoder.embedding_dim), f"[{pooling}] embedding z shape {tuple(z.shape)}")
        model.eval()
        with torch.no_grad():
            z_batch = model.embed(batch["strokes"], batch["lengths"])[i]
            L = batch["lengths"][i:i + 1]
            z_alone = model.embed(batch["strokes"][i:i + 1, :L], L)[0]
        check(torch.allclose(z_batch, z_alone, atol=1e-5), f"[{pooling}] padding does not change z")

    print("[3] backprop")
    model = build_model({"hidden_dim": 32}, F, C)
    loss = torch.nn.functional.cross_entropy(model(batch["strokes"], batch["lengths"]), batch["labels"])
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    check(all(g is not None and torch.isfinite(g).all() for g in grads), "all parameters get finite gradients")

    print("[4] overfit one tiny batch")
    model = build_model({"hidden_dim": 64}, F, C)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(batch["strokes"], batch["lengths"]), batch["labels"])
        loss.backward()
        opt.step()
        losses.append(loss.item())
    check(losses[-1] < 0.5 * losses[0], f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    print("[5] checkpoint save/load")
    path = os.path.join(workdir, "ckpt", "tmp.pt")
    save_checkpoint(path, model, opt, categories=list(categories))
    restored = build_model({"hidden_dim": 64}, F, C)
    restored.load_state_dict(load_checkpoint(path)["model_state"])
    model.eval(); restored.eval()
    with torch.no_grad():
        same = torch.allclose(model(batch["strokes"], batch["lengths"]), restored(batch["strokes"], batch["lengths"]))
    check(same, "restored model gives identical logits")

    print("[6] end-to-end train + evaluate (2 epochs)")
    out = os.path.join(workdir, "run")
    cfg = {
        "seed": 0,
        "output_dir": out,
        "data": {"data_dir": data_dir, "categories": list(categories), "train_per_class": 40,
                 "val_per_class": 10, "test_per_class": 10, "batch_size": 16},
        "model": {"encoder": "lstm", "hidden_dim": 32},
        "train": {"epochs": 2, "lr": 1e-3, "grad_clip": 1.0, "device": "cpu"},
    }
    history = train(cfg)
    check(len(history["val_acc"]) == 2, "history has 2 epochs")
    for f in ("best.pt", "last.pt", "history.json", "curves.png", "config.json"):
        check(os.path.isfile(os.path.join(out, f)), f"wrote {f}")
    emb_path = os.path.join(out, "test_embeddings.npz")
    evaluate(os.path.join(out, "best.pt"), "test", device="cpu", save_embeddings=emb_path)
    emb = np.load(emb_path)
    check(emb["embeddings"].shape == (10 * C, 64), f"exported embeddings {emb['embeddings'].shape}")
    print("\nSMOKE TEST PASSED")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="directory with real <category>.npz files (default: synthetic)")
    p.add_argument("--categories", nargs="+")
    p.add_argument("--workdir", help="where to write outputs (default: a temp dir)")
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = a.workdir or tmp
        if a.data_dir:
            data_dir, cats = a.data_dir, a.categories
            if not cats:
                p.error("--categories is required with --data-dir")
        else:
            data_dir = os.path.join(workdir, "synthetic")
            os.makedirs(data_dir, exist_ok=True)
            write_synthetic_npz(data_dir)
            cats = SYNTH_CATEGORIES
            print(f"using synthetic stroke-3 data in {data_dir}")
        run(data_dir, cats, workdir)


if __name__ == "__main__":
    main()
