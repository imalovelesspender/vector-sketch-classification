"""Training entry point.

Usage::

    python -m src.train --config configs/lstm_baseline.json
    python -m src.train --config configs/lstm_baseline.json --epochs 2 --categories cat apple

Trains on the ``train`` split, validates on ``valid`` every epoch, and writes to
``output_dir``: ``last.pt``, ``best.pt`` (highest validation accuracy),
``history.json``, ``curves.png`` and the resolved ``config.json``.
The test split is never touched here; use ``src.evaluate`` for the final test.
"""

import argparse
import copy
import json
import os

import torch

from src.data import build_dataloaders, build_datasets
from src.engine import run_epoch
from src.models.classifier import build_model
from src.utils import get_device, plot_curves, save_checkpoint, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--data-dir", help="override data.data_dir")
    p.add_argument("--categories", nargs="+", help="override data.categories")
    p.add_argument("--output-dir", help="override output_dir")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--device", help="override train.device (auto, cpu, cuda, mps)")
    p.add_argument("--seed", type=int, help="override seed")
    return p.parse_args(argv)


def load_config(args) -> dict:
    with open(args.config) as f:
        cfg = json.load(f)
    cfg.pop("_comment", None)
    if args.data_dir:
        cfg["data"]["data_dir"] = args.data_dir
    if args.categories:
        cfg["data"]["categories"] = args.categories
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.device:
        cfg["train"]["device"] = args.device
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def train(cfg: dict) -> dict:
    """Run training from a resolved config dict and return the history."""
    cfg = copy.deepcopy(cfg)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = get_device(train_cfg.get("device", "auto"))

    datasets = build_datasets(
        data_cfg["data_dir"],
        data_cfg["categories"],
        train_per_class=data_cfg.get("train_per_class"),
        val_per_class=data_cfg.get("val_per_class"),
        max_seq_len=data_cfg.get("max_seq_len"),
        stroke_format=data_cfg.get("stroke_format", "stroke3"),
        seed=seed,
        splits=("train", "valid"),
    )
    loaders = build_dataloaders(
        datasets, data_cfg.get("batch_size", 64), data_cfg.get("num_workers", 0), seed
    )
    train_ds = datasets["train"]
    print(
        f"device={device} classes={train_ds.num_classes} "
        f"train={len(train_ds)} valid={len(datasets['valid'])} "
        f"scale_factor={train_ds.scale_factor:.4f}"
    )

    model = build_model(model_cfg, train_ds.input_dim, train_ds.num_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg.get("weight_decay", 0.0)
    )
    meta = {
        "config": cfg,
        "categories": train_ds.categories,
        "scale_factor": train_ds.scale_factor,
        "input_dim": train_ds.input_dim,
    }
    save_json(cfg, os.path.join(out_dir, "config.json"))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    for epoch in range(1, train_cfg["epochs"] + 1):
        tr = run_epoch(model, loaders["train"], device, optimizer, train_cfg.get("grad_clip"))
        va = run_epoch(model, loaders["valid"], device)
        for k, v in (("train", tr), ("val", va)):
            history[f"{k}_loss"].append(v["loss"])
            history[f"{k}_acc"].append(v["acc"])
        print(
            f"epoch {epoch:3d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} "
            f"| valid loss {va['loss']:.4f} acc {va['acc']:.4f}"
        )
        save_checkpoint(os.path.join(out_dir, "last.pt"), model, optimizer, epoch=epoch, val=va, **meta)
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, optimizer, epoch=epoch, val=va, **meta)
        save_json(history, os.path.join(out_dir, "history.json"))
        plot_curves(history, os.path.join(out_dir, "curves.png"))

    print(f"best valid acc {best_acc:.4f}; outputs in {out_dir}")
    return history


def main(argv=None):
    train(load_config(parse_args(argv)))


if __name__ == "__main__":
    main()
