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
from src.utils import get_device, load_checkpoint, plot_curves, save_checkpoint, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="JSON config file")
    p.add_argument("--data-dir", help="override data.data_dir")
    p.add_argument("--categories", nargs="+", help="override data.categories")
    p.add_argument("--output-dir", help="override output_dir")
    p.add_argument("--epochs", type=int, help="override train.epochs")
    p.add_argument("--device", help="override train.device (auto, cpu, cuda, mps)")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument("--resume", action="store_true",
                   help="continue from <output_dir>/last.pt if it exists (optimizer state, epoch, history)")
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
    if args.resume:
        cfg["resume"] = True
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
        data_format=data_cfg.get("data_format", "sketchrnn"),
        normalize=data_cfg.get("normalize", True),
    )
    from src.data.quickdraw import load_manifest
    from src.data.stroke_collate import get_collate

    representation = data_cfg.get("representation", "point_based")
    coordinate_scale = data_cfg.get("coordinate_scale")
    if representation == "stroke_based" and coordinate_scale is None:
        manifest = load_manifest(data_cfg["data_dir"]) or {}
        coordinate_scale = manifest.get("normalization", {}).get("scale", manifest.get("scale_factor"))
        if coordinate_scale is None:
            raise ValueError("stroke_based: set data.coordinate_scale or provide manifest.json with normalization.scale")
        if cfg["model"].get("encoder") == "stroke_embed":
            cfg["model"].setdefault("coordinate_scale", coordinate_scale)
    collate_fn = get_collate(representation, coordinate_scale)
    loaders = build_dataloaders(
        datasets, data_cfg.get("batch_size", 64), data_cfg.get("num_workers", 0), seed,
        collate_fn=collate_fn
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

    # Optional Weights & Biases logging: enabled when WANDB_PROJECT is set (or cfg["wandb"]["project"]).
    # Nothing changes when neither is set or wandb is not installed.
    wb = _init_wandb(cfg, meta)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    start_epoch = 1
    last_path = os.path.join(out_dir, "last.pt")
    if cfg.get("resume") and os.path.isfile(last_path):
        ck = load_checkpoint(last_path, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck["epoch"] + 1
        hist_path = os.path.join(out_dir, "history.json")
        if os.path.isfile(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            best_acc = max(history["val_acc"], default=-1.0)
        print(f"resumed from {last_path}: epoch {ck['epoch']} done, best valid acc {best_acc:.4f}")
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        tr = run_epoch(model, loaders["train"], device, optimizer, train_cfg.get("grad_clip"))
        va = run_epoch(model, loaders["valid"], device)
        for k, v in (("train", tr), ("val", va)):
            history[f"{k}_loss"].append(v["loss"])
            history[f"{k}_acc"].append(v["acc"])
        print(
            f"epoch {epoch:3d} | train loss {tr['loss']:.4f} acc {tr['acc']:.4f} "
            f"| valid loss {va['loss']:.4f} acc {va['acc']:.4f}"
        )
        if wb is not None:
            wb.log({"epoch": epoch, "train/loss": tr["loss"], "train/acc": tr["acc"],
                    "valid/loss": va["loss"], "valid/acc": va["acc"],
                    "valid/best_acc": max(best_acc, va["acc"])}, step=epoch)
        save_checkpoint(os.path.join(out_dir, "last.pt"), model, optimizer, epoch=epoch, val=va, **meta)
        if va["acc"] > best_acc:
            best_acc = va["acc"]
            save_checkpoint(os.path.join(out_dir, "best.pt"), model, optimizer, epoch=epoch, val=va, **meta)
        save_json(history, os.path.join(out_dir, "history.json"))
        plot_curves(history, os.path.join(out_dir, "curves.png"))

    print(f"best valid acc {best_acc:.4f}; outputs in {out_dir}")
    if wb is not None:
        wb.summary["best_valid_acc"] = best_acc
        wb.finish()
    return history


def _init_wandb(cfg: dict, meta: dict):
    wcfg = cfg.get("wandb", {}) or {}
    project = wcfg.get("project") or os.environ.get("WANDB_PROJECT")
    if not project or wcfg.get("enabled") is False:
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping W&B logging (pip install wandb)")
        return None
    run = wandb.init(
        project=project,
        entity=wcfg.get("entity") or os.environ.get("WANDB_ENTITY"),
        name=wcfg.get("name") or os.environ.get("WANDB_NAME") or os.path.basename(cfg["output_dir"].rstrip("/")),
        group=wcfg.get("group") or os.environ.get("WANDB_RUN_GROUP"),
        tags=wcfg.get("tags"),
        config={**cfg, "num_classes": len(meta["categories"]), "input_dim": meta["input_dim"]},
        dir=cfg["output_dir"],
    )
    return run


def main(argv=None):
    train(load_config(parse_args(argv)))


if __name__ == "__main__":
    main()
