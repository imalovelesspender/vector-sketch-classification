"""Evaluation entry point.

Primary metric: classification accuracy.

Usage::

    # final test evaluation (run once, after model selection on valid)
    python -m src.evaluate --checkpoint results/lstm_baseline/best.pt --split test

    # also dump sketch embeddings z for retrieval experiments
    python -m src.evaluate --checkpoint results/lstm_baseline/best.pt --split train \
        --save-embeddings results/lstm_baseline/train_embeddings.npz

The model, categories, preprocessing (scale factor, stroke format) and subset
sizes are restored from the checkpoint so evaluation matches training.

TODO: Add error analysis for ambiguous or difficult sketches.
"""

import argparse
import os

import numpy as np
from torch.utils.data import DataLoader

from src.data import build_datasets, collate_sketches
from src.engine import extract_embeddings, run_epoch
from src.models.classifier import build_model
from src.utils import get_device, load_checkpoint, save_json


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--data-dir", help="override the data_dir stored in the checkpoint")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--save-embeddings", help="write embeddings/labels/preds to this .npz")
    return p.parse_args(argv)


def evaluate(checkpoint, split="test", data_dir=None, device="auto", batch_size=256, save_embeddings=None):
    ckpt = load_checkpoint(checkpoint)
    cfg = ckpt["config"]
    data_cfg = cfg["data"]
    device = get_device(device)
    per_class = {
        "train": data_cfg.get("train_per_class"),
        "valid": data_cfg.get("val_per_class"),
        "test": data_cfg.get("test_per_class"),
    }
    ds = build_datasets(
        data_dir or data_cfg["data_dir"],
        ckpt["categories"],
        train_per_class=per_class["train"],
        val_per_class=per_class["valid"],
        test_per_class=per_class["test"],
        max_seq_len=data_cfg.get("max_seq_len"),
        stroke_format=data_cfg.get("stroke_format", "stroke3"),
        seed=cfg.get("seed", 0),
        scale_factor=ckpt["scale_factor"],
        splits=(split,),
        data_format=data_cfg.get("data_format", "sketchrnn"),
        normalize=data_cfg.get("normalize", True),
    )
    # Fixed order (no shuffling), even for the train split, so exported
    # embeddings line up with dataset indices.
    loader = DataLoader(ds[split], batch_size=batch_size, shuffle=False, collate_fn=collate_sketches)

    model = build_model(cfg["model"], ckpt["input_dim"], len(ckpt["categories"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    metrics = run_epoch(model, loader, device)
    metrics.update(split=split, num_samples=len(ds[split]), checkpoint=checkpoint, epoch=ckpt.get("epoch"))
    print(f"{split}: loss {metrics['loss']:.4f} acc {metrics['acc']:.4f} (n={metrics['num_samples']})")
    out_dir = os.path.dirname(checkpoint)
    save_json(metrics, os.path.join(out_dir, f"metrics_{split}.json"))

    if save_embeddings:
        emb = extract_embeddings(model, loader, device)
        os.makedirs(os.path.dirname(save_embeddings) or ".", exist_ok=True)
        np.savez(save_embeddings, categories=np.array(ckpt["categories"]), **emb)
        print(f"saved {emb['embeddings'].shape} embeddings to {save_embeddings}")
    return metrics


def main(argv=None):
    a = parse_args(argv)
    evaluate(a.checkpoint, a.split, a.data_dir, a.device, a.batch_size, a.save_embeddings)


if __name__ == "__main__":
    main()
