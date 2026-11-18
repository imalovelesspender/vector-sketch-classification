"""Shared train / evaluation loops."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.stroke_split import STROKE_KEYS


def _extra_inputs(batch: dict, device: torch.device) -> dict:
    """Stroke tensors from the stroke_based collate, if present (else empty)."""
    return {k: batch[k].to(device) for k in STROKE_KEYS if k in batch}


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> dict:
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates.

    Returns mean cross-entropy loss and accuracy over all samples.
    """
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            strokes = batch["strokes"].to(device)
            lengths = batch["lengths"]
            labels = batch["labels"].to(device)
            logits = model(strokes, lengths, **_extra_inputs(batch, device))
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                (loss / labels.size(0)).backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return {"loss": total_loss / count, "acc": correct / count}


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader, device: torch.device) -> dict:
    """Collect embeddings z, labels and predictions for every sample in ``loader``."""
    model.eval()
    zs, labels, preds = [], [], []
    for batch in loader:
        logits, z = model(
            batch["strokes"].to(device), batch["lengths"], return_embedding=True,
            **_extra_inputs(batch, device),
        )
        zs.append(z.cpu().numpy())
        labels.append(batch["labels"].numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return {
        "embeddings": np.concatenate(zs),
        "labels": np.concatenate(labels),
        "preds": np.concatenate(preds),
    }
