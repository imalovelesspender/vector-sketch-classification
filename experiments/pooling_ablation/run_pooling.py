"""Pooling ablation using Anson's adapter and the pinned historical pipeline.

The companion notebook prepares configurations. This file runs one experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE / "historical"
ARCHIVE = HERE / "reference"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources():
    expected = read_json(ARCHIVE / "historical_source_hashes.json")
    for relative, digest in expected.items():
        if sha256(HISTORICAL / relative) != digest:
            raise ValueError(f"Historical source differs: {relative}")
    return expected


def make_config(data_dir, output_dir, pooling="mean", epochs=10):
    """Preserve all historical choices except paths, pooling and epoch budget."""
    if pooling not in ("last", "mean") or epochs < 1:
        raise ValueError("Expected last/mean pooling and positive epochs")
    cfg = read_json(ARCHIVE / "baseline_config.json")
    cfg["data"]["data_dir"] = str(Path(data_dir).resolve())
    prep = cfg["data"]["preprocessing"]
    prep["manifest"] = str(Path(data_dir).resolve() / "manifest.json")
    prep["adapter"] = str(HISTORICAL / "psc_run" / "run_processed.py")
    if Path(prep["manifest"]).is_file():
        prep["manifest_sha256"] = sha256(prep["manifest"])
    cfg["output_dir"] = str(Path(output_dir).resolve())
    cfg["model"]["pooling"] = pooling
    cfg["train"]["epochs"] = epochs
    return cfg


def comparable(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.pop("output_dir")
    cfg["model"].pop("pooling")
    cfg["data"].pop("data_dir")
    cfg["data"]["preprocessing"].pop("manifest")
    cfg["data"]["preprocessing"].pop("adapter")
    return cfg


def check_config(cfg):
    reference = read_json(ARCHIVE / "baseline_config.json")
    # Longer budgets are supported, but each comparison must have the same budget.
    reference["train"]["epochs"] = cfg["train"]["epochs"]
    if cfg["data"]["preprocessing"]["manifest_sha256"] != reference["data"]["preprocessing"]["manifest_sha256"]:
        # A private rebuild has its own metadata hash and must pass its full audit.
        # Original historical checkpoints still require identical hashes in comparable().
        check_manifest(cfg)
        reference["data"]["preprocessing"]["manifest_sha256"] = cfg["data"]["preprocessing"]["manifest_sha256"]
    if comparable(cfg) != comparable(reference):
        raise ValueError("Unexpected changes beyond pooling, epoch budget and paths")
    if cfg["model"]["pooling"] not in ("last", "mean"):
        raise ValueError("Unsupported pooling")
    if not isinstance(cfg["train"]["epochs"], int) or cfg["train"]["epochs"] < 1:
        raise ValueError("Invalid epoch budget")


def check_manifest(cfg):
    path = Path(cfg["data"]["data_dir"]) / "manifest.json"
    expected = cfg["data"]["preprocessing"]["manifest_sha256"]
    if sha256(path) != expected:
        raise ValueError("Processed manifest differs from Anson's data. Resolve provenance before training.")
    manifest = read_json(path)
    categories = sorted(manifest["class_to_index"], key=manifest["class_to_index"].get)
    if categories != cfg["data"]["categories"] or len(categories) != 345:
        raise ValueError("Category order differs from baseline")
    if manifest["archive_version"] != "full" or manifest["normalization"]["fit_split"] != "train":
        raise ValueError("Expected training-normalized full archives")
    original = read_json(ARCHIVE / "baseline_config.json")["data"]["preprocessing"]["manifest_sha256"]
    if expected != original:
        audit = read_json(path.parent / "preparation_audit.json")
        flags = ("complete", "official_source_checksums_verified", "all_shards_readback_verified",
                 "normalization_recomputed_from_all_training_points")
        if not all(audit.get(flag) is True for flag in flags):
            raise ValueError("Private rebuild audit is incomplete")
        if audit.get("manifest_sha256") != expected or audit.get("preprocessor_sha256") != sha256(HERE / "prepare_data.py"):
            raise ValueError("Private rebuild provenance differs")
        if (manifest.get("preprocessing_version") != 2 or manifest.get("representation") != "stroke-3"
                or manifest.get("subsampling") is not False or manifest.get("length_cap") is not None
                or manifest.get("per_sketch_scaling") != {"method": "max_bbox_dimension", "target": 1.0, "degenerate": "identity"}
                or abs(manifest["normalization"]["scale"] - 0.12035210854952101) > 1e-12
                or manifest.get("split_counts") != {"train": 43988874, "valid": 862500, "test": 862500}
                or audit.get("source_archives") != 345):
            raise ValueError("Private rebuild does not match the preprocessing protocol")
    return manifest


def load_adapter():
    # Isolated process: never import the results branch's placeholder src/train.py.
    if "src" in sys.modules:
        raise RuntimeError("Run in a fresh Python process to avoid another src package")
    sys.path.insert(0, str(HISTORICAL))
    sys.path.insert(0, str(HISTORICAL / "psc_run"))
    import run_processed
    return run_processed


def execute(config_path, account, reference_run=None):
    cfg = read_json(config_path)
    check_config(cfg)
    sources = verify_sources()
    check_manifest(cfg)
    if not account or account.startswith("REPLACE"):
        raise ValueError("Supply the verified CIS262065 Slurm account from projects")
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ACCOUNT") != account:
        raise RuntimeError("Run inside a Slurm allocation charged to the specified project account")
    output = Path(cfg["output_dir"])
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    reference = Path(reference_run).resolve() if reference_run else None
    if reference:
        other = read_json(reference / "config.json")
        if comparable(other) != comparable(cfg):
            raise ValueError("Reference settings differ beyond pooling and paths")

    adapter = load_adapter()
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU: select the allocated compute kernel, not the login node")
    datasets = {}
    load_start = time.monotonic()
    for split, limit in (("train", 5000), ("valid", 500)):
        ds = adapter.PreparedDataset(cfg["data"]["data_dir"], cfg["data"]["categories"],
                                     split, limit, 250, cfg["seed"])
        counts = np.bincount(ds.labels, minlength=345)
        if len(counts) != 345 or not np.all(counts == limit):
            raise ValueError(f"Expected exactly {limit} examples per class in {split}")
        if reference and ds.ids != read_json(reference / f"{split}_sample_ids.json"):
            raise ValueError(f"Exact {split} sample IDs/order differ from reference")
        datasets[split] = ds
        print(f"Loaded {split}: {len(ds):,}; exact reference IDs checked: {bool(reference)}", flush=True)
    if not set(datasets["train"].ids).isdisjoint(datasets["valid"].ids):
        raise ValueError("Train/validation IDs overlap")

    output.mkdir(parents=True, exist_ok=False)
    for split, ds in datasets.items():
        write_json(output / f"{split}_sample_ids.json", ds.ids)
        write_json(output / f"{split}_class_counts.json", np.bincount(ds.labels, minlength=345).tolist())

    def cached_build(directory, categories, train_per_class=None, val_per_class=None,
                     test_per_class=None, max_seq_len=None, stroke_format="stroke3",
                     seed=0, scale_factor=None, splits=("train", "valid", "test")):
        if (Path(directory).resolve() != Path(cfg["data"]["data_dir"]).resolve()
                or list(categories) != cfg["data"]["categories"]
                or train_per_class != 5000 or val_per_class != 500
                or max_seq_len != 250 or stroke_format != "stroke3"
                or seed != 42 or scale_factor not in (None, 1.0)
                or any(s not in datasets for s in splits)):
            raise ValueError("Unexpected dataset request; test is not enabled")
        return {s: datasets[s] for s in splits}

    adapter.training.build_datasets = cached_build
    adapter.evaluation.build_datasets = cached_build
    provenance = {
        "historical_commit": "5fe0e29ee627d1a36f979a52a7659eca2c761878",
        "source_hashes": sources, "runner_sha256": sha256(__file__),
        "slurm_account": account, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "reference_run": str(reference) if reference else None,
        "exact_reference_ids_verified": bool(reference), "test_evaluated": False,
        "data_loading_seconds": time.monotonic() - load_start,
    }
    write_json(output / "provenance.json", provenance)
    started = time.monotonic()
    history = adapter.training.train(cfg)
    elapsed = time.monotonic() - started
    if len(history["val_acc"]) != cfg["train"]["epochs"]:
        raise RuntimeError("Epoch count mismatch")
    if not all(np.isfinite(values).all() for values in history.values()):
        raise RuntimeError("Nonfinite training metrics")
    metrics = adapter.evaluation.evaluate(str(output / "best.pt"), split="valid", device="cuda")
    if abs(metrics["acc"] - max(history["val_acc"])) > 1e-8:
        raise RuntimeError("Best checkpoint validation mismatch")
    write_json(output / "training_runtime.json", {"training_seconds": elapsed})
    print("Completed; test split untouched.", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument("--reference-run", help="Directory with config.json and original train/valid sample ID lists")
    args = parser.parse_args()
    execute(args.config, args.account, args.reference_run)


if __name__ == "__main__":
    main()
