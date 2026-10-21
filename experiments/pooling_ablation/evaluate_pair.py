"""Evaluate both validation-selected pooling checkpoints on identical test samples."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_pooling as driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--account", default="cis260329p")
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ACCOUNT") != args.account:
        raise RuntimeError("Use the project compute allocation")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    torch.set_num_threads(1)
    driver.verify_sources()
    configs, selection = {}, {}
    for name in ("last", "mean"):
        folder = root / name
        cfg = driver.read_json(folder / "config.json")
        driver.check_config(cfg)
        driver.check_manifest(cfg)
        valid = driver.read_json(folder / "metrics_valid.json")
        driver.read_json(folder / "training_runtime.json")
        if cfg["model"]["pooling"] != name or valid["split"] != "valid":
            raise ValueError("Unexpected model or checkpoint-selection split")
        configs[name] = cfg
        selection[name] = {"checkpoint_sha256": driver.sha256(folder / "best.pt"),
                           "validation_accuracy": valid["acc"], "selected_epoch": valid["epoch"]}
    if driver.comparable(configs["last"]) != driver.comparable(configs["mean"]):
        raise ValueError("The two training configurations differ beyond pooling")
    frozen = root / "plan/test_evaluation_selection.json"
    if frozen.exists() and driver.read_json(frozen) != selection:
        raise ValueError("Selected checkpoints changed after test evaluation was planned")
    driver.write_json(frozen, selection)
    adapter = driver.load_adapter()
    cfg = configs["last"]
    print("Loading fixed test split: 500 examples per class", flush=True)
    data = adapter.PreparedDataset(cfg["data"]["data_dir"], cfg["data"]["categories"], "test", 500, 250, cfg["seed"])
    counts = np.bincount(data.labels, minlength=345)
    assert len(data) == 172500 and np.all(counts == 500)
    id_set = set(data.ids)
    for name in ("last", "mean"):
        for split in ("train", "valid"):
            assert id_set.isdisjoint(driver.read_json(root / name / f"{split}_sample_ids.json"))
    ids_hash = hashlib.sha256("".join(identifier + "\n" for identifier in data.ids).encode()).hexdigest()
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=0, collate_fn=adapter.collate_sketches)
    for name in ("last", "mean"):
        folder = root / name
        result_path = folder / "metrics_test.json"
        if result_path.exists():
            old = driver.read_json(folder / "test_provenance.json")
            assert old["checkpoint_sha256"] == selection[name]["checkpoint_sha256"] and old["sample_ids_sha256"] == ids_hash
            print("ALREADY EVALUATED", name, driver.read_json(result_path), flush=True)
            continue
        checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        assert checkpoint["config"] == configs[name] and checkpoint["epoch"] == selection[name]["selected_epoch"]
        model = adapter.build_model(configs[name]["model"], 3, 345).cuda().eval()
        model.load_state_dict(checkpoint["model_state"])
        correct = top5_correct = samples = 0
        loss_sum = 0.0
        predictions = []
        started = time.monotonic()
        with torch.inference_mode():
            for step, batch in enumerate(loader, 1):
                labels = batch["labels"].cuda()
                logits = model(batch["strokes"].cuda(), batch["lengths"])
                top = logits.topk(5, dim=1).indices
                correct += int((top[:, 0] == labels).sum())
                top5_correct += int((top == labels[:, None]).any(1).sum())
                loss_sum += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
                samples += len(labels)
                predictions.append(top.cpu().numpy())
                if step % 200 == 0:
                    print(f"TEST {name}: {samples:,}/{len(data):,}", flush=True)
        metrics = {"loss": loss_sum / samples, "acc": correct / samples,
                   "top5_acc": top5_correct / samples, "split": "test", "num_samples": samples,
                   "checkpoint": str(folder / "best.pt"), "epoch": checkpoint["epoch"],
                   "evaluation_seconds": time.monotonic() - started}
        assert samples == 172500
        driver.write_json(folder / "test_sample_ids.json", data.ids)
        driver.write_json(folder / "test_class_counts.json", counts.tolist())
        np.savez_compressed(folder / "test_predictions.npz", labels=np.asarray(data.labels), top5=np.concatenate(predictions))
        driver.write_json(folder / "test_provenance.json", {
            **selection[name], "sample_ids_sha256": ids_hash, "seed": cfg["seed"],
            "manifest_sha256": cfg["data"]["preprocessing"]["manifest_sha256"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"], "account": args.account,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "evaluation_source_sha256": driver.sha256(__file__), "precision": "float32"})
        driver.write_json(result_path, metrics)
        print("TEST COMPLETE", name, metrics, flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()
    print("BOTH TEST EVALUATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

