"""Meaningful synthetic CPU checks for the notebook, adapter and training handoff."""
import ast
import copy
import json
from pathlib import Path
import tempfile

import nbformat
import numpy as np
import torch

import run_pooling as driver


def main():
    notebook = nbformat.read(driver.HERE.parents[1] / "notebooks/pooling_ablation.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            if "SUBMIT_JOB =" in cell.source:
                assert "SUBMIT_JOB = False" in cell.source
    driver.verify_sources()
    original = driver.read_json(driver.ARCHIVE / "baseline_config.json")
    config = driver.make_config("unused-synthetic-input", "new-output", "mean")
    driver.check_config(config)
    assert driver.comparable(config) == driver.comparable(original)
    # A 30-epoch comparison must not silently reuse a 10-epoch baseline.
    longer = driver.make_config("unused-synthetic-input", "new-output", "mean", 30)
    driver.check_config(longer)
    assert driver.comparable(longer) != driver.comparable(original)
    mutated = copy.deepcopy(config)
    mutated["data"]["train_per_class"] = 4000
    try:
        driver.check_config(mutated)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed sample count was accepted")

    adapter = driver.load_adapter()
    torch.set_num_threads(1)
    # Keep all temporary test data inside this known writable experiment folder.
    with tempfile.TemporaryDirectory(prefix="pooling-check-", dir=driver.HERE) as temp:
        root = Path(temp)
        categories = ["apple", "cat"]
        manifest = {"representation": "stroke-3", "archive_version": "full",
                    "normalization": {"fit_split": "train"},
                    "class_to_index": dict(zip(categories, range(2)))}
        driver.write_json(root / "manifest.json", manifest)
        rng = np.random.default_rng(7)
        for label, category in enumerate(categories):
            (root / category).mkdir()
            for split in ("train", "valid"):
                sequences = []
                for length in (3, 4, 5, 251, 8, 10):
                    s = rng.normal(size=(length, 3)).astype(np.float32)
                    s[:, 2] = 0
                    s[-1, 2] = 1
                    sequences.append(s)
                offsets = np.cumsum([0] + [len(s) for s in sequences])
                np.savez(root / category / f"{split}.npz", points=np.concatenate(sequences),
                         offsets=offsets, labels=np.full(6, label),
                         ids=np.array([f"{category}/full/{split}/{i}" for i in range(6)]))
        sets = adapter.build_prepared(str(root), categories, 4, 4, max_seq_len=250,
                                      seed=42, splits=("train", "valid"))
        again = adapter.build_prepared(str(root), categories, 4, 4, max_seq_len=250,
                                       seed=42, splits=("train", "valid"))
        assert sets["train"].ids == again["train"].ids
        assert sets["valid"].ids == again["valid"].ids
        assert set(sets["train"].ids).isdisjoint(sets["valid"].ids)
        assert all(len(s) <= 250 for s in sets["train"].sketches)
        assert all(not identifier.endswith("/3") for identifier in sets["train"].ids)
        assert sets["train"].scale_factor == 1.0
        # The adapter must return stored normalized points unchanged.
        identifier = sets["train"].ids[0]
        category, _, split, index = identifier.split("/")
        with np.load(root / category / f"{split}.npz") as shard:
            a, b = shard["offsets"][int(index):int(index)+2]
            np.testing.assert_array_equal(sets["train"].sketches[0].numpy(), shard["points"][a:b])

        batch = adapter.collate_sketches([sets["train"][i] for i in range(4)])
        parameter_counts = []
        for pooling in ("last", "mean"):
            model_cfg = {"encoder": "lstm", "hidden_dim": 4, "num_layers": 1,
                         "bidirectional": True, "dropout": 0.0, "pooling": pooling,
                         "embedding_dim": None}
            model = adapter.build_model(model_cfg, 3, 2).eval()
            parameter_counts.append(sum(p.numel() for p in model.parameters()))
            with torch.no_grad():
                z = model.embed(batch["strokes"], batch["lengths"])
                padded = torch.nn.functional.pad(batch["strokes"], (0, 0, 0, 7))
                assert z.shape == (4, 8)
                torch.testing.assert_close(z, model.embed(padded, batch["lengths"]))
                if pooling == "mean":
                    packed = torch.nn.utils.rnn.pack_padded_sequence(batch["strokes"], batch["lengths"],
                                                                     batch_first=True, enforce_sorted=False)
                    outputs, _ = model.encoder.lstm(packed)
                    unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
                    expected = torch.stack([unpacked[i, :int(n)].mean(0) for i, n in enumerate(batch["lengths"])])
                    torch.testing.assert_close(z, expected)
            cfg = {"seed": 42, "output_dir": str(root / f"output-{pooling}"),
                   "data": {"data_dir": str(root), "categories": categories,
                            "train_per_class": 4, "val_per_class": 4, "max_seq_len": 250,
                            "batch_size": 4, "num_workers": 0}, "model": model_cfg,
                   "train": {"epochs": 2, "lr": 0.001, "grad_clip": 1.0, "device": "cpu"}}
            adapter.training.build_datasets = lambda *args, **kwargs: sets
            adapter.evaluation.build_datasets = lambda *args, **kwargs: {s: sets[s] for s in kwargs["splits"]}
            history = adapter.training.train(cfg)
            metric = adapter.evaluation.evaluate(str(Path(cfg["output_dir"]) / "best.pt"),
                                                  split="valid", device="cpu")
            assert abs(metric["acc"] - max(history["val_acc"])) < 1e-8
        assert parameter_counts[0] == parameter_counts[1]
    print("PASS: notebook schema/cells, source hashes, config controls, exact sampling, length filtering,")
    print("normalization preservation, padding-aware mean/last pooling, matched capacity, training and checkpoint evaluation.")


if __name__ == "__main__":
    main()
