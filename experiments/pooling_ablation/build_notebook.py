"""Build the readable notebook and copy exact historical sources into isolation."""
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMIT = "5fe0e29ee627d1a36f979a52a7659eca2c761878"
archive = ROOT / "docs/experiments/lstm_345class_20261014"
expected = json.loads((archive / "historical_source_hashes.json").read_text())
for relative, digest in expected.items():
    if relative == "psc_run/run_processed.py":
        # Read the Git blob so checkout CRLF conversion cannot change the digest.
        source = "fdea08baf749dff88d4d33a439dc5d277b992800:docs/experiments/lstm_345class_20261014/run_processed.py"
    else:
        source = f"{COMMIT}:{relative}"
    data = subprocess.check_output(["git", "show", source], cwd=ROOT)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Source hash differs: {relative}")
    target = HERE / "historical" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
reference = HERE / "reference"
reference.mkdir(exist_ok=True)
for name, original in {"historical_source_hashes.json": "historical_source_hashes.json",
                       "baseline_config.json": "config.json",
                       "hidden128_config.json": "hidden128_config.json",
                       "hidden512_config.json": "hidden512_config.json"}.items():
    (reference / name).write_bytes((archive / original).read_bytes())

from notebook_content import cells

for index, cell in enumerate(cells):
    cell["id"] = f"pooling-{index:02d}"
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Pooling ablation (PyTorch 2.11)",
            "language": "python", "name": "pooling_ablation_project"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
target = ROOT / "notebooks/pooling_ablation.ipynb"
target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {target}; {len(cells)} cells; historical source hashes verified")
