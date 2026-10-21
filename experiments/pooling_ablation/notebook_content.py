"""Cell sources for the pooling ablation experiment notebook."""
import textwrap

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": textwrap.dedent(source).strip() + "\n"})


md('''
# Pooling ablation

**Vector Sketch Classification**

This experiment evaluates how sequence pooling affects classification of QuickDraw
sketches. A bidirectional LSTM with last-state pooling is compared with an otherwise
identical model using mean pooling. The primary metric is validation accuracy;
validation cross-entropy and training time provide additional comparisons.

The experiment reuses the historical LSTM implementation and processed-data adapter
from the archived 345-class baseline.

## 1. Experimental protocol

| Setting | Value |
|---|---|
| Categories | 345, in preprocessing-manifest order |
| Training / validation examples | 5,000 / 500 per class |
| Input | Normalized stroke-3 sequences, original dataset splits |
| Length filter | Exclude sequences longer than 250 points |
| Encoder | One-layer bidirectional LSTM, hidden size 256 per direction |
| Sketch embedding | 512 dimensions |
| Pooling | Last state or mean over valid timesteps |
| Dropout | 0 |
| Objective / optimizer | Cross-entropy / Adam |
| Learning rate / weight decay | 0.001 / 0 |
| Gradient clipping | 1.0 |
| Batch size / seed / epochs | 128 / 42 / 10 |
| Model selection | Highest validation accuracy |

Last-state pooling concatenates the final hidden state from each direction.
Mean pooling averages bidirectional outputs over valid timesteps, excluding padding.
Both produce the same embedding width and parameter count.

The training subset remains fixed across epochs. The mean run must match the ordered
training and validation sample-ID lists of its reference run. The test split is excluded
from model development and evaluated separately after both checkpoints are selected.
''')
code('''
from pathlib import Path
import getpass
import importlib.util
import json
import os
import shlex
import subprocess
import sys

ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "experiments/pooling_ablation/run_pooling.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Repository root not found")
EXPERIMENT = ROOT / "experiments/pooling_ablation"
spec = importlib.util.spec_from_file_location("pooling_driver", EXPERIMENT / "run_pooling.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
print("Repository:", ROOT)
print("Python:", sys.executable)
''')
md('''
## 2. Configuration

`pair` runs the last-state control followed by the mean-pooling ablation.
`mean_only` uses an existing baseline with its resolved configuration and ordered sample-ID
lists. Changing the epoch budget requires a reference with the same budget.

`DATA_DIR` identifies the prepared `processed_full` directory. Run outputs are written
under a new `OUTPUT_ROOT`. The Slurm account is explicitly set to the project account.
''')
code('''
ACCOUNT = "cis260329p"
DATA_DIR = Path("/ocean/projects/cis260329p/slin44/data/processed_full_rebuilt_v2")
OUTPUT_ROOT = Path("/ocean/projects") / ACCOUNT / getpass.getuser() / "pooling_ablation_seed42"
RUN_MODE = "pair"  # pair | mean_only
REFERENCE_RUN = None  # Existing baseline directory for mean_only
EPOCHS = 10
MODULE = "pytorch/26.05-2.11-py3"
EXECUTION_MODE = "interactive"  # interactive uses the notebook's GPU allocation; batch submits a job
''')
md('''
## 3. Data and source verification

Historical source files are checked against archived SHA-256 hashes. A private data
rebuild must additionally pass its preparation audit: official archive checksums,
training-fitted normalization, complete split counts and saved-shard readback. Its
manifest hash is recorded independently from the historical experiment. The preprocessing includes per-sketch geometric scaling and
training-fitted numerical normalization; the adapter applies no additional normalization.
Exact sample IDs, class counts and split separation are checked by the training runner
before optimization begins.
''')
code('''
if RUN_MODE not in ("pair", "mean_only"):
    raise ValueError("Expected pair or mean_only")
if RUN_MODE == "mean_only" and REFERENCE_RUN is None:
    raise ValueError("mean_only requires a reference baseline")
DATA_DIR = DATA_DIR.resolve()
OUTPUT_ROOT = OUTPUT_ROOT.resolve()
driver.verify_sources()
if not (DATA_DIR / "preparation_audit.json").is_file():
    raise RuntimeError("Private dataset preparation is incomplete; inspect the preparation log before training")
last_cfg = driver.make_config(DATA_DIR, OUTPUT_ROOT / "last", "last", EPOCHS)
mean_cfg = driver.make_config(DATA_DIR, OUTPUT_ROOT / "mean", "mean", EPOCHS)
for cfg in (last_cfg, mean_cfg):
    driver.check_config(cfg)
    driver.check_manifest(cfg)
    if Path(cfg["output_dir"]).exists():
        raise FileExistsError(f"Run directory already exists: {cfg['output_dir']}")
assert driver.comparable(last_cfg) == driver.comparable(mean_cfg)
print("Source and data provenance verified")
print("Training examples: 1,725,000 | Validation examples: 172,500")
print(json.dumps({"model": mean_cfg["model"], "train": mean_cfg["train"]}, indent=2))
''')
md('''
## 4. Execution plan

Resolved configurations and a Slurm script are saved before execution. In interactive
mode, training uses the notebook kernel's existing GPU allocation. Batch mode submits
a separate job. The paired experiment runs sequentially on one V100 GPU. Each run uses a fresh Python process
and the isolated historical source tree. The second run verifies exact sample identity
against the first; in `mean_only` mode it verifies against the specified baseline.

The resource request allows up to four hours. Actual training duration is measured
separately from queue wait and initial data loading.
''')
code('''
PLAN = OUTPUT_ROOT / "plan"
if RUN_MODE == "pair":
    reference_run = OUTPUT_ROOT / "last"
else:
    reference_run = Path(REFERENCE_RUN).resolve()
    for name in ("config.json", "train_sample_ids.json", "valid_sample_ids.json"):
        if not (reference_run / name).is_file():
            raise FileNotFoundError(reference_run / name)
    if driver.comparable(driver.read_json(reference_run / "config.json")) != driver.comparable(mean_cfg):
        raise ValueError("Reference configuration differs beyond pooling and paths")
PLAN.mkdir(parents=True, exist_ok=False)
last_path, mean_path = PLAN / "last_config.json", PLAN / "mean_config.json"
driver.write_json(last_path, last_cfg)
driver.write_json(mean_path, mean_cfg)
runner = EXPERIMENT / "run_pooling.py"
commands = []
if RUN_MODE == "pair":
    commands.append(shlex.join(["python", "-u", str(runner), "--config", str(last_path), "--account", ACCOUNT]))
commands.append(shlex.join(["python", "-u", str(runner), "--config", str(mean_path),
                            "--account", ACCOUNT, "--reference-run", str(reference_run)]))
script = "\\n".join([
    "#!/bin/bash", "#SBATCH --job-name=pooling-345", "#SBATCH --partition=GPU-shared",
    "#SBATCH --gpus=v100-32:1", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
    "#SBATCH --cpus-per-task=5", "#SBATCH --mem=64G", "#SBATCH --time=04:00:00",
    "set -euo pipefail", "cd " + shlex.quote(str(ROOT)), "module load " + shlex.quote(MODULE),
    "export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1",
    'echo "Job ${SLURM_JOB_ID}; account ${SLURM_JOB_ACCOUNT}; host $(hostname)"',
    "nvidia-smi", *commands, ""])
SCRIPT = PLAN / "train_pooling.sbatch"
SCRIPT.write_text(script, encoding="utf-8", newline="\\n")
submit_args = ["sbatch", "--parsable", "--account", ACCOUNT,
               "--output", str(PLAN / "pooling-%j.log"), str(SCRIPT)]
print(script)
print("Submission command:", shlex.join(submit_args))
''')
code('''
RUN_TRAINING = False
SUBMIT_JOB = False
if EXECUTION_MODE == "interactive" and RUN_TRAINING:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_ACCOUNT") != ACCOUNT:
        raise RuntimeError("Select a notebook server inside the cis260329p compute allocation")
    for command in commands:
        args = shlex.split(command)
        args[0] = sys.executable
        subprocess.run(args, check=True, cwd=ROOT)
elif EXECUTION_MODE == "batch" and SUBMIT_JOB:
    submitted = subprocess.run(submit_args, text=True, capture_output=True, check=True)
    JOB_ID = submitted.stdout.strip().split(";")[0]
    driver.write_json(PLAN / "submission.json", {"job_id": JOB_ID, "account": ACCOUNT,
                                               "command": submit_args})
    print("Job ID:", JOB_ID)
    print("Status:", f"squeue -j {JOB_ID}")
    print("Log:", str(PLAN / f"pooling-{JOB_ID}.log"))
else:
    print("Training disabled; enable RUN_TRAINING for interactive mode or SUBMIT_JOB for batch mode")
''')
code('''
log_path = OUTPUT_ROOT / "plan" / "interactive.log"
status_path = OUTPUT_ROOT / "plan" / "execution_status.json"
if status_path.is_file():
    print(status_path.read_text())
if log_path.is_file():
    print("\\n".join(log_path.read_text(errors="replace").splitlines()[-20:]))
else:
    print("No background execution log")
''')
md('''
## 5. Validation results

Each run saves its configuration, sample IDs, class counts, learning curves,
checkpoints, validation metrics, source and job provenance, and training duration.
The comparison uses each run's validation-selected checkpoint.

The cells below read completed run artifacts. Missing artifacts indicate that a run
has not completed. Recorded final metrics are included at the end of this notebook.
''')
code('''
rows = []
for name in ("last", "mean"):
    folder = OUTPUT_ROOT / name
    if not (folder / "metrics_valid.json").is_file():
        print(name, "not completed")
        continue
    metric = driver.read_json(folder / "metrics_valid.json")
    runtime_path = folder / "training_runtime.json"
    rows.append({"pooling": name, "validation_accuracy_percent": 100 * metric["acc"],
                 "validation_loss": metric["loss"], "best_epoch": metric["epoch"],
                 "training_minutes": driver.read_json(runtime_path)["training_seconds"] / 60
                 if runtime_path.is_file() else None})
print(json.dumps(rows, indent=2))
if len(rows) == 2:
    delta = rows[1]["validation_accuracy_percent"] - rows[0]["validation_accuracy_percent"]
    print(f"Mean minus last: {delta:+.4f} percentage points")
elif RUN_MODE == "mean_only" and rows:
    reference_metrics = driver.read_json(Path(REFERENCE_RUN) / "metrics_valid.json")
    delta = rows[0]["validation_accuracy_percent"] - 100 * reference_metrics["acc"]
    print(f"Mean minus reference last: {delta:+.4f} percentage points")
''')
code('''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name in ("last", "mean"):
    path = OUTPUT_ROOT / name / "history.json"
    if not path.is_file():
        continue
    history = driver.read_json(path)
    epochs = range(1, len(history["val_acc"]) + 1)
    axes[0].plot(epochs, history["train_loss"], "--", label=f"{name} train")
    axes[0].plot(epochs, history["val_loss"], label=f"{name} valid")
    axes[1].plot(epochs, [100 * x for x in history["train_acc"]], "--", label=f"{name} train")
    axes[1].plot(epochs, [100 * x for x in history["val_acc"]], label=f"{name} valid")
for axis, ylabel in zip(axes, ("Cross-entropy loss", "Accuracy (%)")):
    axis.set(xlabel="Epoch", ylabel=ylabel)
    if axis.lines:
        axis.legend()
    axis.grid(alpha=0.25)
fig.tight_layout()
plt.show()
''')
md('''
## 6. Historical reference and interpretation

| Hidden size per direction | Pooling | Epochs completed | Best validation accuracy | Best epoch |
|---|---|---|---|---|
| 128 | Last | 10 | 72.02% | 10 |
| 256 | Last | 10 | 75.19% | 10 |
| 512 | Last | 10 | 76.26% | 5 |

These are archived results, not outputs of the current experiment. Only the historical
hidden-256 baseline has a reported test accuracy (75.40%). A historical comparison
requires matching source, preprocessing, training settings and ordered sample IDs.

The pooling comparison measures performance at a fixed training budget and one seed.
It does not establish convergence or statistical significance. Validation performance,
overfitting trends and training duration should be reported together.

### References

- [Historical LSTM pipeline, commit 5fe0e29](https://github.com/imalovelesspender/vector-sketch-classification/tree/5fe0e29ee627d1a36f979a52a7659eca2c761878)
- [Archived baseline and ablation artifacts](https://github.com/imalovelesspender/vector-sketch-classification/tree/fdea08baf749dff88d4d33a439dc5d277b992800/docs/experiments/lstm_345class_20261014)
''')
md('''
## 7. Final test evaluation

After both runs finish, their validation-selected checkpoints are frozen before test
evaluation. Both use the same 500 test sketches per class (172,500 total), with the
same length filter and seed. Test IDs, predictions, top-1/top-5 accuracy, loss and
checkpoint provenance are saved in each run directory. Existing completed test
evaluations are reused only when the checkpoint and sample-ID hashes match.
''')
code('''
RUN_TEST = False
if RUN_TEST:
    test_command = [sys.executable, "-u", str(EXPERIMENT / "evaluate_pair.py"),
                    "--output-root", str(OUTPUT_ROOT), "--account", ACCOUNT]
    test_log = OUTPUT_ROOT / "plan" / "test_evaluation.log"
    with test_log.open("a") as stream:
        with subprocess.Popen(test_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, cwd=ROOT, bufsize=1) as evaluation:
            for line in evaluation.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if evaluation.wait():
                raise RuntimeError(f"Test evaluation failed; inspect {test_log}")
for name in ("last", "mean"):
    path = OUTPUT_ROOT / name / "metrics_test.json"
    print(name, driver.read_json(path) if path.is_file() else "test not evaluated")
''')

md('## 8. Recorded results\n\nBoth models completed 10 epochs. Checkpoints were selected on validation before final testing on the same 172,500 sketches.\n\n| Pooling | Best validation | Test top-1 | Test top-5 |\n|---|---:|---:|---:|\n| last | 75.1930% | 75.4017% | 93.7704% |\n| mean | 74.3101% | 74.2255% | 93.2638% |\n\nMean pooling reduced test accuracy by 1.1762 percentage points at this fixed budget and seed. This result does not establish performance across seeds or after convergence.\n\n[Configurations, metrics, provenance and learning curves](../docs/experiments/pooling_ablation_seed42/README.md). Full run artifacts remain under results/pooling_ablation_seed42/ and in project storage.\n')
