"""Validate experiment inputs and submit the pooling comparison to Slurm."""
from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import run_pooling as driver

ACCOUNT = "cis260329p"
PROJECT = Path("/ocean/projects") / ACCOUNT
MODULE = "pytorch/26.05-2.11-py3"


def find_data(explicit=None):
    baseline = driver.read_json(driver.ARCHIVE / "baseline_config.json")
    if explicit:
        candidates = [Path(explicit)]
    else:
        personal = PROJECT / getpass.getuser()
        candidates = [personal / "data/processed_full_rebuilt_v2", Path(baseline["data"]["data_dir"])]
        for base in (personal, PROJECT / "shared", PROJECT / "awang34"):
            if not base.is_dir():
                continue
            for pattern in ("processed_full", "data/processed_full", "*/data/processed_full", "*/processed_full"):
                candidates.extend(base.glob(pattern))
    checked = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.add(candidate)
        try:
            config = driver.make_config(candidate, "/unused-output")
            driver.check_manifest(config)
            for category in config["data"]["categories"]:
                for split in ("train", "valid"):
                    path = candidate / category / f"{split}.npz"
                    if not path.is_file() or not os.access(path, os.R_OK) or path.stat().st_size == 0:
                        raise PermissionError(path)
            return candidate
        except (OSError, ValueError, KeyError):
            if explicit:
                raise
    raise FileNotFoundError(
        "No readable processed_full dataset passed provenance verification. "
        "Specify --data-dir /accessible/path/to/processed_full. Checked: "
        + ", ".join(str(p) for p in checked)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--reference-run", type=Path,
                        help="Run mean only against a baseline containing exact sample IDs")
    parser.add_argument("--submit", action="store_true", help="Submit after preflight checks")
    args = parser.parse_args()
    if sys.platform != "linux" or not PROJECT.is_dir() or not shutil.which("sbatch"):
        raise RuntimeError("Execute this launcher on Bridges-2")
    if getpass.getuser() != "slin44":
        raise RuntimeError("This project launcher is configured for slin44")
    projects = subprocess.run(["projects"], text=True, capture_output=True, check=True)
    if ACCOUNT not in projects.stdout:
        raise RuntimeError(f"Project account {ACCOUNT} is absent from projects output")
    driver.verify_sources()
    data = find_data(args.data_dir)
    # Resolve the module before consuming a GPU allocation.
    subprocess.run(["bash", "-lc", "module show " + shlex.quote(MODULE)], check=True)
    jobs = subprocess.run(["squeue", "--noheader", "--user", getpass.getuser(),
                           "--format", "%j %T %i"], text=True, capture_output=True, check=True)
    if any(line.split() and line.split()[0] == "pooling-345" for line in jobs.stdout.splitlines()):
        raise RuntimeError("An active pooling-345 job already exists; inspect squeue before resubmitting")
    output = args.output_root or (PROJECT / getpass.getuser() /
                                  ("pooling_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    output = output.resolve()
    configs = {name: driver.make_config(data, output / name, name) for name in ("last", "mean")}
    reference = args.reference_run.resolve() if args.reference_run else output / "last"
    if args.reference_run:
        for name in ("config.json", "train_sample_ids.json", "valid_sample_ids.json"):
            if not (reference / name).is_file():
                raise FileNotFoundError(reference / name)
        if driver.comparable(driver.read_json(reference / "config.json")) != driver.comparable(configs["mean"]):
            raise ValueError("Reference settings differ from the proposed mean run")
    for cfg in configs.values():
        driver.check_config(cfg)
    plan = output / "plan"
    if output.exists():
        raise FileExistsError(f"Use a fresh output root: {output}")
    plan.mkdir(parents=True, exist_ok=False)
    for name, cfg in configs.items():
        driver.write_json(plan / f"{name}_config.json", cfg)
    runner = driver.HERE / "run_pooling.py"
    commands = []
    if not args.reference_run:
        commands.append(shlex.join(["python", "-u", str(runner), "--config",
                                    str(plan / "last_config.json"), "--account", ACCOUNT]))
    commands.append(shlex.join(["python", "-u", str(runner), "--config", str(plan / "mean_config.json"),
                                "--account", ACCOUNT, "--reference-run", str(reference)]))
    script = "\n".join([
        "#!/bin/bash", "#SBATCH --job-name=pooling-345", "#SBATCH --partition=GPU-shared",
        "#SBATCH --gpus=v100-32:1", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=5", "#SBATCH --mem=64G", "#SBATCH --time=04:00:00",
        "set -euo pipefail", "cd " + shlex.quote(str(driver.HERE.parents[1])),
        "module load " + shlex.quote(MODULE),
        "export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1",
        'echo "Job ${SLURM_JOB_ID}; account ${SLURM_JOB_ACCOUNT}; host $(hostname)"',
        "nvidia-smi", *commands, ""])
    script_path = plan / "train_pooling.sbatch"
    script_path.write_text(script, encoding="utf-8", newline="\n")
    common = ["--account", ACCOUNT, "--output", str(plan / "pooling-%j.log"), str(script_path)]
    subprocess.run(["sbatch", "--test-only", *common], check=True)
    print("Preflight passed", flush=True)
    print("Data:", data, flush=True)
    print("Account:", ACCOUNT, "| Outputs:", output, flush=True)
    if not args.submit:
        print("Not submitted. Submission command:", shlex.join(["sbatch", "--parsable", *common]))
        return
    submitted = subprocess.run(["sbatch", "--parsable", *common], text=True, capture_output=True, check=True)
    job_id = submitted.stdout.strip().split(";")[0]
    driver.write_json(plan / "submission.json", {"job_id": job_id, "account": ACCOUNT,
                                                "output_root": str(output), "data_dir": str(data)})
    print("Submitted job:", job_id, flush=True)
    print("Log:", plan / f"pooling-{job_id}.log", flush=True)
    subprocess.run(["squeue", "-j", job_id], check=False)


if __name__ == "__main__":
    main()
