#!/bin/bash
# Gate between preprocessing and training: runs the read-only preflight on a compute
# node. If it fails, the training array (afterok on this job) never starts.
#   sbatch --dependency=afterok:<preprocess> --export=ALL,CHECKOUT=...,PREP_ROOT=... psc_run/preflight_gate.sbatch
#SBATCH --job-name=qd-preflight
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:20:00
#SBATCH --output=psc_run/preflight-%j.log
set -euo pipefail
cd "${CHECKOUT:?set CHECKOUT}"
module load pytorch/26.05-2.11-py3
grep -q 'PROCESSED DATA VERIFIED' "${PREP_ROOT:?set PREP_ROOT}"/preprocess-*.log
python -u psc_run/preflight_independent345.py --data-dir "$PREP_ROOT/data/processed_full" \
  --verification "$PREP_ROOT/processed_full_verification.json"
my_quotas 2>&1 | grep -A2 'Project directory' || true
