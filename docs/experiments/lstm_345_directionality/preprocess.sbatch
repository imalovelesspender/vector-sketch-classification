#!/bin/bash
# Regenerate processed_full with Sihui's notebook code, then verify it. CPU work
# only; it runs on GPU-shared because the allocation has no CPU (RM) hours.
# Run the download on the login node first (see README), then from PREP_ROOT:
#   sbatch --export=ALL,PREP_ROOT=...,PROTOCOL=... preprocess.sbatch
#SBATCH --job-name=qd-preprocess
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
#SBATCH --output=preprocess-%j.log
set -euo pipefail
cd "${PREP_ROOT:?set PREP_ROOT to the data-preprocessing checkout}"
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u preprocess_full.py
python -u verify_processed.py --data-dir data/processed_full \
  --protocol-config "${PROTOCOL:?set PROTOCOL to protocol_config.json}" \
  --report processed_full_verification.json
du -sh data/sketchrnn data/processed_full
