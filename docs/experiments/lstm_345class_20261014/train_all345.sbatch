#!/bin/bash
#SBATCH --job-name=quickdraw-345
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --output=psc_run/train345-%j.log
set -euo pipefail
cd /ocean/projects/cis260329p/awang34/vector-sketch-classification-lstm-pipeline
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u psc_run/run_all345.py \
  --data-dir /ocean/projects/cis260329p/awang34/vector-sketch-classification-data-preprocessing/data/processed_full \
  --output-dir results/lstm_processed_345class_20261014
