#!/bin/bash
#SBATCH --job-name=qd-hidden
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --array=0-1%1
#SBATCH --output=psc_run/hidden-%A_%a.log
set -euo pipefail
cd /ocean/projects/cis260329p/awang34/vector-sketch-classification-lstm-pipeline
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
sizes=(128 512)
hidden=${sizes[$SLURM_ARRAY_TASK_ID]}
python -u psc_run/run_hidden345.py \
  --data-dir /ocean/projects/cis260329p/awang34/vector-sketch-classification-data-preprocessing/data/processed_full \
  --baseline-dir results/lstm_processed_345class_20261014 \
  --hidden-dim "$hidden" \
  --output-dir "results/lstm_345_hidden${hidden}_20261018"
