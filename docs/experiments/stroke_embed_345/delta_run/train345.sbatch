#!/bin/bash
#SBATCH --job-name=stroke345
#SBATCH --account=bjwy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/u/sjin5/vector-sketch-classification/docs/experiments/stroke_embed_345/results/%x-%j.log
# Usage (repo root):
#   sbatch --export=ALL,CONFIG=configs/stroke_embed.json,TAG=stroke_embed docs/experiments/stroke_embed_345/delta_run/train345.sbatch
#   baseline : CONFIG=configs/lstm_baseline.json TAG=lstm_baseline
#   1-epoch  : add EPOCHS=1,TAG=stroke_embed_1ep and --time=00:40:00
set -euo pipefail
EXP_DIR=/u/sjin5/vector-sketch-classification/docs/experiments/stroke_embed_345
cd "$EXP_DIR"
module load python
source /u/sjin5/env/bin/activate
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

CONFIG=${CONFIG:-configs/stroke_embed.json}        # repo-relative
TAG=${TAG:-stroke_embed}
EPOCHS=${EPOCHS:-}                                 # empty -> epochs from config (10)
DATA=${DATA:-/projects/bjwy/shared/processed_full}
OUT=$EXP_DIR/results/${TAG}_345class_$(date +%Y%m%d)_${SLURM_JOB_ID}
mkdir -p "$EXP_DIR/results"

# Weights & Biases live curves (one-time: wandb login). Comment out to disable.
export WANDB_PROJECT=vector-sketch-classification
export WANDB_NAME="${TAG}_${SLURM_JOB_ID}"
export WANDB_RUN_GROUP=stroke_embed_345

echo "config=$CONFIG tag=$TAG epochs=${EPOCHS:-config} data=$DATA out=$OUT job=$SLURM_JOB_ID"
git rev-parse HEAD; git status --short | head
nvidia-smi --query-gpu=name,memory.total --format=csv
cd delta_run && python -u run_all345.py --config "$CONFIG" --data-dir "$DATA" --output-dir "$OUT" ${EPOCHS:+--epochs $EPOCHS}