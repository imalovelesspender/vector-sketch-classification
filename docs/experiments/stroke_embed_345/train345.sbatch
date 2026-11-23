#!/bin/bash
#SBATCH --job-name=stroke345
#SBATCH --account=bjwy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=results/%x-%j.log
# Usage:  sbatch --export=ALL,CONFIG=configs/stroke_embed.json,TAG=stroke_embed delta_run/train345.sbatch
#         (baseline: CONFIG=configs/lstm_baseline.json TAG=lstm_baseline)
set -euo pipefail
cd /projects/bjwy/sjin5/code
module load python
source /projects/bjwy/sjin5/env/bin/activate
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
CONFIG=${CONFIG:-configs/stroke_embed.json}
TAG=${TAG:-stroke_embed}
DATA=${DATA:-/projects/bjwy/sjin5/qd-data/data/processed_full}
OUT=results/${TAG}_345class_$(date +%Y%m%d)_${SLURM_JOB_ID}
echo "config=$CONFIG tag=$TAG data=$DATA out=$OUT"
git rev-parse HEAD; git status --short | head
nvidia-smi --query-gpu=name,memory.total --format=csv
cd delta_run && python -u run_all345.py --config "$CONFIG" --data-dir "$DATA" --output-dir "../$OUT"   # CONFIG is repo-relative
