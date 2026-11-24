#!/bin/bash
#SBATCH --job-name=test345
#SBATCH --account=bjwy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/u/sjin5/vector-sketch-classification/docs/experiments/stroke_embed_345/results/%x-%j.log
# Usage (repo root):
#   sbatch --export=ALL,RUN=stroke_embed_345class_20261123_24420500 docs/experiments/stroke_embed_345/delta_run/evaluate_test.sbatch
# RUN = folder name under results/. Run ONCE per model, after settings are frozen; selection is by validation.
set -euo pipefail
EXP_DIR=/u/sjin5/vector-sketch-classification/docs/experiments/stroke_embed_345
cd "$EXP_DIR"
module load python
source /u/sjin5/env/bin/activate
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
: "${RUN:?set RUN=<folder name under results/>}"
RUN_DIR=$EXP_DIR/results/$RUN
DATA=${DATA:-/projects/bjwy/shared/processed_full}
[ -f "$RUN_DIR/best.pt" ] || { echo "no best.pt in $RUN_DIR"; exit 1; }
echo "run=$RUN_DIR data=$DATA job=$SLURM_JOB_ID"
sha256sum "$RUN_DIR/best.pt"
cd delta_run && python -u -c "
import run_processed as a
a.evaluation.build_datasets = a.build_prepared
a.evaluation.evaluate('$RUN_DIR/best.pt', split='test', data_dir='$DATA', device='cuda')
"
sha256sum "$RUN_DIR/best.pt"     # unchanged -> the checkpoint was not touched by the test pass