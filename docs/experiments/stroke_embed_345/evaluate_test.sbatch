#!/bin/bash
#SBATCH --job-name=test345
#SBATCH --account=bjwy-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.log
# Usage:  sbatch --export=ALL,RUN=results/stroke_embed_345class_20261123_12345 delta_run/evaluate_test.sbatch
# Run ONCE per model, only after settings are frozen; selection is by validation.
set -euo pipefail
cd /projects/bjwy/sjin5/code
module load python
source /projects/bjwy/sjin5/env/bin/activate
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
: "${RUN:?set RUN=results/<run dir>}"
DATA=${DATA:-/projects/bjwy/sjin5/qd-data/data/processed_full}
sha256sum "$RUN/best.pt"
cd delta_run && python -u -c "
import run_processed as a
a.evaluation.build_datasets = a.build_prepared
a.evaluation.evaluate('../$RUN/best.pt', split='test', data_dir='$DATA', device='cuda')
"
cd .. && sha256sum "$RUN/best.pt"
