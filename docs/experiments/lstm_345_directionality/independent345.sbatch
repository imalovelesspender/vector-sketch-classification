#!/bin/bash
# Independent directionality ablation on Mae's regenerated data, one task at a time:
# 0 = BiLSTM-256x2 reference (Anson's run_all345.py, unchanged), 1 = UniLSTM-512,
# 2 = UniLSTM-256. Tasks 1-2 check their config and exact sample IDs against task 0.
#   sbatch --export=ALL,CHECKOUT=...,DATA_DIR=... psc_run/independent345.sbatch
#SBATCH --job-name=qd-direction
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=psc_run/independent-%A_%a.log
set -euo pipefail
cd "${CHECKOUT:?set CHECKOUT to the historical checkout root}"
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
: "${DATA_DIR:?set DATA_DIR to processed_full}"
reference=results/lstm_345_bi256
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python -u psc_run/run_all345.py --data-dir "$DATA_DIR" --output-dir "$reference"
else
  # Only start once the reference run finished all 10 epochs.
  python -c "import json,sys; h=json.load(open(sys.argv[1])); assert len(h['val_acc']) == 10" \
    "$reference/history.json"
  sizes=(none 512 256)
  hidden=${sizes[$SLURM_ARRAY_TASK_ID]}
  python -u psc_run/run_direction345.py --data-dir "$DATA_DIR" --baseline-dir "$reference" \
    --hidden-dim "$hidden" --output-dir "results/lstm_345_uni${hidden}"
fi
