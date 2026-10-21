#!/bin/bash
# Gate for the seed / parameter-matched runs: preflight_seeds345.py (writes the
# reference test IDs and re-evaluates the seed-42 checkpoints on them). The
# training array depends on this job succeeding (afterok).
#SBATCH --job-name=qd-seeds-gate
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH --output=psc_run/seeds-gate-%j.log
set -euo pipefail
cd "${CHECKOUT:?set CHECKOUT}"
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -u psc_run/preflight_seeds345.py --data-dir "${DATA_DIR:?set DATA_DIR}" \
  --verification "${VERIFICATION:?set VERIFICATION}" --reference-dir results/lstm_345_bi256 \
  --reference-test-ids results/seed_reference/test_sample_ids.json
