#!/bin/bash
# Seed replicates (43, 44) of BiLSTM-256x2 / UniLSTM-512 / UniLSTM-256 and the
# parameter-matched UniLSTM-378 (seeds 42-44); one task per row of seed_runs.tsv.
# At most three tasks run at once; results do not depend on the order.
#SBATCH --job-name=qd-seeds
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:30:00
#SBATCH --array=0-8%3
#SBATCH --output=psc_run/seeds-%A_%a.log
set -euo pipefail
cd "${CHECKOUT:?set CHECKOUT}"
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
read -r task output_dir bidirectional hidden seed < <(awk -F'\t' -v t="$SLURM_ARRAY_TASK_ID" '$1 == t' psc_run/seed_runs.tsv)
test "$task" = "$SLURM_ARRAY_TASK_ID"
python -u psc_run/run_seeded345.py --data-dir "${DATA_DIR:?set DATA_DIR}" \
  --reference-dir results/lstm_345_bi256 --reference-test-ids results/seed_reference/test_sample_ids.json \
  --bidirectional "$bidirectional" --hidden-dim "$hidden" --train-seed "$seed" --output-dir "$output_dir"
