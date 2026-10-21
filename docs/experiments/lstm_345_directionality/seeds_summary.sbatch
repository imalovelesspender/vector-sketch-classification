#!/bin/bash
# Seed-aggregated table and curves; runs after every array task has ended (afterany),
# reports missing runs and exits non-zero if any are missing.
#SBATCH --job-name=qd-seeds-summary
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:15:00
#SBATCH --output=psc_run/seeds-summary-%j.log
set -euo pipefail
cd "${CHECKOUT:?set CHECKOUT}"
module load pytorch/26.05-2.11-py3
python -u psc_run/summarize_seeds345.py --reference-test-ids results/seed_reference/test_sample_ids.json \
  --output results/lstm_345_seeds_summary
