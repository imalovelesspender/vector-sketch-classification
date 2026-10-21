#!/bin/bash
# Test-split evaluation of the three finished runs, same procedure as Anson's
# evaluate_test.sbatch (best-on-validation checkpoint, hash checked before and after).
# Runs only after all three training tasks succeeded, then writes the comparison.
#   sbatch --dependency=afterok:<training array> --export=ALL,CHECKOUT=...,PREP_ROOT=... psc_run/evaluate_test_all.sbatch
#SBATCH --job-name=qd-direction-test
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:30:00
#SBATCH --output=psc_run/direction-test-%j.log
set -euo pipefail
cd "${CHECKOUT:?set CHECKOUT}"
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
for run in results/lstm_345_bi256 results/lstm_345_uni512 results/lstm_345_uni256; do
  test ! -e "$run/metrics_test.json"
  sha256sum "$run/best.pt"
  python -u -c 'import sys; from psc_run import run_processed as a; a.evaluation.build_datasets=a.build_prepared; a.evaluation.evaluate(sys.argv[1] + "/best.pt", split="test", device="cuda")' "$run"
  sha256sum "$run/best.pt"
done
python -u psc_run/summarize_direction345.py \
  --verification "${PREP_ROOT:?set PREP_ROOT}/processed_full_verification.json" \
  --output results/lstm_345_directionality_summary
