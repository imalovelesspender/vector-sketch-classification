#!/bin/bash
# OPTIONAL test-split evaluation of a finished run, same procedure as Anson's
# evaluate_test.sbatch. Only run this if the team decided to report test numbers
# for the Uni runs before looking at their validation results.
#   sbatch --export=ALL,CHECKOUT=...,RUN=results/lstm_345_uni512 psc_run/evaluate_test_direction.sbatch
#SBATCH --job-name=qd-direction-test
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=01:00:00
#SBATCH --output=psc_run/direction-test-%j.log
set -euo pipefail
cd "${CHECKOUT:?set CHECKOUT}"
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum "${RUN:?set RUN}/best.pt"
python -u -c 'import sys; from psc_run import run_processed as a; a.evaluation.build_datasets=a.build_prepared; a.evaluation.evaluate(sys.argv[1] + "/best.pt", split="test", device="cuda")' "$RUN"
sha256sum "$RUN/best.pt"
