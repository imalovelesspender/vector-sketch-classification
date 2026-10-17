#!/bin/bash
set -euo pipefail
cd /ocean/projects/cis260329p/awang34/vector-sketch-classification-lstm-pipeline
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sha256sum results/lstm_processed_345class_20261014/best.pt
python -u -c 'from psc_run import run_processed as a; a.evaluation.build_datasets=a.build_prepared; a.evaluation.evaluate("results/lstm_processed_345class_20261014/best.pt", split="test", device="cuda")'
sha256sum results/lstm_processed_345class_20261014/best.pt
