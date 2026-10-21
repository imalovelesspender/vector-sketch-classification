#!/bin/bash
# Directionality ablation, one task at a time:
#   0 = BiLSTM-256x2, 1 = UniLSTM-512, 2 = UniLSTM-256.
# Submit from the ablation checkout root after prepare_data.sbatch and the preflight:
#   sbatch --export=ALL,DATA_DIR=/ocean/projects/cis260329p/mcui3/qd-data/data/processed_full psc_run/direction345.sbatch
#SBATCH --job-name=qd-direction
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --array=0-2%1
#SBATCH --output=psc_run/direction-%A_%a.log
set -euo pipefail
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
directions=(true false false)
sizes=(256 512 256)
names=(bi256 uni512 uni256)
i=$SLURM_ARRAY_TASK_ID
python -u psc_run/run_direction345.py \
  --data-dir "${DATA_DIR:?set DATA_DIR to processed_full}" \
  --selection-dir selection \
  --bidirectional "${directions[$i]}" \
  --hidden-dim "${sizes[$i]}" \
  --output-dir "results/lstm345_dir_${names[$i]}"
