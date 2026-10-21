#!/bin/bash
# Build processed_full/ with Sihui's notebook, verify it, and fix the shared sample IDs.
# The 345 .full.npz archives are downloaded beforehand on the login node
# (python run_notebook.py --stop-after-download); here they are only reused.
# Submit from the ablation checkout root:
#   sbatch --export=ALL,DATA_REPO=/ocean/projects/cis260329p/mcui3/qd-data psc_run/prepare_data.sbatch
#SBATCH --job-name=qd-prep
#SBATCH --account=cis260329p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --output=psc_run/prepare-%j.log
set -euo pipefail
CHECKOUT=$PWD
DATA_REPO=${DATA_REPO:?set DATA_REPO to the data-preprocessing checkout}
module load pytorch/26.05-2.11-py3
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
test ! -e "$DATA_REPO/data/processed_full" && test ! -e selection
cd "$DATA_REPO"
python -u "$CHECKOUT/psc_run/run_notebook.py"
python -u "$CHECKOUT/psc_run/verify_processed.py" --data-dir data/processed_full --summary data/verify_summary.json
cd "$CHECKOUT"
python -u psc_run/make_selection.py --data-dir "$DATA_REPO/data/processed_full" --output-dir selection
du -sh "$DATA_REPO/data/sketchrnn" "$DATA_REPO/data/processed_full" selection
