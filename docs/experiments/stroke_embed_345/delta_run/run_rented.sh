#!/bin/bash
# Same experiment as train345.sbatch, on a rented GPU box (RunPod / Lambda / Vast, any CUDA image).
# Usage, from the repo root on the box:
#   bash docs/experiments/stroke_embed_345/delta_run/run_rented.sh setup            # once: pip deps
#   bash .../run_rented.sh train stroke_embed  configs/stroke_embed.json  [EPOCHS]  # resumable: rerun same TAG to continue
#   bash .../run_rented.sh train lstm_baseline configs/lstm_baseline.json
#   bash .../run_rented.sh test  <results folder name>                              # test split, once per model
#   bash .../run_rented.sh pack                                                      # results tarball without checkpoints
# DATA: directory holding manifest.json (default: the uploaded subset).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)           # .../stroke_embed_345/delta_run
EXP_DIR=$(dirname "$HERE")
REPO=$(cd "$EXP_DIR/../../.." && pwd)
DATA=${DATA:-/workspace/processed_subset}
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export WANDB_PROJECT=${WANDB_PROJECT:-vector-sketch-classification}   # export WANDB_PROJECT= to disable
export WANDB_RUN_GROUP=stroke_embed_345
cd "$REPO"

case "${1:-}" in
  setup)
    python -m pip install -q torch numpy matplotlib wandb || pip install -q torch numpy matplotlib wandb
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
    [ -f "$DATA/manifest.json" ] || { echo "no manifest in $DATA (upload processed_subset.tgz and tar -xzf it)"; exit 1; }
    [ -n "${WANDB_PROJECT}" ] && (wandb status >/dev/null 2>&1 || echo "run: wandb login   (or export WANDB_PROJECT= to skip W&B)")
    ;;
  train)
    TAG=${2:?TAG}; CONFIG=${3:?config path, repo-relative}; EPOCHS=${4:-}
    OUT=$EXP_DIR/results/${TAG}_345class_$(date +%Y%m%d)_rented
    mkdir -p "$EXP_DIR/results"
    RESUME=""; [ -f "$OUT/last.pt" ] && RESUME="--resume" && echo "resuming $OUT"
    export WANDB_NAME="${TAG}_rented"
    { echo "config=$CONFIG tag=$TAG epochs=${EPOCHS:-config} data=$DATA out=$OUT"
      git rev-parse HEAD; nvidia-smi --query-gpu=name,memory.total --format=csv; } | tee -a "$EXP_DIR/results/${TAG}_rented.log"
    cd "$HERE" && python -u run_all345.py --config "$CONFIG" --data-dir "$DATA" --output-dir "$OUT" \
        ${EPOCHS:+--epochs $EPOCHS} $RESUME 2>&1 | tee -a "$EXP_DIR/results/${TAG}_rented.log"
    ;;
  test)
    RUN_DIR=$EXP_DIR/results/${2:?results folder name}
    [ -f "$RUN_DIR/best.pt" ] || { echo "no best.pt in $RUN_DIR"; exit 1; }
    sha256sum "$RUN_DIR/best.pt" | tee -a "$RUN_DIR/test.log"
    cd "$HERE" && python -u -c "
import run_processed as a
a.evaluation.build_datasets = a.build_prepared
a.evaluation.evaluate('$RUN_DIR/best.pt', split='test', data_dir='$DATA', device='cuda')
" 2>&1 | tee -a "$RUN_DIR/test.log"
    sha256sum "$RUN_DIR/best.pt" | tee -a "$RUN_DIR/test.log"
    ;;
  pack)
    cd "$EXP_DIR" && tar --exclude='*.pt' --exclude='wandb' -czf "$REPO/stroke_embed_345_results.tgz" results
    ls -la "$REPO/stroke_embed_345_results.tgz"; echo "download this and extract into docs/experiments/stroke_embed_345/ on your PC"
    ;;
  *) sed -n 2,9p "$0"; exit 1;;
esac
