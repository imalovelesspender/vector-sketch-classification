#!/bin/bash
# Login-node watcher: waits for the archive download to finish, re-verifies every
# archive's MD5, then submits the whole chain with SLURM dependencies:
#   preprocess -> preflight gate -> 3 training tasks (one at a time) -> test eval + summary
# Every later job uses afterok and --kill-on-invalid-dep=yes, so any failure
# cancels everything downstream instead of training on unverified data.
#   nohup bash submit_overnight.sh <download_pid> > submit_overnight.log 2>&1 &
set -euo pipefail
M=/ocean/projects/cis260329p/mcui3
PREP=$M/qd-data
CHECKOUT=$M/qd-ablation
DOWNLOAD_PID=${1:?download pid}
MIN_SU=20  # worst case of the whole chain at its time limits is about 17 GPU-hours

while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do sleep 30; done
echo "$(date) download process exited"
grep -q 'STOPPED AFTER DOWNLOAD' "$PREP/download.log"
cd "$PREP/data/sketchrnn"
test "$(ls *.full.npz | wc -l)" -eq 345
test -z "$(ls *.part 2>/dev/null)"
module load pytorch/26.05-2.11-py3 >/dev/null 2>&1
python - <<'PY'
import base64, hashlib, json
from pathlib import Path
inventory = {Path(i['name']).name: i for i in json.loads(Path('download_manifest.json').read_text())}
full = [n for n in inventory if n.endswith('.full.npz')]
assert len(full) == 345
for name in full:
    digest = hashlib.md5()
    with open(name, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    assert Path(name).stat().st_size == inventory[name]['size'], name
    assert base64.b64encode(digest.digest()).decode() == inventory[name]['md5Hash'], name
print('ALL 345 .full.npz ARCHIVES MATCH GOOGLE SIZE AND MD5')
PY
du -sh "$PREP/data/sketchrnn"

# Budget and storage before submitting anything.
my_quotas 2>&1 | grep -A3 'Project directory'
balance=$(projects 2>&1 | grep -m1 -oE '[0-9,]+ / [0-9,]+ SU' | cut -d' ' -f1 | tr -d ,)
echo "GPU SU balance: $balance"
test "$balance" -ge "$MIN_SU"

# Swap in the verified scripts (the old psc_run held the download runner).
cd "$CHECKOUT"
if [ -d psc_run_new ]; then mv psc_run psc_run_superseded && mv psc_run_new psc_run; fi
sha256sum psc_run/*

cd "$PREP"
prep=$(sbatch --parsable --export=ALL,PREP_ROOT=$PREP,PROTOCOL=$PREP/protocol_config.json preprocess.sbatch)
cd "$CHECKOUT"
gate=$(sbatch --parsable --dependency=afterok:$prep --kill-on-invalid-dep=yes \
  --export=ALL,CHECKOUT=$CHECKOUT,PREP_ROOT=$PREP psc_run/preflight_gate.sbatch)
train=$(sbatch --parsable --dependency=afterok:$gate --kill-on-invalid-dep=yes \
  --export=ALL,CHECKOUT=$CHECKOUT,DATA_DIR=$PREP/data/processed_full psc_run/independent345.sbatch)
test_eval=$(sbatch --parsable --dependency=afterok:$train --kill-on-invalid-dep=yes \
  --export=ALL,CHECKOUT=$CHECKOUT,PREP_ROOT=$PREP psc_run/evaluate_test_all.sbatch)
echo "SUBMITTED preprocess=$prep gate=$gate train=$train test_eval=$test_eval"
squeue -u "$USER" -o '%.10i %.9P %.18j %.2t %.10M %.30E %R'
