#!/bin/bash
# Submit the seed / parameter-matched chain (run on a login node from the checkout root,
# only after approval):  bash psc_run/submit_seeds.sh
#   gate (afterok) -> 9-task array, 3 at a time (afterany) -> summary
set -euo pipefail
M=/ocean/projects/cis260329p/mcui3
CHECKOUT=$PWD
E="--export=ALL,CHECKOUT=$CHECKOUT,DATA_DIR=$M/qd-data/data/processed_full,VERIFICATION=$M/qd-data/processed_full_verification.json"
test -z "$(squeue -u "$USER" -h -n qd-seeds,qd-seeds-gate,qd-seeds-summary)"   # never submit twice
projects_out=$(projects 2>&1 < /dev/null)
balance=$(awk 'match($0, /[0-9,]+ \/ [0-9,]+ SU/) {s = substr($0, RSTART, RLENGTH); sub(/ .*/, "", s); gsub(/,/, "", s); print s; exit}' <<< "$projects_out")
echo "GPU SU balance: $balance"
test "$balance" -ge 40   # conservative ceiling of this chain is about 34 SU
gate=$(sbatch --parsable $E psc_run/seeds_gate.sbatch)
array=$(sbatch --parsable --dependency=afterok:$gate --kill-on-invalid-dep=yes $E psc_run/seeds345.sbatch)
summary=$(sbatch --parsable --dependency=afterany:$array $E psc_run/seeds_summary.sbatch)
printf 'gate=%s\nseeds_array=%s\nsummary=%s\n' "$gate" "$array" "$summary" | tee psc_run/seeds_chain_jobs.txt
squeue -u "$USER" -o '%.14i %.18j %.10T %.12l %.30E %R'
