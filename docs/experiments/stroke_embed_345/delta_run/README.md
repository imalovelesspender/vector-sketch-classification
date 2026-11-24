# stroke_embed_345

Layout: this folder (`docs/experiments/stroke_embed_345/`) holds `delta_run/` (scripts) and `results/<run>/`
(one folder per Slurm job). Data is shared by the team at `/projects/bjwy/shared/processed_full`
(Sihui's preprocessing, regenerated on Delta 2026-11-23, 22 min on one A40; `sketchrnn/` raw archives next to it).
Submit everything from the repo root (`/u/sjin5/vector-sketch-classification`).

One-time: `module load python && python -m venv /u/sjin5/env && source /u/sjin5/env/bin/activate &&
python -m pip install torch numpy matplotlib wandb` then `wandb login`.

## 1. Train (345 classes, validation only)

```bash
S=docs/experiments/stroke_embed_345/delta_run
sbatch --time=00:40:00 --export=ALL,CONFIG=configs/stroke_embed.json,TAG=stroke_embed_1ep,EPOCHS=1 $S/train345.sbatch   # pipeline check
sbatch --export=ALL,CONFIG=configs/stroke_embed.json,TAG=stroke_embed   $S/train345.sbatch
sbatch --export=ALL,CONFIG=configs/lstm_baseline.json,TAG=lstm_baseline $S/train345.sbatch   # control, same data/split
squeue -u $USER
```

Each job writes `results/<TAG>_345class_<date>_<jobid>/`: `config.json`, `history.json`, `curves.png`,
`metrics_valid.json`, `training_runtime.json`, `{train,valid}_class_counts.json`, `planned_config.json`,
plus `best.pt` / `last.pt` / `*_sample_ids.json` (git-ignored). The Slurm log is `results/stroke345-<jobid>.log`.
Live curves: W&B project `vector-sketch-classification`, run `<TAG>_<jobid>` (comment out the `WANDB_*` exports to disable).

Ablations: copy `configs/stroke_embed.json`, set one of `use_order` / `use_location` / `use_shape` to `false`,
submit with that CONFIG and a matching TAG (e.g. `stroke_embed_noorder`).

Queue tips: short `--time` and small `--mem` get backfilled first; `scontrol update JobId=<id> TimeLimit=.. NumCPUs=..`
shrinks a pending job without losing its place; chain with `--dependency=afterok:<id>`.

## 2. Test (once per run, after settings are frozen)

```bash
sbatch --export=ALL,RUN=stroke_embed_345class_20261123_<jobid> $S/evaluate_test.sbatch
```
Writes `metrics_test.json` into that run folder; log at `results/test345-<jobid>.log`.

## 3. Finalize and commit

```bash
python $S/archive.py                       # README.md, source_hashes.json, .gitignore for this folder
git add docs/experiments/stroke_embed_345 && git commit -m "Archive stroke_embed 345-class runs" && git push
```

## Files

| file | role |
|---|---|
| `run_processed.py` | Anson's adapter + `--config`; finds the repo root by walking up to `src/` + `configs/` |
| `run_all345.py` | Anson's 345-class runner + `--config`, `--epochs`; forces `processed` / `normalize=false`; writes `training_runtime.json` |
| `train345.sbatch` | 1×A40, 4 h; `CONFIG`, `TAG`, `EPOCHS`, `DATA` via `--export`; W&B on |
| `evaluate_test.sbatch` | test split for one `results/<RUN>`; checksums `best.pt` before/after |
| `smoke.sbatch` | 3-class 1-epoch GPU check of both configs, writes to /tmp only |
| `archive.py` | README / hashes / `.gitignore` (checkpoints, sample ids, `wandb/`) |

`src/train.py` logs to W&B only when `WANDB_PROJECT` is set; otherwise it runs unchanged.