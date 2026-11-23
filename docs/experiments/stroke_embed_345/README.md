# Running on Delta and archiving like the other experiment folders

One-time setup (already done if `/projects/bjwy/sjin5/{code,env,qd-data}` exist):

```bash
cd /projects/bjwy/sjin5
git clone -b stroke-base/embed https://github.com/imalovelesspender/vector-sketch-classification.git code
module load python && python -m venv env && source env/bin/activate && pip install torch numpy matplotlib
# qd-data/data/processed_full: Sihui's preprocessing (see scripts/preprocess/ or Mae's lstm_345_directionality README)
```

## 1. Train (345 classes, validation only)

```bash
cd /projects/bjwy/sjin5/code
sbatch --export=ALL,CONFIG=configs/stroke_embed.json,TAG=stroke_embed   delta_run/train345.sbatch
sbatch --export=ALL,CONFIG=configs/lstm_baseline.json,TAG=lstm_baseline delta_run/train345.sbatch   # control
squeue -u $USER ; tail -f results/stroke345-<jobid>.log
```

Output: `results/<TAG>_345class_<date>_<jobid>/` with `config.json`, `history.json`, `curves.png`,
`best.pt`, `last.pt`, `metrics_valid.json`, `training_runtime.json`, `{train,valid}_sample_ids.json`,
`{train,valid}_class_counts.json`, `planned_config.json` -- the same set Anson's and Mae's runs produced.

Ablations: copy `configs/stroke_embed.json`, flip one of `use_order` / `use_location` / `use_shape`
to `false`, and submit with that CONFIG and a matching TAG (e.g. `stroke_embed_noorder`).

Smoke test first, on a login node (CPU, 3 classes, 1 epoch, a minute):
```bash
python -m src.train --config configs/stroke_embed.json --epochs 1 --categories cat apple bicycle \
    --device cpu --data-dir /projects/bjwy/sjin5/qd-data/data/processed_full
```

## 2. Test (once, after settings are frozen)

```bash
sbatch --export=ALL,RUN=results/stroke_embed_345class_<date>_<jobid> delta_run/evaluate_test.sbatch
```
Writes `metrics_test.json` into the run directory and prints the checkpoint SHA-256 before and after.

## 3. Archive into docs/experiments/

```bash
python delta_run/archive.py results/stroke_embed_345class_<date>_<jobid> \
    --name stroke_embed_345class_<date> \
    --logs results/stroke345-<jobid>.log results/test345-<jobid2>.log
git add docs/experiments/stroke_embed_345class_<date> && git commit -m "Archive stroke_embed 345-class run" && git push
```

`archive.py` copies the metric/curve/config files, the scripts and the Slurm logs, writes
`source_hashes.json` (SHA-256 of every `src/` file, configs, `delta_run/`, `best.pt`, git commit)
and a `README.md` with the settings and result tables filled in.  Checkpoints and sample-ID lists
are left in `results/` as in the other folders.

## Files

| file | from |
|---|---|
| `run_processed.py` | Anson's adapter (`lstm_345class_20261014`) + `--config` + accepts `data_format`/`normalize` from the current `train.py` + passes stroke tensors in the padding check |
| `run_all345.py` | Anson's 345-class runner + `--config`, `--epochs`, writes `training_runtime.json` |
| `train345.sbatch`, `evaluate_test.sbatch` | Delta headers (`bjwy-delta-gpu`, `gpuA40x4`, 1 GPU) |
| `archive.py` | new |

Dry-run on CPU with synthetic shards in Sihui's npz layout: train -> valid metrics -> test eval -> archive,
for both `configs/stroke_embed.json` and `configs/lstm_baseline.json`.
