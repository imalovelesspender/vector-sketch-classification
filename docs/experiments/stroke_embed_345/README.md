# stroke_embed_345

Delta, `/projects/bjwy/sjin5/vector-sketch-classification`, branch `stroke-base/embed`, commit `42e80951629802816ed1c444ee7d48ec43ebc034`.
Data: Sihui's `processed_full` (345 classes, 5,000 / 500 / 500 per class, >250 points excluded, seed 42).
Training: Adam lr 0.001, clip 1.0, batch 128; checkpoint by best validation top-1; test evaluated once.

## Runs

| run | representation | encoder | top LSTM (hidden / bi / pooling) | token (dim / stroke_hidden / max_strokes) | ablation flags | epochs |
|---|---|---|---|---|---|---|
| `lstm_baseline_345class_20261124_rented` | `point_based` | `lstm` | 256 / bi=True / last | - / - / - | shape=-, loc=-, order=- | 10 |
| `lstm_point_centred_345class_20261124_rented` | `point_based` | `lstm` | 256 / bi=True / last | - / - / - | shape=-, loc=-, order=- | 10 |
| `stroke_embed_345class_20261124_rented` | `stroke_based` | `stroke_embed` | 256 / bi=True / last | 128 / 64 / 250 | shape=True, loc=True, order=True | 10 |

## Results

| run | best epoch | valid top-1 | valid loss | test top-1 | final train acc | train min |
|---|---|---|---|---|---|---|
| `lstm_baseline_345class_20261124_rented` | 10 | 0.7535 | 0.9247 | 0.7533 | 0.7974 | 67.3 |
| `lstm_point_centred_345class_20261124_rented` | 10 | 0.7610 | 0.9035 | 0.7612 | 0.8066 | 74.4 |
| `stroke_embed_345class_20261124_rented` | 10 | 0.7211 | 1.0830 | 0.7201 | 0.7726 | 86.0 |

Context: BiLSTM-256 point_based reference (Anson, `lstm_345class_20261014`) 75.19% valid / 75.40% test, 711,513 params.

## Files

`results/<run>/`: config.json, planned_config.json, history.json, curves.png, metrics_valid.json,
metrics_test.json, training_runtime.json, {train,valid}_class_counts.json.
Checkpoints and sample-ID lists stay on Delta (git-ignored). Slurm logs `train-*.log`, `test-*.log`.
`source_hashes.json`: SHA-256 of src/, configs/, delta_run/, each best.pt, and the git commit.
Scripts copied here: run_processed.py, run_all345.py, train345.sbatch, evaluate_test.sbatch, smoke.sbatch, archive.py, README.md.
