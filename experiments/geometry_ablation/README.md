# Anson: independent geometry-input and architecture ablations

## Scope grounded in team documents

The 11/17 meeting table assigns Anson the middle `stroke` column: LSTM and
`Sihui's earlier work / Anson` for Transformer. The linked slides describe
ordinary stroke data using `(x, y, dx, dy, pen)`, and separately describe
SketchXAI's learned shape + location + order embedding.

Sources:
- Meeting notes: https://docs.google.com/document/d/1qWnb6IwPR7QjUsBF1OUFmjx2rzsBwJz0UVpeukSrVlQ/edit
- Linked team slides: https://docs.google.com/presentation/d/1WVXzSYvc13aahKEbl267s565aUNMbrMkI4x3YROwxeo/edit
- Sihui reference: `transformer-baseline`, commit `9ea9c26603870c364609dd077cdc5a35a5fb23b3`.

These are independently written models and experiment code. They do not import
Sohyun's stroke splitter/encoder or copy the XAI shape/location/order implementation.
The historical team `src/` files are left untouched.

## Representation: important terminology

`point3`: `(dx, dy, pen_up)`.

`geometry5`: `(dx, dy, pen_up, centered_x, centered_y)`. This carries the same
five types of information as the slide's `(x, y, dx, dy, pen)` in a convenient
column order. Reconstruct XY with cumulative deltas multiplied by the stored
training-fitted coordinate scale, then center on the sketch's mean point.

**Every model token is still a point.** Stroke boundaries are retained in pen_up.
This evaluates richer geometry features. It does not claim that whole strokes
have become single tokens, and it is not an XAI hierarchy. A true whole-stroke
token experiment requires a separate agreed definition. No resampling was added.

## Models and controls

- BiLSTM, hidden 256 per direction, 1 layer, final forward/backward hidden states,
  512-dimensional sketch representation. No dropout.
- Transformer, width 128, 4 layers, 4 attention heads, FF width 384, sinusoidal point
  positions, CLS pooling, pre-norm, GELU. Dropout 0.1; attention-weight dropout 0.
- Geometry-5 Transformer has 707,161 parameters with a 345-class head, matching the
  model size recorded in the 11/7 notes. This is not proof of recovered weights,
  identical sample IDs or reproduced accuracy.

The 14-config matrix includes both point3 controls, geometry5 base models, LSTM
depth 2/3 and width 128/512, and Transformer depth 2/6, width 64/256 and heads 2/8.
Within each family, each variation changes one named factor relative to its base.
Transformer FF width stays 3 times model width. LSTM width is per direction.
Depth and width change parameter counts, which are reported. All three Transformer
heads settings have identical parameter counts, checked before training.

The first development pass uses a common Adam optimizer, LR 0.001, weight decay 0,
batch 128, clip 1, seed 42, float32, and 5 epochs. The common short-run protocol is
not Sihui's original 20-epoch AdamW/warmup/cosine training recipe. Do not attribute
differences versus historical numbers to model structure alone.

## Development data and results

The local diagnostic data are Google's public **standard .npz** archives, five
classes: airplane, apple, bicycle, cat, dog. 500 train and 100 validation sketches
per category; same IDs across all 14 models. No test arrays are accessed.
Invalid, empty/degenerate and >250-point sequences are excluded, not truncated.
Scale per sketch by longest bounding-box side, then fit a shared coordinate std
on the selected training set only. These preprocessing inputs and sample IDs are
different from the historical full.npz/processed_full experiments.

Cache metadata include archive hashes, normalization and IDs. Results save the
source-code hashes, cache hashes, full matrix, per-epoch history, validation-selected
checkpoints, parameter counts, train+validation time, and peak GPU memory.
Reloaded best-checkpoint validation must reproduce the reported score.

The pilot may identify implementation problems or candidates for larger experiments.
One seed, five classes and five epochs do not establish a universal best model,
convergence, significant improvements or performance on 345 classes. Top-5 is
trivial with five classes and should not be highlighted as a result.

## Commands

Run from this worktree root with a CUDA-enabled PyTorch environment:

```powershell
python -m experiments.geometry_ablation.checks
python -m experiments.geometry_ablation.run --prepare --archive-dir PATH_TO_ARCHIVES --cache NEW_CACHE --output NEW_RESULTS --epochs 5 --train-per-class 500 --valid-per-class 100 --device cuda
```

Existing caches can be reused without `--prepare` and `--archive-dir`. To repeat
training with another initialization seed, use `--seed 43` with the same cache.
The cached sample IDs stay fixed. `--runs NAME ...` selects a subset of configs.

## Formal full-data experiments

The formal protocol rebuilds a 345-class cache from the full archives, with 5,000
train and 500 validation samples per class. It uses Sihui's historical preprocessing
source and the old adapter's seeded selection rule. Full-data normalization is
fitted on all training sketches before filtering. Inputs are stored as memory-mapped
arrays so the complete corpus need not fit in RAM. Raw test arrays are not used.

`pipeline.py` waits for successful cache completion, then runs the 14 configurations
for a common 20 epochs, with the four basic representation controls first. Every
epoch saves recoverable state. `run --resume` rejects changed code/config/data
hashes and resumes the optimizer, model and random states. Final publication is
conditional on all 14 result checks passing. The pipeline's local state and logs
are under the workspace's `outputs/geometry5_pipeline_20261118/` and neighboring logs.

Never use development scores as replacements for midterm test results. The exact
old processed manifest and sample IDs still require independent verification for
a direct historical comparison. New point3 and geometry5 controls share the rebuilt
samples with each other. Completed formal results, when available, are archived at
`docs/experiments/geometry5_345_20261118/` with validation clearly separated from test.
No final test evaluation is scheduled by this pipeline.

The data were rebuilt because PSC has a full migration outage including storage:
https://www.psc.edu/b2-migration-2026/

No shared Google documents, main branch, Sohyun branch, W&B account or remote
GitHub repository is modified by these experiments.
