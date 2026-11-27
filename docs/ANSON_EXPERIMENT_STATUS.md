# Geometry5 full-data follow-up — November 27, 2026

The short-budget screening is complete; **the Geometry5 assignment is not**.
The follow-up implementation is in `experiments/geometry_followup/README.md`:
five controlled Transformer configurations on the full rebuilt cache, 15 epochs
each; two deeper Geometry5 LSTMs, 20 epochs each, with audited existing full-data
1-layer controls. These new runs are pending completion, not reported results.
Final model selection remains joint work with Mae. See the local worker state
and per-epoch histories for live status; this document is not a live job monitor.

## Completed short-budget screening

**Complete and ready for handoff:** 14/14 configurations on 345 classes, 1,000
train / 100 validation sketches per class, five epochs, seed 42. All results
were checkpoint-validated. No final test or retrieval evaluation was performed.

- [Results and protocol](experiments/geometry5_short_20261124/README.md)
- [Handoff to Mae](experiments/geometry5_short_20261124/HANDOFF_TO_MAE.md)
- [Chinese conclusions](experiments/geometry5_short_20261124/CONCLUSION_ZH.md)

Five candidate inference checkpoints have been packaged separately with class
order and preprocessing metadata. Prediction/embedding identity after packaging
was verified. The inference helper is in `experiments/geometry_short_budget/inference.py`.
Large weights and raw data are not committed.

Geometry5 still uses point tokens with added centered XY; it is not an XAI
stroke hierarchy. Depth/width runs are not parameter-matched. The original
20-epoch task was stopped with checkpoints preserved; its partial numbers are
not mixed into the completed short-budget table.

Final model selection and frontend/backend contract alignment remain joint
work with Mae. The notes below preserve earlier development history.

# Anson experiment status — 2026-11-18

## What this work implements

The team slides distinguish `(dx, dy, pen)` from richer `(x, y, dx, dy, pen)`
input, and separately describe the XAI shape/location/order encoder. This work
independently implements the first two feature sets for BiLSTM and Transformer,
plus one-factor-at-a-time depth/width/heads ablations. It does not use Sohyun's
XAI encoder.

**Terminology:** input tokens remain individual points, including the pen flag.
Adding centered absolute coordinates does not turn a whole stroke into one token.
The experiments should be described as point3 versus geometry5 unless the team
explicitly agrees on another meaning of the ordinary `stroke` column.

## Completed development check

Five classes: airplane, apple, bicycle, cat, dog. 2,500 train / 500 validation
sketches, standard `.npz` archives, seed 42, five epochs, same sample IDs for all
14 configurations, Adam 0.001, batch 128, float32. No test predictions.

Observed best validation top-1:

- BiLSTM: point3 control 56.0%; geometry5 base 57.6%.
- Geometry5 BiLSTM depth 2/3: both 67.2%; width 128: 60.4%; width 512: 56.8%.
- Transformer: point3 control 74.8%; geometry5 base 77.0%.
- Geometry5 Transformer depth 2/6: 74.2% / 71.0%.
- Geometry5 Transformer width 64/256: 71.4% / 71.2%.
- Geometry5 Transformer heads 2/4/8: 74.0% / 77.0% / 73.4%.

These are short, single-seed development measurements, not 345-class results or
final test accuracy. Higher capacity did not consistently help within this short
budget. The experiment does not establish a universal optimal architecture,
statistical significance or convergence. Depth/width affect parameter count.
Transformer heads are varied at fixed width and identical parameter counts.

Detailed logs, metrics, actual versus 345-class-head parameter counts and the
exact executed source snapshots will be archived with the final results.

## Data recovery for full experiments

Located Sihui's official-download instructions on `data-preprocessing` at
`a2153dd2144e29fcf62051beb694a5d1f0127f90`. The notebook's canonical Git bytes hash
to `af55d144d5bee88d56fa4d3e71a76fd525612a51db1a4290f15dd28083cdf11c`, exactly the
preprocessing-source hash recorded in the historical baseline archive.

All 345 `.full.npz` archives (6.45 GiB) have been downloaded and verified against
Google's official MD5 values. The full training-only normalization and compact
train/validation cache are being rebuilt. Selection follows the historical
adapter: sorted categories, independent seed-42 RNG per split, lengths >250
excluded, 5,000 train / 500 validation samples per category. Do not silently mix
standard and full archives.

Original processed-manifest/sample-ID artifacts have not been independently
recovered. Describe new runs as rebuilt controls until input equivalence is
verified, rather than claiming the historical model has been recovered.

## Implementation checks

- Geometry reconstruction and pen flags.
- Padding and mixed-length batch invariance.
- Finite gradients across all 14 model configurations.
- Weight save/reload reproduces outputs and validation accuracy.
- Memmapped versus regular cache reader identity.
- Interrupted/resumed training has identical weights and metrics to continuous
  training in the tested two-epoch configuration.

## Publication scope

Push the independent Anson branch and final reproducible results, not raw data,
credentials or large checkpoint binaries. Keep final 345-class results separate
from the development measurements. Do not merge into main automatically.

## References

- Team notes: https://docs.google.com/document/d/1qWnb6IwPR7QjUsBF1OUFmjx2rzsBwJz0UVpeukSrVlQ/edit
- Team slides: https://docs.google.com/presentation/d/1WVXzSYvc13aahKEbl267s565aUNMbrMkI4x3YROwxeo/edit
- Sihui preprocessing: https://github.com/imalovelesspender/vector-sketch-classification/blob/data-preprocessing/data/README.md
- Sihui model reference: https://github.com/imalovelesspender/vector-sketch-classification/blob/transformer-baseline/notebooks/Transformer%20Stroke%20Level.ipynb
- PyTorch layers: https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
