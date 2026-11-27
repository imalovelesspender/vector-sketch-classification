# Geometry5 full-data follow-up — November 27

The 14-configuration November 24 study was a five-epoch screening, not completion
of Anson's Geometry5 assignment. This follow-up addresses final model selection.

## Why these runs

- Train a controlled Transformer comparison long enough to go beyond the initial
  screen: same local full-data samples, 15 epochs for every configuration.
- Point3 information versus Geometry5 information with identical total parameter
  counts: zero the two XY columns in the control. It is a capacity-controlled
  three-active-feature input, not the historical three-input projection.
- Independently vary depth 4 -> 6, width 128 -> 256, and heads 4 -> 8.
  Hold FFN width at 384, including the width256 experiment. Depth/width are not
  parameter-matched comparisons; heads are. The unused XY weights in the control
  get no data gradient; total parameter equality is not equal active information.
- Finish full-data 20-epoch Geometry5 BiLSTM depth2/depth3, as the strongest
  short-budget candidates. Reuse the existing completed 20-epoch 1-layer Point3
  and Geometry5 controls only after code, cache, epoch-count and metric audits.

## Shared local data

345 categories, 5,000 train and 500 validation per class; 1,725,000 / 172,500
sketches. Official full archives, training-fitted normalization, maximum 250
points (longer sketches excluded), fixed seed42 selection and persisted IDs.
Features `[dx,dy,pen_up,centered_x,centered_y]` remain POINT tokens, not whole
strokes. No XAI hierarchy or Sohyun model implementation is used.

Exact data identity with teammates' historical artifacts is **not certified**:
same counts and seed are insufficient. New local comparisons share the same
audited cache. Do not rank these scores against Sihui/Mae without ordered-ID,
normalization, precision and training-budget verification.

## Transformer training and reference

Reference: `transformer-baseline@22d4de978546f1603a55f4997cb5f76bfca856a5`,
`notebooks/Transformer 5-features.ipynb`; independently implemented classes.
Pre-norm bidirectional self-attention, CLS pooling, zero-based sinusoidal point
positions, dropout0.1, attention-weight dropout0. AdamW lr0.0003, wd0.01,
batch128, clip1.0, 5% warmup and cosine decay, seed42, bfloat16 autocast when
supported. Fifteen epochs are an equal budget, not a convergence guarantee.

The zero-based positions, optimizer, schedule and BF16 differ from the old
short-budget runner. Old checkpoints are not resumed into the new Transformer
protocol. Checks map identical weights into reference classes extracted from
the pinned notebook and verify exact float32 / supported BF16 outputs.

The LSTM uses the unchanged original runner, Adam0.001 and float32 for20 epochs;
architecture-family comparisons therefore include optimization/budget differences.

## Reproduce

Use Python3.12, NumPy2.3.5 and PyTorch2.11.0+cu128 in the stable local runtime.

```text
python -m experiments.geometry_followup.checks
python -u -m experiments.geometry_followup.run --cache FULL_CACHE --output NEW_RESULT_DIR --epochs 15 --device cuda
python -u -m experiments.geometry_ablation.run --cache FULL_CACHE --output NEW_LSTM_RESULT_DIR --epochs 20 --device cuda --runs lstm_geometry5_depth2 lstm_geometry5_depth3
```

The Windows worker runs both stages, prevents duplicate launches via a held
file lock, and inhibits idle sleep only while plugged in. It resumes completed
epoch checkpoints on relaunch with identical source/data/protocol hashes.
A manual sleep, shutdown or reboot still stops progress; no startup task is added.
Any training, audit or publication failure is saved and stops the worker.

Select by validation, verify a selected checkpoint reload, then archive histories,
protocols and separate Transformer/LSTM result tables to Anson's branch. Do not
commit weights, raw data or credentials. Do not merge main or message teammates.
No test-set predictions are made during tuning; test follows configuration freeze.
