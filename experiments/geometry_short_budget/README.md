# November 24: user-approved deadline screening

The user explicitly approved replacing today's delivery with a short-budget
screening: all 345 classes, 1,000 training and 100 validation sketches per class,
14 configurations, 5 epochs per configuration. The earlier 20-epoch task was
stopped and its histories/checkpoints preserved separately.

This is a new experiment with fresh model initialization. No partial long-budget
scores are substituted into its comparison table.

## Data

`prepare.py` selects a fixed, stratified random subset of the existing rebuilt
full-archive cache using seed 42. It preserves the numeric feature values,
normalization, category labels and original source IDs. Saved parent row indices
make the selection auditable. Validation examples never enter training.

Counts: 345,000 train and 34,500 validation. Original full-training normalization
is inherited unchanged. No test arrays or predictions are used.

## Models

Reuse Anson's independent `experiments.geometry_ablation` implementations and
the complete 14-config matrix. Both representations use point tokens: point3
contains dx/dy/pen and geometry5 adds centered XY. It is not whole-stroke
tokenization or Sohyun's XAI encoder.

Every run starts with seed 42 and uses Adam LR 0.001, weight decay 0, batch 128,
gradient clipping 1, float32. BiLSTM dropout is 0; Transformer dropout is 0.1,
attention-weight dropout is 0. Width/depth variations change parameter count.
Transformer heads are varied at fixed width. Feed-forward width stays 3x model width.

## Execution and publication

The independent Windows worker keeps the computer awake on AC while running,
saves live progress, resumes verified checkpoints after up to three unexplained
exits, and stops on a model/data traceback. It does not register an autorun task
or alter persistent power settings. Manual shutdown or reboot still stops it.

After all 14 configurations complete and sample counts, epoch counts and
checkpoint-validation results pass checks, the worker publishes metrics,
protocol, histories, Chinese conclusions and an English team update to
`docs/experiments/geometry5_short_20261124/` on Anson's branch. Raw data,
checkpoint binaries and credentials are not committed. Main is not merged.

The local live state is `outputs/geometry5_short_supervisor_live.json` in the
workspace. The training log is `outputs/geometry5_short_training_20261124.log`.
The original full-budget outputs remain in `outputs/geometry5_full_20261118/`.

## Interpretation

Report results as **five-epoch screening validation accuracy**. One seed and
a small training budget cannot establish convergence, statistical significance
or the best fully trained architecture. Do not compare this table directly with
the midterm test results or combine it with 20-epoch validation numbers.
