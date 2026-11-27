# Geometry5 handoff to Mae — November 27

## Completed scope

The Geometry5 exploratory screening is complete: 14 configurations, 345 classes,
1,000 train and 100 validation sketches per class, five epochs, seed 42. All runs
used identical selected source IDs, shared normalization and the same optimizer
budget. Histories, configurations, parameters and timing are in this folder.
All validation-selected checkpoints were reloaded and their scores verified.

The input is **point tokens with five features**: dx, dy, pen_up, centered_x,
centered_y. This is not one token per whole stroke and is not the XAI
shape/location/order encoder.

## Findings to carry into the report

- Input features: LSTM 63.27% -> 65.20%; Transformer 58.23% -> 64.11% when adding XY.
- BiLSTM depth 1/2/3: 65.20% / 69.10% / 69.84%.
- Transformer depth 2/4/6: 60.08% / 64.11% / 66.23%.
- Transformer width 64/128/256: 59.83% / 64.11% / 60.12%.
- Transformer heads 2/4/8: 64.27% / 64.11% / 64.41%; only a small observed difference.

Depth/width variations change parameter count, so these are configuration versus
accuracy/cost comparisons, not a fixed-parameter-budget proof. Heads use identical
width and parameter counts. One seed and five epochs do not establish significance
or full convergence. Every run's best validation epoch was the final epoch.

**Not completed or claimed:** parameter-matched depth/width comparisons, completed
20-epoch runs, final test evaluation, retrieval-quality evaluation, or final
model selection. Do not mix this table with earlier full-budget/test scores.

## Candidate checkpoints available separately

Anson has packaged `geometry5_mae_handoff.zip` with inference-only weights,
architecture code, class order, preprocessing metadata, hashes and a working
inference/embedding helper. Raw data and optimizer states are omitted. Large
checkpoint binaries are not tracked in this repository.

- `lstm_geometry5_base`: 65.20% validation; 1 layer, hidden 256/direction; 512-D embedding.
- `lstm_geometry5_depth2`: 69.10%; 2 layers, hidden 256/direction; 512-D embedding.
- `lstm_geometry5_depth3`: 69.84%; 3 layers, hidden 256/direction; 512-D embedding.
- `transformer_geometry5_base`: 64.11%; 4 layers, width 128, 4 heads; 128-D embedding.
- `transformer_geometry5_depth6`: 66.23%; 6 layers, width 128, 4 heads; 128-D embedding.

These are candidates for discussion, not an automatic selection of the final
demo model. The checkpoint pack passed prediction and embedding identity checks
on six mixed-length validation sketches per candidate; see `handoff_audit.json`.

## Input and loading contract

1. The helper receives already normalized stroke-3 `[T,3]`, 1..250 points. The
   three columns are dx, dy, binary pen-up. Do not truncate longer sketches silently.
2. Apply per-sketch longest-bounding-box-side scaling and the saved training std
   exactly once. Never fit a new std to the user's drawing or to validation/test.
3. Reconstruct XY as `cumsum(normalized dx/dy) * coordinate_scale`; subtract the
   sketch's mean XY. Append these two columns to obtain `[T,5]`.
4. Classifier indices follow the checkpoint's `classes` list; do not regenerate
   label ordering independently. `load_candidate()` sets evaluation mode.
5. `model.embed()` returns unnormalized embeddings; the helper also returns
   top-five softmax scores. Scores are not calibrated confidence values.

From the repository or the extracted checkpoint package:

```python
import numpy as np
from experiments.geometry_short_budget.inference import load_candidate, extract_one

model, metadata = load_candidate("checkpoints/transformer_geometry5_depth6/inference.pt")
result = extract_one(model, metadata, np.load("already_normalized_stroke3.npy"))
embedding = result["embedding"]
top5 = result["top5"]
```

For retrieval, extract references and queries using the same checkpoint,
preprocessing and feature order. The reference index should contain training
sketches only. If using cosine search, apply the same L2-normalization rule to
reference and query embeddings. Retrieval accuracy has not been measured here.

## Items to align with Mae

- Confirm comparable sample/training budgets before combining Mae's Point3 results
  with this Geometry5 screening or selecting a final model.
- Agree on the final checkpoint; larger/deeper short-budget winners may not be
  the best fully trained or fastest demo models.
- Agree on frontend/backend payloads: raw versus processed stroke coordinates,
  coordinate units, point order, pen-up convention, and returned labels/scores/
  neighbor IDs and sketch coordinates. Then Anson can implement the frontend
  against a concrete backend contract.

See `CONCLUSION_ZH.md` for the Chinese explanation and `README.md` for all 14 rows.
