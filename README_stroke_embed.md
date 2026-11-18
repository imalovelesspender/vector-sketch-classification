# stroke_based representation + stroke_embed encoder for the baseline LSTM

Base branch: `origin/lstm-pipeline`. Sihui's preprocessing is read unchanged.
Stroke tokens are built exactly as in Sihui's stroke-level Transformer notebook
(`transformer-baseline`, `notebooks/Transformer Stroke Level.ipynb`), so the LSTM
and Transformer stroke models differ only in the encoder that reads the tokens.

## Names

| config key | values | meaning |
|---|---|---|
| `data.representation` | `point_based` | (dx, dy, pen) point sequence -- baseline input |
|                       | `stroke_based` | points grouped into strokes; adds `stroke_pts (N,P,4)`, `stroke_lens`, `stroke_xy (N,2)`, `n_strokes` |
| `model.encoder`       | `lstm` | baseline `LSTMEncoder` |
|                       | `stroke_embed` | shape + location + order tokens -> baseline `LSTMEncoder` |

`stroke_based` geometry (notebook cell 8/10): absolute xy = cumsum(dx, dy) x sigma
(back to bbox units), centred on the sketch mean; each stroke's shape is
`(x - x0, y - y0, draw, end)` local to its start point; `stroke_xy` is the centred
start point. Pen-up jumps position the next stroke but are not part of its shape.
sigma is read from `manifest.json` `normalization.scale`.

## Data flow

```
processed_full (Sihui)  ->  build_datasets(data_format="processed", normalize=False)
  ->  collate_stroke_based            [src/data/stroke_collate.py]  (group_strokes in src/data/stroke_split.py)
  ->  StrokeLSTMEncoder               [src/models/stroke_embed.py]
        shape:    BiLSTM(4 -> 64x2) final states -> Linear(128 -> 128)
      + location: Linear(2 -> 128)(stroke_xy)
      + order:    Embedding(250 -> 128)[stroke index]      -> dropout -> (B, S, 128)
  ->  top LSTMEncoder (BiLSTM-256, last state)             -> z (B, 512)
  ->  Linear(512 -> 345)   (unchanged head)
```

## Files

| file | status |
|---|---|
| `src/data/stroke_split.py` | new -- `group_strokes`, `centred_coordinates` (no parameters) |
| `src/data/stroke_collate.py` | new -- `collate_stroke_based`, `get_collate(representation, sigma)` |
| `src/models/stroke_embed.py` | new -- `StrokeLSTMEncoder` |
| `configs/stroke_embed.json` | new |
| `src/data/quickdraw.py` | edit -- `build_dataloaders(..., collate_fn=None)` |
| `src/train.py` | edit -- reads `data.representation`, sigma from manifest |
| `src/engine.py` | edit -- forwards `stroke_*` tensors when present |
| `src/models/classifier.py` | edit -- passes extra kwargs; registers `stroke_embed` |
| `src/models/rnn.py` | edit -- `LSTMEncoder.forward` ignores extra kwargs |

Diffs for the edited files are in `patches/`. With `representation` absent and
`encoder: lstm` the code path is the original one; baseline runs are unaffected.

All four combinations run; `stroke_embed` with a `point_based` batch groups strokes
inside the model (needs `model.coordinate_scale`, set automatically by train.py).

## Checks run (CPU)

* `group_strokes` output == Sihui's notebook `add_centered_coordinates` + `group_strokes`,
  executed verbatim on the same batches (incl. 1-point sketches and consecutive pen-ups)
* synthetic shards in Sihui's npz layout -> `build_datasets` -> both representations x both
  encoders -> `run_epoch`, `extract_embeddings`
* collate path == in-model path; extra padding rows leave logits unchanged
* `use_order` / `use_location` / `use_shape` ablations build and run

Parameters (345 classes): stroke_embed 1,052,249 vs baseline BiLSTM-256 711,513 --
not parameter-matched (top-LSTM input is 128-d, not 3-d).

## Run

```bash
python -m src.train --config configs/stroke_embed.json --epochs 1 --categories cat apple --device cpu
```
For the 345-class cluster run, add a `--config` argument to the team adapter (`run_all345.py`)
and point it at `configs/stroke_embed.json`; the adapter already sets `data_format`/`normalize`.
