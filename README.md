# Sequence Models and Retrieval for Vector Sketch Classification

**CMU 11-785 Introduction to Deep Learning — Open Project, Team 21**

## Team

- Anson Wang
- Mae Cui
- Sihui Lin
- Sohyun Jin

## Overview

We study vector sketch classification on the Google QuickDraw dataset. Instead
of rasterized images, drawings are handled as ordered stroke-coordinate
sequences. We compare an RNN baseline with a Transformer classifier, and then
test whether retrieving similar training sketches improves classification.

> **Status:** Proposal / early implementation stage. No experiments have been
> run yet.

## Research Questions

1. How effectively can sequence models classify vector drawings directly from
   stroke-coordinate sequences?
2. Does a Transformer improve classification compared with an RNN baseline?
3. Can retrieval of similar sketches further improve classification, especially
   for ambiguous drawings?

## Dataset

[The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)
contains millions of drawings across hundreds of object categories. Each
drawing is stored as an ordered sequence of stroke coordinates that records
the points and boundaries of each stroke (not SVG path commands).

We plan to use a manageable subset with the same preprocessing and
train/validation/test splits for every model. The subset size and
preprocessing details are still to be decided. See [`data/README.md`](data/README.md).

## Planned Models

1. **RNN baseline**: a recurrent sequence classifier guided by the TensorFlow
   QuickDraw recurrent classification tutorial.
2. **Transformer**: an attention-based sequence classifier over stroke
   sequences.
3. **Retrieval-augmented model**: retrieves similar sketches from a reference
   collection drawn from the training set and incorporates their
   representations into classification. Retrieval excludes the query itself,
   and validation and test sketches are never in the reference collection.

Architectures and hyperparameters are still to be decided.

## LSTM Classification Pipeline

```
stroke sequence (B, T, 3) + lengths
  -> LSTMEncoder (packed, padding-aware)      src/models/rnn.py
  -> sketch embedding z (B, D)
  -> linear head -> logits (B, num_classes)   src/models/classifier.py
  -> cross-entropy loss
```

### Setup

```bash
pip install -r requirements.txt
```

### 1. Prepare data

Download the per-category `.npz` files into `data/sketchrnn/` as described in
[`data/README.md`](data/README.md). Nothing is downloaded automatically.

To use the processed shards from `data_preprocessing/preprocess.ipynb`
(`data/processed_full/<category>/{train,valid,test}.npz` + `manifest.json`),
use `configs/lstm_baseline_processed.json`. It sets `"data_format": "processed"`
and `"normalize": false`, because those coordinates are already normalized;
labels are remapped to the position of each category in the config. Test the
reader with `python -m tests.processed_format_test` (synthetic fixture).

### Expected input format

- One `<category>.npz` per category, with keys `train`, `valid`, `test`.
- Each sketch is an `(seq_len, 3)` stroke-3 array `(dx, dy, pen_lift)`.
- `QuickDrawStrokeDataset` (`src/data/sketchrnn.py`) samples a configurable
  number of sketches per class, divides `dx, dy` by the standard deviation of
  the training offsets (Sketch-RNN normalization, reused for valid/test), and
  can optionally convert to stroke-5 (`"stroke_format": "stroke5"`).
- `collate_sketches` zero-pads each batch and returns
  `strokes` `(B, T_max, F)`, `lengths` `(B,)` and `labels` `(B,)`.

### 2. Smoke test

```bash
python -m tests.smoke_test                                   # synthetic stroke-3 data
python -m tests.smoke_test --data-dir data/sketchrnn --categories angel cat apple
```

Checks tensor shapes, padding handling, forward pass, backprop, that loss drops
when overfitting one tiny batch, checkpoint save/load, and a 2-epoch
train + evaluate run.

### 3. Train

```bash
python -m src.train --config configs/lstm_baseline.json
# overrides: --categories cat apple --epochs 2 --output-dir results/debug --device cpu --seed 1
```

`configs/lstm_baseline.json` holds the data subset sizes, model and optimizer
settings and the seed (starting defaults, not tuned). Each epoch prints training
loss/accuracy and validation loss/accuracy. Outputs in `output_dir`:

| file | content |
|------|---------|
| `best.pt` | checkpoint with the highest validation accuracy |
| `last.pt` | checkpoint after the last epoch |
| `history.json` | per-epoch train/valid loss and accuracy |
| `curves.png` | loss and accuracy curves |
| `config.json` | resolved config |

Checkpoints also store the categories, normalization scale factor and config, so
evaluation reproduces the training preprocessing.

### 4. Evaluate (final test)

Select the model on validation accuracy, then evaluate on the test split once:

```bash
python -m src.evaluate --checkpoint results/lstm_baseline/best.pt --split test
```

This prints and writes `metrics_test.json` next to the checkpoint.

### Sketch embeddings for retrieval

`SketchClassifier.embed(strokes, lengths)` returns `z`, and
`model(strokes, lengths, return_embedding=True)` returns `(logits, z)`. To dump
embeddings for a whole split:

```bash
python -m src.evaluate --checkpoint results/lstm_baseline/best.pt --split train \
    --save-embeddings results/lstm_baseline/train_embeddings.npz
```

The `.npz` contains `embeddings`, `labels`, `preds` and `categories`.

### Swapping the encoder

`SketchClassifier` only needs an encoder with `forward(strokes, lengths) -> z`
and an `embedding_dim` attribute. To add a Transformer, implement that interface
in `src/models/transformer.py` and register it in `build_model`
(`src/models/classifier.py`), then set `"encoder"` in the config.

## Evaluation Plan

- **Primary metric:** classification accuracy
- Shared evaluation setup across all models
- Possible experiments and ablations:
  - RNN vs. Transformer
  - Transformer vs. retrieval-augmented Transformer
  - Different numbers of retrieved examples (top-k)
  - Similar retrieved examples vs. randomly selected examples
  - Error analysis on ambiguous or difficult sketches

## Hypothesis

We hypothesize that a Transformer may better capture relationships across
strokes and that similar reference sketches may help classify ambiguous
drawings. These are hypotheses to test rather than expected guarantees.

## Repository Structure

```
vector-sketch-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/              # Local data only (not committed); see data/README.md
├── notebooks/         # Exploration notebooks
├── src/
│   ├── data/
│   │   └── quickdraw.py   # Sketch-RNN .npz dataset, padding collate, loaders
│   ├── models/
│   │   ├── rnn.py         # RNN baseline (LSTM encoder)
│   │   ├── classifier.py  # Encoder -> z -> linear head; build_model
│   │   ├── transformer.py # Transformer classifier
│   │   └── retrieval.py   # Retrieval-augmented classifier
│   ├── engine.py      # Shared train/eval loops, embedding extraction
│   ├── utils.py       # Seeding, checkpoints, plotting
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── smoke_test.py  # End-to-end smoke test (synthetic or tiny real subset)
├── configs/           # Experiment configs
├── results/           # Experiment outputs
└── docs/              # Proposal, reports, notes
```

## Progress

- [x] Project proposal
- [x] Repository setup
- [ ] Literature review
- [x] Choose deep learning framework (PyTorch)
- [ ] Dataset subset selection and preprocessing
- [ ] Train/validation/test splits
- [ ] RNN baseline (pipeline implemented, not yet trained/tuned)
- [ ] Transformer classifier
- [ ] Retrieval-augmented classifier
- [ ] Ablations (top-k, similar vs. random retrieval)
- [ ] Error analysis
- [ ] Midterm report
- [ ] Final report

## References

1. Google Creative Lab. *The Quick, Draw! Dataset.*
   https://github.com/googlecreativelab/quickdraw-dataset
2. TensorFlow. *Recurrent Neural Networks for Drawing Classification.*
   https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/recurrent_quickdraw.md
3. P. Xu, C. K. Joshi, and X. Bresson. *Multi-Graph Transformer for Free-Hand
   Sketch Recognition.* arXiv:1912.11258, 2019.
4. P. Xu et al. *SketchMate: Deep Hashing for Million-Scale Human Sketch
   Retrieval.* CVPR, 2018.
5. D. Ha and D. Eck. *A Neural Representation of Sketch Drawings.* ICLR, 2018.
