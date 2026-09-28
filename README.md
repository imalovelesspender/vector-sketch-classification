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
│   ├── data/          # Data loading and preprocessing
│   ├── models/
│   │   ├── rnn.py         # RNN baseline
│   │   ├── transformer.py # Transformer classifier
│   │   └── retrieval.py   # Retrieval-augmented classifier
│   ├── train.py
│   └── evaluate.py
├── configs/           # Experiment configs
├── results/           # Experiment outputs
└── docs/              # Proposal, reports, notes
```

## Progress

- [x] Project proposal
- [x] Repository setup
- [ ] Literature review
- [ ] Choose deep learning framework
- [ ] Dataset subset selection and preprocessing
- [ ] Train/validation/test splits
- [ ] RNN baseline
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
