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
- [x] PyTorch environment and local notebook kernel
- [x] Full Sketch-RNN download and preprocessing
- [x] Preserve official train/validation/test splits
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

## Local preprocessing setup

The Anaconda environment is named `vector-classification` (Python 3.12).
From Anaconda Prompt, change to this repository, then run:

```sh
conda activate vector-classification
python -m jupyterlab data_preprocessing/preprocess.ipynb
```

Select **Python (vector-classification)** as the notebook kernel.
To recreate the environment on another machine:

```sh
conda create -n vector-classification python=3.12 pip -y
conda activate vector-classification
python -m pip install -r requirements.txt
python -m ipykernel install --user --name vector-classification --display-name "Python (vector-classification)"
```

`data/` contains downloaded archives and prepared shards and is ignored except
for its README. `data_preprocessing/` contains the tracked notebook, including dataset
download, inspection, processing, and PyTorch batch conversion. The notebook uses all 345 full archives by default and
reuses completed prepared outputs. See `data/README.md` for exact preprocessing.

The notebook now downloads the dataset itself on first run; no data is required
in a fresh clone before starting it. Basic analysis packages and PyTorch are
included in `requirements.txt`. `math` is part of Python itself.

On this machine, the environment uses the official PyTorch CUDA 12.8 build for
the RTX 5060 GPU. Teammates can select a build for their own hardware using
https://pytorch.org/get-started/locally/. The default requirements remain portable.
