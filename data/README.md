# QuickDraw data

The entire official Sketch-RNN collection is in `sketchrnn/`: 345 standard
category archives and 345 `.full.npz` category archives (about 10.1 GiB total).
This is the vector representation chosen for this project. Raw NDJSON and
bitmap exports are not needed for the recurrent classification baseline.

Preprocessing code is tracked separately in **`../data_preprocessing/preprocess.ipynb`**.
This folder holds only local data and this README.
The notebook includes a download cell for teammates starting from a fresh clone.
No Google account or Cloud SDK is required.

From the repository root:

```sh
conda activate vector-classification
python -m pip install -r requirements.txt
python -m jupyterlab data_preprocessing/preprocess.ipynb
```

The notebook uses all 345 `.full.npz` archives by default. Set `USE_FULL=False`
in the setup cell to use the standard archives (70,000 train / 2,500 valid /
2,500 test per category). Choose one version per category; combining both
versions duplicates examples. Download metadata and official MD5 checksums
are recorded in `sketchrnn/download_manifest.json`.

Stored sketches use stroke-3 `[dx, dy, pen_up]`. Stroke-5 is the original
Sketch-RNN generative model's input format, not the archive storage format.
Stroke-3 plus lengths is sufficient for the recurrent classifier.

The notebook validates every sequence, preserves original train/valid/test
membership, and fits a shared coordinate standard deviation on training only, after uniform
per-sketch scaling to a longest bounding-box dimension of 1.
It retains all points: no subsampling, length cap, extra simplification, or
augmentation. Each worker processes one category at a time, with four workers
by default. Reduce `WORKERS` to 1 if memory is limited.

Prepared category shards go in `processed_full/` (or `processed_standard/`).
Each shard stores numeric `points`, sequence `offsets`, integer class `labels`,
and source-row `ids`. They load without pickle. A completion manifest records
labels, scale, source version, counts, and length statistics. The notebook
also plots sketches and demonstrates mixed-class batches with padding masks.

Processing all categories can take several minutes and use multiple GiB of
additional disk space. Completed outputs are reused automatically when
reopening the notebook. Set
`REBUILD=True` in the setup cell to regenerate them. Training should shuffle samples
across classes and respect lengths/masks. Retrieval references use training
only and exclude the query ID.

Unused standalone inspection scripts have been removed; plotting and inspection
are implemented directly in the notebook.
Downloads and prepared data are ignored by Git; the notebook and README are
tracked.

Source and attribution: [Google, Inc., Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), CC BY 4.0.

## Which preprocessing is already present?

Google documents the top-left alignment, uniform 255 scaling, 1-pixel
resampling, and RDP epsilon 2.0 pipeline for simplified NDJSON. The separate
Sketch-RNN archive section explicitly documents RDP epsilon 2.0; it does not
state that these archives implement that entire simplified-NDJSON pipeline.
They already contain relative stroke-3 sequences and existing splits.
We do not repeat RDP or resampling. We add our own uniform per-sketch scaling
to a unit bounding-box dimension before training-derived numerical scaling.

## Exactly what our notebook adds

1. Select the full archive for every category and preserve its train/valid/test rows.
2. Validate nonempty [T,3] arrays, finite values, and binary pen flags. Invalid data raises an error.
3. Convert coordinates to float32 and assign deterministic integer class labels and source-row IDs.
4. Reconstruct each sketch's bounding box and divide dx/dy by its longest dimension, preserving aspect ratio; zero-size sketches remain unchanged.
5. Fit one standard deviation across the geometrically scaled training dx/dy values, then apply it to every split. Pen flags stay unchanged.
6. Save variable-length sequences as flat points plus offsets, labels, and IDs, with a manifest of counts and length statistics.
7. At batch time only, zero-pad to the batch's longest sequence and return true lengths and a mask. Padding is not stored in the dataset.

There is no extra alignment, 255 scaling, resampling, RDP, truncation, clipping,
subsampling, augmentation, stroke reordering, or stroke-5 conversion.

## Teammate first run

Create/activate the environment and install `requirements.txt`, open the
notebook, and run its setup and download cells first. They create
`data/sketchrnn/` automatically. `DOWNLOAD_BOTH_VERSIONS=True` downloads
the entire collection; False downloads only the selected standard/full version.
New downloads are MD5-verified. Existing files of the expected size are reused;
set `VERIFY_EXISTING_DOWNLOADS=True` to recheck them. Processing then creates
the ignored prepared data locally.

Requirements include PyTorch, pandas, SciPy, scikit-learn, tqdm, NumPy,
Matplotlib, and the Jupyter tools. `math` and other
Python standard-library modules do not need separate installation. The notebook
also demonstrates conversion of batches into PyTorch tensors.

Uniform per-sketch scaling precedes dataset-wide numerical normalization.
The completion manifest records the geometric scaling policy and preprocessing
version; the notebook automatically rebuilds outputs from an older policy.
