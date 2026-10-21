# Pooling ablation

Controlled comparison of last-state and mean pooling for 345-class vector sketch classification. The experiment uses the archived one-layer BiLSTM, hidden size 256 per direction, identical training and validation examples, and a fixed 10-epoch budget.

The experiment notebook is [notebooks/pooling_ablation.ipynb](../../notebooks/pooling_ablation.ipynb).

## Implementation

- `run_pooling.py` connects the historical training loop to experiment configs and checks source hashes, preprocessing provenance, project account, class counts and ordered sample IDs.
- `historical/src/` and `historical/configs/lstm_baseline.json` preserve source from commit `5fe0e29ee627d1a36f979a52a7659eca2c761878`.
- `historical/psc_run/run_processed.py` preserves the processed-data adapter from results commit `fdea08baf749dff88d4d33a439dc5d277b992800`.
- `reference/` contains historical configurations and source hashes.
- `prepare_data.py` downloads checksum-verified official full archives and reproduces preprocessing version 2, including normalization fitted on all training points and full readback verification of saved shards.
- `build_notebook.py` and `notebook_content.py` generate the notebook and export pinned source files.
- `verify_workflow.py` validates the notebook and runs synthetic CPU checks for sample selection, pooling, padding and checkpoint evaluation.
- `evaluate_pair.py` evaluates validation-selected checkpoints on identical test samples and records predictions, sample IDs and checkpoint provenance.

Historical sources are isolated from the results branch's top-level placeholders. The model and training loop are unchanged. Both pooling strategies produce a 512-dimensional embedding with the same parameter count.

## Protocol

The paired experiment uses all 345 categories, 5,000 training and 500 validation sketches per category, a 250-point length exclusion, stroke-3 inputs, seed 42, batch size 128, Adam with learning rate 0.001, no weight decay or dropout, gradient clipping 1.0 and 10 epochs. Processed coordinates are not normalized a second time.

The last-state run saves its ordered training and validation IDs. The mean run requires exact agreement with those IDs. A private dataset rebuild has its own manifest hash and must pass a completed preparation audit, including the independently recomputed normalization scale and full dataset counts. A rebuilt paired run is not presented as an exact reproduction of the historical checkpoint. Mean-only execution against the historical baseline requires matching its configuration, manifest and original sample-ID lists. Training excludes test evaluation; the separate final evaluation uses 500 test sketches per class after both checkpoints are selected on validation.

Each run produces checkpoints, per-epoch history, validation metrics, sample IDs, class counts, training duration and source/job provenance. Both runs and final test evaluations are complete: last-state pooling achieved 75.4017% test accuracy and mean pooling achieved 74.2255%. See the [recorded results](../../docs/experiments/pooling_ablation_seed42/README.md). Historical results are labeled separately in the notebook.

## Validation

From the repository root, in a Python environment with PyTorch, NumPy, matplotlib and nbformat:

```bash
python experiments/pooling_ablation/verify_workflow.py
```

The checks use temporary synthetic data and do not submit jobs or train on QuickDraw.
