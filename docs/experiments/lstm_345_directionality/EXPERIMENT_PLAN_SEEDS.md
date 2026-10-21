# Plan: seed replicates and a parameter-matched UniLSTM (prepared, not submitted)

## Goal

The first ablation ran one seed per model, and both of its comparisons changed the parameter count.
This round adds:

1. **Two more training seeds (43, 44)** for BiLSTM-256x2, UniLSTM-512 and UniLSTM-256, giving three seeds
   per model when combined with the existing seed-42 runs.
2. **UniLSTM-378**, parameter-matched to BiLSTM-256x2: 709,851 vs 711,513 parameters (−0.23%).
   It is run with seeds 42, 43 and 44.

| Task | Run | Direction | Hidden | Embedding | Params | Train seed |
|---|---|---|---|---|---|---|
| 0 | `lstm_345_uni378` | uni | 378 | 378 | 709,851 | 42 |
| 1–4 | `*_s43` (bi256, uni512, uni256, uni378) | | | | | 43 |
| 5–8 | `*_s44` (bi256, uni512, uni256, uni378) | | | | | 44 |

The existing seed-42 runs (`lstm_345_bi256`, `lstm_345_uni512`, `lstm_345_uni256`) are reused unchanged.
The exact rows are in [`seed_runs.tsv`](seed_runs.tsv).

## What is fixed and what varies

In the historical code a single `cfg['seed']` drives three things:
- sample selection, through `build_datasets` / `PreparedDataset`;
- the train-loader shuffle and the model initialization (`set_seed` in `train()`);
- the test subset that `evaluate()` re-draws.

Changing it would therefore change the data.
[`run_seeded345.py`](run_seeded345.py) separates the two roles:
- **Selection seed stays at 42.** Train, validation and test are each selected once with seed 42 and passed
  preloaded to `train()` and `evaluate()`, which ignore their own `seed` for data.
- **Training seed = `--train-seed`.** It sets `cfg['seed']`, so it controls only the initialization and shuffle order.
  The run's `config.json` differs from the seed-42 BiLSTM reference only in `model.bidirectional`,
  `model.hidden_dim`, `seed` and `output_dir`. This is asserted before training.

Everything else is identical to the first ablation: the code (`5fe0e29` + archived adapter, hash-checked), data
(`processed_full`, manifest `bcb35d25…`), 345 classes, 5,000/500/500 per class, ≤250 points, Adam 1e-3,
weight decay 0, clip 1.0, batch 128, 10 epochs, best-on-validation, and the V100-32 with `pytorch/26.05-2.11-py3`.

## Explicit sample-ID checks

| Split | Check | Where |
|---|---|---|
| train | byte-identical to `lstm_345_bi256/train_sample_ids.json` (sha `67e6b8af…`) | every run, before training; summary |
| valid | byte-identical to `lstm_345_bi256/valid_sample_ids.json` (sha `97160bec…`) | every run, before training; summary |
| test | byte-identical to `results/seed_reference/test_sample_ids.json` | every new run, before training; summary |
| test, seed-42 runs | those runs saved no test IDs. The gate re-evaluates their `best.pt` on the reference test IDs and requires the recorded `metrics_test.json` accuracy (≤ 2 sketches of 172,500) | gate |

Per-class counts (5,000/500/500) and pairwise disjointness of the three splits are also asserted.

## Jobs

`bash psc_run/submit_seeds.sh` runs from `qd-ablation/` only after approval. It refuses to run if any of these
jobs are already queued or if the balance is below 40 SU.

| Step | Script | Resources | Limit | Dependency |
|---|---|---|---|---|
| gate | `seeds_gate.sbatch` → `preflight_seeds345.py` | 1× V100-32, 5 CPU | 0:30 | – |
| 9 runs | `seeds345.sbatch`, array 0–8, 3 at a time | 1× V100-32, 5 CPU each | 1:30 each | afterok gate (+kill-on-invalid-dep) |
| summary | `seeds_summary.sbatch` → `summarize_seeds345.py` | 1× V100-32, 5 CPU (GPU-shared minimum) | 0:15 | afterany array |

All three scripts pass `sbatch --test-only` on Bridges-2, which validates them without queueing anything.

Outputs:
- `results/lstm_345_*_s43|_s44`, `results/lstm_345_uni378`: checkpoints, history, curves,
  `metrics_valid.json`, `metrics_test.json`, all three sample-ID lists and `run_metadata.json`.
- `results/seed_reference/test_sample_ids.json`.
- `results/lstm_345_seeds_summary/` (mean ± std per model, paired per-seed differences, validation curves with seed ranges).

## Cost and runtime estimate

Measured overnight (wall clock, V100-32, including about 6–7 min of data loading): BiLSTM-256x2 46.0 min,
UniLSTM-512 44.6 min, UniLSTM-256 28.1 min.

UniLSTM-378 is interpolated from the Uni-256 and Uni-512 training times (21.7 and 37.0 min) at about
29 min of training, or about 37 min wall clock. Each new run also loads and evaluates the test split, adding about 1 min.

| Item | GPU-hours |
|---|---|
| 2 × BiLSTM-256x2 | 1.57 |
| 2 × UniLSTM-512 | 1.52 |
| 2 × UniLSTM-256 | 0.97 |
| 3 × UniLSTM-378 | 1.85 |
| gate + summary | ~0.1 |
| **Expected total** | **≈ 6.0** |
| Ceiling (all time limits) | 14.25 |

**SU.** The balance went from 89 to 83 SU for about 2.6 GPU-hours overnight, so the effective charge was about 2.3 SU
per GPU-hour, or possibly a per-job minimum. Budget conservatively:
- **Expected: about 14 SU** (about 6 SU at the nominal 1 SU per GPU-hour).
- **Worst case: about 33 SU.**
- The balance is 83 SU, and the submit script requires at least 40.

**Wall time:** about 2–2.5 h with three tasks in parallel, plus queue wait.

**Storage:** about 80 MB per run (ID lists 54 MB, checkpoints 9–30 MB), about 0.8 GB in total. About 38 GB is free.

## Local test

[`test_seeds_local.py`](test_seeds_local.py) passed on CPU. It builds a 5fe0e29 checkout and synthetic shards for the
real 345 categories, with tiny counts and over-long sketches that must be filtered out, and runs a reduced
protocol (only counts, epochs, batch size and device differ). It covers:
- seed-42 stand-in runs → gate (reference test IDs, exact re-evaluation of seed-42 test metrics) → all 9 planned
  runs → summary;
- IDs byte-identical across every run and split;
- same training seed → bit-identical weights and history, while seeds 43 and 44 differ;
- negative tests that must fail: existing output, an unplanned architecture, altered test IDs,
  altered train IDs, a changed learning rate in the reference, and preflight with existing outputs.

## Interpretation plan

Fixed before seeing results:
- Report mean ± sample std over the three seeds, together with the per-seed values.
- Report paired per-seed differences for three comparisons:
  - Uni-512 − Bi-256: same embedding, 1.74× the parameters;
  - Bi-256 − Uni-256: same hidden size, 2× the parameters and 2× the embedding;
  - **Uni-378 − Bi-256: same parameters, embedding 378 vs 512.**
- A difference is called consistent only if it has the same sign for all three seeds. With n = 3 there is no
  significance test, and any claim should say so.
- Selection is on validation; test is reported once per run.
