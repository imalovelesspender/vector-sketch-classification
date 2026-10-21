# Audit of the 345-class directionality ablation (2026-10-21)

All numbers below come from the files in [`results/`](results/), which are copied from
`/ocean/projects/cis260329p/mcui3/qd-ablation/results/` on PSC. Checkpoints (`best.pt`, `last.pt`),
the 48.6 MB / 4.6 MB sample-ID lists and the dataset stay on PSC. Their SHA-256 hashes are recorded here.

## 1. Jobs

All six jobs completed with exit code 0 on V100-32 nodes (GPU-shared, 1 GPU, 5 CPUs) [`results/jobs.txt`](results/jobs.txt).

| Job | Step | Elapsed |
|---|---|---|
| 49532872 | preprocessing + verification | 34:08 |
| 49532873 | preflight gate | 0:09 |
| 49532874_0 | BiLSTM-256x2 | 45:59 |
| 49532874_1 | UniLSTM-512 | 44:37 |
| 49532874_2 | UniLSTM-256 | 28:04 |
| 49532875 | test evaluation + summary | 1:13 |

The total is about 2.6 GPU-hours. Each training job includes about 6–7 minutes of data loading. The pure training time is 37.0 min for
UniLSTM-512 and 21.7 min for UniLSTM-256. The BiLSTM runner (`run_all345.py`) does not record it.

## 2. Model configurations and parameter counts

Source: [`results/checkpoint_audit.json`](results/checkpoint_audit.json). It was produced on PSC on CPU by building each
model from its saved `config.json` and loading `best.pt` and `last.pt` with `strict=True` ("All keys matched").

| Run | `bidirectional` | `hidden_dim` | Embedding (pooled) | LSTM params | Head params | Total |
|---|---|---|---|---|---|---|
| BiLSTM-256x2 | true | 256 | 512 | 2 × 267,264 = 534,528 | 345×512+345 = 176,985 | **711,513** |
| UniLSTM-512 | false | 512 | 512 | 1,058,816 | 176,985 | **1,235,801** |
| UniLSTM-256 | false | 256 | 256 | 267,264 | 345×256+345 = 88,665 | **355,929** |

There is one LSTM layer per direction, with `num_layers=1`, `dropout=0`, `pooling=last`,
`embedding_dim=null` (no projection) and stroke-3 input (3 features). A PyTorch LSTM direction has
`4H×3 + 4H×H + 4H + 4H` parameters: input weights, recurrent weights, and *two* bias vectors.
For H=256 that is 3,072 + 262,144 + 1,024 + 1,024 = 267,264.

**UniLSTM-256 has 355,929 parameters.** The value 335,929 quoted earlier was a transposed-digit typo. The
count is consistent across the formula, the model built from the config, the checkpoint state dicts, and
the runtime assertions in `preflight_independent345.py` and `run_direction345.py`.

Checkpoint epochs: BiLSTM-256x2 best = last = epoch 10. UniLSTM-512 best = epoch 6, last = epoch 10.
UniLSTM-256 best = last = epoch 10. Each checkpoint's stored `config.model` equals the run's `config.json`.

## 3. Fairness of the comparison

| Item | Status | Evidence |
|---|---|---|
| Code | identical | `5fe0e29` + archived `run_processed.py`; all source SHA-256s verified at runtime |
| Data / preprocessing | identical | one `processed_full` (manifest `bcb35d25…`), regenerated with Sihui's notebook; normalization scale 0.12035210854952101 = Sihui's value; split counts 43,988,874 / 862,500 / 862,500 [`results/data/`](results/data/) |
| Classes | identical | all 345, manifest order = archived label mapping |
| Train / valid sample IDs | **identical** | train `67e6b8af…` (1,725,000), valid `97160bec…` (172,500) for all three runs [`results/sample_ids_sha256.txt`](results/sample_ids_sha256.txt); asserted at runtime and again by the summary |
| Per-class counts | identical | 5,000 train / 500 valid for every class in every run |
| Training settings | identical | `config.json` of the three runs differs **only** in `model.bidirectional`, `model.hidden_dim`, `output_dir` |
| Optimizer / schedule | identical | Adam, lr 1e-3, weight decay 0, grad clip 1.0, batch 128, 10 epochs, no LR schedule, seed 42 |
| Checkpoint selection | identical | best validation accuracy; validated again after training |
| Test evaluation | identical procedure | same evaluation function, 500 test sketches per class drawn with the same seed and filter; test IDs were not exported, so their identity across runs is implied by the deterministic selection rather than hashed |
| Hardware / software | identical | V100-32, `pytorch/26.05-2.11-py3` (torch 2.11.0+cu126) |

Reproducibility check: this BiLSTM-256x2 run's `history.json` is **bit-identical** to Anson's archived
baseline (`docs/experiments/lstm_345class_20261014/history.json`). The test accuracy is also identical (0.7540173913).
So the regenerated data and sample selection reproduce his run exactly, even though the manifest file bytes differ.

What each comparison changes:

| Comparison | Direction | Hidden / direction | Embedding | Parameters |
|---|---|---|---|---|
| BiLSTM-256x2 vs UniLSTM-512 | changes | 256 vs 512 | **same (512)** | **differs: 711,513 vs 1,235,801 (×1.74)** |
| UniLSTM-256 vs BiLSTM-256x2 | changes | **same (256)** | **differs: 256 vs 512** | **differs: 355,929 vs 711,513 (×2.0)** |
| UniLSTM-256 vs UniLSTM-512 | same (uni) | 256 vs 512 | differs | differs (×3.5) |

No comparison holds both the embedding size and the parameter count fixed. Parameter-matched alternatives
built from the same code: BiLSTM-350x2 = 1,235,845 parameters (≈ UniLSTM-512). UniLSTM-378 = 709,851 (≈ BiLSTM-256x2).

## 4. Learning curves (from `history.json`)

| Epoch | Bi-256 val loss | Bi-256 val acc | Uni-512 train loss | Uni-512 val loss | Uni-512 val acc | Uni-256 val loss | Uni-256 val acc |
|---|---|---|---|---|---|---|---|
| 5 | 0.9575 | 74.51 | 0.7717 | 0.8930 | 75.94 | 1.0325 | 72.74 |
| 6 | 0.9423 | 74.91 | 0.7286 | **0.8881** | **76.24** | 1.0116 | 73.24 |
| 7 | 0.9336 | 75.12 | 0.6929 | 0.8912 | 76.24 | 0.9935 | 73.76 |
| 8 | 0.9330 | 75.12 | 0.6628 | 0.8913 | 76.23 | 0.9900 | 73.71 |
| 9 | 0.9323 | 75.19 | 0.6374 | 0.8997 | 76.22 | 0.9802 | 74.08 |
| 10 | **0.9287** | **75.19** | 0.6150 | 0.9108 | 75.99 | **0.9670** | **74.27** |

**Observed for UniLSTM-512:**
- Validation loss reaches its minimum at epoch 6 (0.8881). It is higher at every later epoch (0.9108 at epoch 10, +2.6%).
- Training loss keeps falling (0.7286 → 0.6150).
- The train−valid accuracy gap widens from 3.6 to 6.6 points.
- Validation accuracy plateaus for epochs 6–9 (76.24 → 76.22) and falls to 75.99 at epoch 10.

That is the usual sign that overfitting has begun, measured by validation loss. In accuracy terms the decline is small,
0.25 points by epoch 10. The best checkpoint (epoch 6) is the one that was evaluated.

**Observed for BiLSTM-256x2 and UniLSTM-256:** both reach their best validation loss and accuracy at the
last epoch (10), and their train−valid gaps are smaller (4.5 and 2.3 points).

**Not established (speculation):**
- Whether UniLSTM-512 would keep degrading.
- Whether the other two models would overtake it with more epochs.
- Whether the gaps exceed seed-to-seed variation; only seed 42 was run.

The 10-epoch budget plausibly favours the faster-converging larger model. That cannot be tested with these runs.

## 5. Precision

Each split has 172,500 sketches. The binomial standard error of an accuracy near 75% is about 0.10 points, so
differences of about 1 point are well beyond evaluation-sample noise. Training noise across seeds was not measured.
Per-sketch predictions were not saved, so paired tests such as McNemar are not available.
