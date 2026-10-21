# Pooling ablation results

345 classes; 5,000 training, 500 validation and 500 test sketches per class; seed 42; 10 epochs per model. One-layer BiLSTM with hidden size 256 per direction and identical parameter count (711,513). Only pooling changes between runs. Test checkpoints were selected on validation before test evaluation.

| Pooling | Best validation | Test top-1 | Test top-5 | Selected epoch |
|---|---:|---:|---:|---:|
| last | 75.1930% | 75.4017% | 93.7704% | 10 |
| mean | 74.3101% | 74.2255% | 93.2638% | 10 |

Mean minus last test top-1: -1.1762 percentage points. This is one seed and a fixed epoch budget.

Both test evaluations use the same 172,500 test sketches. Sample-ID hashes are recorded in test_provenance.json. Full checkpoints, predictions, ordered sample IDs, configurations and logs are retained locally under results/pooling_ablation_seed42/ and remotely in project storage. Large run artifacts are excluded from Git; the compact results in this folder are suitable for the repository.

This experiment uses an independently audited rebuild of the processed dataset. It is a matched pooling comparison, not a claim of exact sample identity with the historical Anson run.
