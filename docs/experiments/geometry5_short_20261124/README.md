# 345-class short-budget architecture screening

**User-approved November 24 deadline protocol. This is not the original 20-epoch experiment.**

All 14 configurations: 345,000 train / 34,500 validation sketches, all 345 classes, 5 epochs, seed 42, Adam 0.001, batch 128, float32. Same sampled IDs and training-fitted normalization for every model. No test-set predictions.

Inputs remain point tokens: point3 = dx/dy/pen; geometry5 adds centered XY. This is not whole-stroke tokenization or an XAI hierarchical encoder.

| Configuration | Best val. top-1 | Best epoch | Parameters | Train + valid seconds |
|---|---:|---:|---:|---:|
| lstm_point3_control | 63.27% | 5 | 711,513 | 1195.2 |
| lstm_geometry5_base | 65.20% | 5 | 715,609 | 1194.9 |
| transformer_point3_control | 58.23% | 5 | 706,905 | 929.9 |
| transformer_geometry5_base | 64.11% | 5 | 707,161 | 931.1 |
| lstm_geometry5_depth2 | 69.10% | 5 | 2,292,569 | 2193.7 |
| lstm_geometry5_depth3 | 69.84% | 5 | 3,869,529 | 3232.6 |
| lstm_geometry5_width128 | 58.86% | 5 | 226,905 | 1296.4 |
| lstm_geometry5_width512 | 67.55% | 5 | 2,479,449 | 1253.3 |
| transformer_geometry5_depth2 | 60.08% | 5 | 376,409 | 563.6 |
| transformer_geometry5_depth6 | 66.23% | 5 | 1,037,913 | 1298.8 |
| transformer_geometry5_width64 | 59.83% | 5 | 189,913 | 724.2 |
| transformer_geometry5_width256 | 60.12% | 5 | 2,724,697 | 1795.8 |
| transformer_geometry5_heads2 | 64.27% | 5 | 707,161 | 832.4 |
| transformer_geometry5_heads8 | 64.41% | 5 | 707,161 | 1230.8 |

Depth/width change parameter count. Transformer FF width stays 3x model width. Heads are varied at fixed width. A single seed and five epochs do not establish convergence, statistical significance or a universally best architecture. Historical full-budget and test scores must not be mixed into this table.

Source: team slides and Sihui preprocessing/model references documented in `experiments/geometry_ablation/README.md`. Cache source hashes, source-code hashes, sampling and the approved budget change are recorded in protocol.json.
