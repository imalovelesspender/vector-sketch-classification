# 345-class LSTM directionality ablation

Same regenerated data, same train/valid sample IDs (train 67e6b8afdbd1…, valid 97160becfbfc…), same settings; only direction and hidden size differ. Single seed.

| Run | Bidirectional | Hidden | Embedding | Params | Best epoch | Valid acc | Test acc | Valid loss | Test loss | Final train acc | Train min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BiLSTM-256x2 | True | 256 | 512 | 711,513 | 10 | 75.19% | 75.40% | 0.9287 | 0.9221 | 79.68% | n/a |
| UniLSTM-512 | False | 512 | 512 | 1,235,801 | 6 | 76.24% | 76.22% | 0.8881 | 0.8834 | 82.58% | 37 |
| UniLSTM-256 | False | 256 | 256 | 355,929 | 10 | 74.27% | 74.19% | 0.9670 | 0.9651 | 76.54% | 22 |
