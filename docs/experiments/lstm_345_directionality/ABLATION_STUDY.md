## Ablation Study: LSTM Directionality

**Setup.** We test whether reading a sketch's stroke sequence in both directions helps classification
on all 345 QuickDraw categories. Three single-layer LSTM classifiers were trained under one protocol:
- the same preprocessed data (Sketch-RNN `.full` archives, per-sketch scaling to unit bounding box,
  global standard deviation fitted on the training split, scale 0.12035);
- the same sampled subset: 5,000 training and 500 validation sketches per class, sketches longer than
  250 points excluded, so 1,725,000 / 172,500 sketches with identical sample IDs in all three runs;
- the same training settings: Adam with learning rate 1e-3 and no weight decay, gradient clipping 1.0, batch size 128,
  10 epochs, seed 42, last-hidden-state pooling, no dropout.

Only the direction and the hidden size differ. The checkpoint with the highest validation accuracy was
selected and then evaluated once on 500 test sketches per class (172,500). All runs used one NVIDIA V100-32GB GPU.

**Table X.** 345-class results (single seed). Accuracy in %.

| Model | Hidden per direction | Embedding dim | Parameters | Best epoch | Val. acc. | Test acc. | Val. loss |
|---|---|---|---|---|---|---|---|
| BiLSTM-256×2 | 256 | 512 | 711,513 | 10 | 75.19 | 75.40 | 0.929 |
| UniLSTM-512 | 512 | 512 | 1,235,801 | 6 | **76.24** | **76.22** | **0.888** |
| UniLSTM-256 | 256 | 256 | 355,929 | 10 | 74.27 | 74.19 | 0.967 |

**Key findings.**
1. *Same embedding size (512):* the unidirectional UniLSTM-512 is 1.05 points more accurate on
   validation and 0.82 on test than the bidirectional BiLSTM-256×2. This comparison is not
   parameter-matched: UniLSTM-512 has 1.74× as many parameters.
2. *Same hidden size per direction (256):* adding a backward direction (UniLSTM-256 → BiLSTM-256×2)
   improves accuracy by 0.92 points (validation) and 1.21 points (test). This doubles both the
   embedding size and the parameter count.
3. Under this budget, spending capacity on a wider single direction was more effective than adding a backward pass.
   We do not observe a benefit specific to bidirectionality once the embedding size is fixed. The two
   effects are entangled with parameter count, however (see Limitations).
4. *Training dynamics (Fig. Y):*
   - UniLSTM-512 reaches its lowest validation loss at epoch 6. After that its validation loss rises every epoch
     (0.888 → 0.911) while its training loss keeps decreasing, and the train–validation accuracy gap
     grows from 3.6 to 6.6 points. Validation accuracy stays flat until epoch 9 and drops 0.25 points at epoch 10.
   - BiLSTM-256×2 and UniLSTM-256 were still improving at epoch 10.
5. *Reproducibility:* the BiLSTM-256×2 run reproduces the team's earlier baseline exactly: an identical
   per-epoch history and an identical test accuracy of 75.40%.

**Limitations.**
- One random seed per model. Seed-to-seed variance is unknown, so differences of about 1 point should be treated as
  indicative until replicated. The evaluation-sample standard error alone is about 0.1 point.
- Neither comparison holds both the embedding size and the parameter count constant.
  UniLSTM-512 has 1.74× the parameters of BiLSTM-256×2.
- The 10-epoch budget is fixed and has no learning-rate schedule. UniLSTM-512 peaked early, while the other two
  models had not converged. A longer budget could change the ranking.
- One architecture family (1-layer LSTM, last-state pooling, no dropout) and one subset size (5,000 per class).
- Test accuracy was computed once for each validation-selected checkpoint and was not used for selection.

**Figures.**
- Fig. Y: training and validation loss and accuracy for the three runs,
  `results/summary/curves_comparison.png`.
- Per-run curves: `results/bi256/curves.png`, `results/uni512/curves.png` and `results/uni256/curves.png`.
