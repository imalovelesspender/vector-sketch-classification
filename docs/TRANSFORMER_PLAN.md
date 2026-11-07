# Transformer implementation plan

## Scope and evidence

The midterm report (Section 7) proposes a Transformer encoder on the same inputs and masks,
compared with the best LSTM on accuracy, parameter count and training time per epoch.
The report baseline is BiLSTM-256 per direction: 75.19% validation / 75.40% test.
UniLSTM-512 reports 76.24% validation / 76.22% test. BiLSTM-512 has 76.26% validation;
its test result is absent. Choose comparison models using validation, not test rankings.
The external tutorial's 70% is context, not a controlled comparison.

Top-level src/model files are placeholders. Executable classifier, engine, data reader and
BiLSTM live under experiments/lstm_baseline_reference/historical/. Promote reusable code to src/
with explicit provenance before adding the new encoder; do not make new development depend
on an experiment archive. Keep completed-run provenance and checkpoint formats intact.

Local results/transformer_345_seed42/ contains checkpoints and configuration for an earlier
128-dimensional, four-layer Transformer. Its status reports 78.3067% best validation.
This is an existing artifact, not a reproduced result from this work. Its provenance explicitly
says historical sample equivalence is not certified. The matching implementation is absent
from current src/. Recover and audit the source and exact sample IDs before reusing weights
or claiming an improvement over report results. Retain these artifacts during cleanup.

## Initial encoder

stroke-3 [B,T,3] -> Linear(3,128) -> add sinusoidal point-position encoding ->
prepend learned CLS token -> 4 bidirectional Transformer encoder layers ->
LayerNorm(CLS output) -> Linear(128,345).

Use four heads, feed-forward width 384, GELU, dropout 0.1, pre-layer normalization,
batch_first=True and a final LayerNorm. These are starting settings, not established optima.
Train from scratch with supervised cross-entropy. This is not a reproduction of Sketch-BERT
pretraining. No additional coordinate normalization or feature changes in the first comparison.

Expose forward(strokes, lengths) -> embedding [B,128] and embedding_dim=128.
Retain the shared SketchClassifier interface, including embed() for later retrieval.
Use encoder='transformer' in the model factory. Add an explicit config and train/evaluate
entry points that share the data loader and metric calculations with the LSTM.

## Masking

Construct a Boolean [B,T] mask using arange(T) >= lengths[:,None], on the input device.
Prepend an all-False CLS column. Pass this as src_key_padding_mask; True marks padding.
Do not infer lengths from zero coordinates: valid points can be zero. Use full self-attention
without a causal mask because the report evaluates complete drawings. Validate 1 <= length <= T
and the configured maximum of 250 real points. Allocate positions for the additional CLS token.
Independently initialize encoder layers rather than retaining identical cloned initial weights.

## Data controls

Keep the same 345-category ordering, official split boundaries, stroke-3 representation,
training-fitted normalization, exclusion of lengths >250 (not truncation), and sample counts:
5,000 train / 500 validation / 500 test per class. Avoid applying normalization twice.
Seed 42 alone does not establish sample identity: persist and compare ordered sample IDs,
manifest hashes, class counts and preprocessing metadata. If historical IDs cannot be recovered,
train a fresh LSTM control and Transformer on an identical audited rebuilt subset, and label
historical report scores separately. Never use test samples for preprocessing or model selection.

## Training and evaluation

Initial optimization: AdamW, lr 3e-4, weight decay 0.01, batch size 128, clip gradients at 1.0,
plain cross-entropy, 5% linear warmup followed by cosine decay over a 10-epoch initial budget.
These choices are proposals. Use validation to tune a small lr/depth grid if needed; log all trials.
A model-specific optimizer comparison is a practical system comparison, not an architecture-only
ablation. Share the training/sample budget and report optimizer differences explicitly.

Select best validation accuracy; evaluate test only after settings are frozen. Report top-1,
top-5, per-class accuracy, parameter count, train/validation curves and training seconds per epoch.
Use the same hardware, precision and timing definition for speed comparisons. Record peak GPU
memory as useful context. Repeat the final comparison over at least three seeds if time permits.
Keep retrieval as a subsequent experiment once the plain classifier comparison is established.

## Verification before a full run

Check output/logit shapes and finite gradients; verify identical predictions in eval mode when
padding is extended or padded values change; compare a sketch alone and in mixed-length batches;
check max-length, length-one and invalid-length behavior; overfit a tiny subset; verify checkpoint
reload reproduces validation logits. Run a short three-class pilot before the 345-class experiment.

PyTorch API reference:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
