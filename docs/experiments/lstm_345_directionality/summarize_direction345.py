"""Comparison table and combined curves for the three directionality runs.

Also re-checks that all three runs used the same train/valid sample IDs and that
those equal the IDs predicted by verify_processed.py.
    python psc_run/summarize_direction345.py --verification .../processed_full_verification.json \
        --output results/lstm_345_directionality_summary
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [('BiLSTM-256x2', 'results/lstm_345_bi256', True, 256, 512, 711513),
        ('UniLSTM-512', 'results/lstm_345_uni512', False, 512, 512, 1235801),
        ('UniLSTM-256', 'results/lstm_345_uni256', False, 256, 256, 355929)]

parser = argparse.ArgumentParser()
parser.add_argument('--verification', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
predicted = json.loads(Path(args.verification).read_text())['predicted_sample_ids_sha256']
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)


def load(path):
    return json.loads(Path(path).read_text())


rows, histories = [], {}
for name, directory, bidirectional, hidden, embedding, parameters in RUNS:
    run = Path(directory)
    cfg = load(run / 'config.json')
    assert cfg['model']['bidirectional'] is bidirectional and cfg['model']['hidden_dim'] == hidden
    for split in ('train', 'valid'):
        digest = hashlib.sha256((run / f'{split}_sample_ids.json').read_bytes()).hexdigest()
        assert digest == predicted[split], f'{name}: {split} sample IDs differ from the verified prediction'
    history = load(run / 'history.json')
    histories[name] = history
    valid, test = load(run / 'metrics_valid.json'), load(run / 'metrics_test.json')
    runtime = run / 'training_runtime.json'
    rows.append({
        'run': name, 'bidirectional': bidirectional, 'hidden_dim': hidden,
        'embedding_dim': embedding, 'parameters': parameters,
        'best_epoch': valid['epoch'], 'valid_acc': valid['acc'], 'valid_loss': valid['loss'],
        'test_acc': test['acc'], 'test_loss': test['loss'],
        'final_train_acc': history['train_acc'][-1],
        'training_minutes': load(runtime)['training_seconds'] / 60 if runtime.exists() else None,
        'output_dir': directory,
    })

(output / 'comparison.json').write_text(json.dumps({
    'rows': rows, 'sample_ids_sha256': predicted,
    'note': 'single seed; checkpoint selected on validation; test evaluated once afterwards'}, indent=2))
lines = ['| Run | Bidirectional | Hidden | Embedding | Params | Best epoch | Valid acc | Test acc '
         '| Valid loss | Test loss | Final train acc | Train min |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    minutes = f"{r['training_minutes']:.0f}" if r['training_minutes'] is not None else 'n/a'
    lines.append(f"| {r['run']} | {r['bidirectional']} | {r['hidden_dim']} | {r['embedding_dim']} "
                 f"| {r['parameters']:,} | {r['best_epoch']} | {r['valid_acc']:.2%} | {r['test_acc']:.2%} "
                 f"| {r['valid_loss']:.4f} | {r['test_loss']:.4f} | {r['final_train_acc']:.2%} | {minutes} |")
table = '\n'.join(lines) + '\n'
(output / 'comparison.md').write_text(
    '# 345-class LSTM directionality ablation\n\nSame regenerated data, same train/valid sample IDs '
    f"(train {predicted['train'][:12]}…, valid {predicted['valid'][:12]}…), same settings; "
    'only direction and hidden size differ. Single seed.\n\n' + table)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, history in histories.items():
    epochs = range(1, len(history['val_acc']) + 1)
    line, = axes[0].plot(epochs, history['val_acc'], marker='o', label=f'{name} valid')
    axes[0].plot(epochs, history['train_acc'], linestyle='--', color=line.get_color(), label=f'{name} train')
    axes[1].plot(epochs, history['val_loss'], marker='o', color=line.get_color(), label=f'{name} valid')
    axes[1].plot(epochs, history['train_loss'], linestyle='--', color=line.get_color(), label=f'{name} train')
for ax, label in zip(axes, ('accuracy', 'loss')):
    ax.set_xlabel('epoch')
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle('345-class QuickDraw: BiLSTM-256x2 vs UniLSTM-512 vs UniLSTM-256')
fig.tight_layout()
fig.savefig(output / 'curves_comparison.png', dpi=150)
print(table)
print('SUMMARY WRITTEN:', output)
