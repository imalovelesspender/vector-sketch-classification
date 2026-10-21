"""Seed-aggregated comparison of BiLSTM-256x2, UniLSTM-512, UniLSTM-256 and UniLSTM-378.

Run from the historical checkout root after the seed array:
    python psc_run/summarize_seeds345.py --reference-test-ids results/seed_reference/test_sample_ids.json \
        --output results/lstm_345_seeds_summary
Seed 42 for the three original architectures comes from the existing runs; their
test IDs were tied to the reference list by preflight_seeds345.py.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARCHS = [('BiLSTM-256x2', True, 256, 512, 711513), ('UniLSTM-512', False, 512, 512, 1235801),
         ('UniLSTM-256', False, 256, 256, 355929), ('UniLSTM-378', False, 378, 378, 709851)]
SEED42 = {(True, 256): 'results/lstm_345_bi256', (False, 512): 'results/lstm_345_uni512',
          (False, 256): 'results/lstm_345_uni256'}
PAIRS = [('UniLSTM-512', 'BiLSTM-256x2', 'same embedding (512); Uni has 1.74x params'),
         ('BiLSTM-256x2', 'UniLSTM-256', 'same hidden/direction (256); Bi has 2x params, 2x embedding'),
         ('UniLSTM-378', 'BiLSTM-256x2', 'parameter-matched (709,851 vs 711,513); embedding 378 vs 512')]

parser = argparse.ArgumentParser()
parser.add_argument('--reference-test-ids', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--plan', default=str(HERE / 'seed_runs.tsv'))
args = parser.parse_args()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()  # noqa: E731
load = lambda path: json.loads(Path(path).read_text())  # noqa: E731
test_reference = sha(args.reference_test_ids)

runs = {(bi, h, 42): d for (bi, h), d in SEED42.items()}
for row in csv.DictReader(open(args.plan), delimiter='\t'):
    runs[(row['bidirectional'] == 'true', int(row['hidden_dim']), int(row['train_seed']))] = row['output_dir']

records, missing, id_hashes = [], [], set()
for name, bi, hidden, embedding, params in ARCHS:
    for seed in (42, 43, 44):
        directory = Path(runs[(bi, hidden, seed)])
        if not (directory / 'metrics_test.json').exists():
            missing.append(str(directory))
            continue
        cfg = load(directory / 'config.json')
        assert cfg['model']['bidirectional'] is bi and cfg['model']['hidden_dim'] == hidden and cfg['seed'] == seed
        ids = (sha(directory / 'train_sample_ids.json'), sha(directory / 'valid_sample_ids.json'))
        id_hashes.add(ids)
        test_ids = directory / 'test_sample_ids.json'
        if test_ids.exists():
            assert sha(test_ids) == test_reference, f'{directory}: test IDs differ'
            test_check = 'hash-identical'
        else:
            assert seed == 42, directory
            test_check = 'reproduced by preflight re-evaluation'
        history, valid, test = (load(directory / f) for f in ('history.json', 'metrics_valid.json', 'metrics_test.json'))
        records.append({'model': name, 'seed': seed, 'dir': str(directory), 'parameters': params,
                        'embedding_dim': embedding, 'best_epoch': valid['epoch'], 'valid_acc': valid['acc'],
                        'test_acc': test['acc'], 'valid_loss': valid['loss'], 'history': history,
                        'test_ids': test_check})
assert len(id_hashes) <= 1, f'train/valid sample IDs differ between runs: {id_hashes}'

output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
by_model = {name: [r for r in records if r['model'] == name] for name, *_ in ARCHS}


def mean_std(values):
    values = [100 * v for v in values]
    if not values:
        return 'n/a'
    return f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}' if len(values) > 1 else f'{values[0]:.2f}'


lines = ['# 345-class LSTM directionality: seeds 42/43/44 and parameter-matched UniLSTM-378', '',
         'Fixed data sample (selection seed 42; identical train/valid/test IDs in every run); only the '
         'training seed (initialization + shuffle) and the architecture vary. Mean ± sample std over seeds, %.', '',
         '| Model | Params | Embedding | n seeds | Valid acc | Test acc | Best epochs | Per-seed valid (42/43/44) |',
         '|---|---|---|---|---|---|---|---|']
for name, bi, hidden, embedding, params in ARCHS:
    rs = sorted(by_model[name], key=lambda r: r['seed'])
    lines.append(f"| {name} | {params:,} | {embedding} | {len(rs)} | {mean_std([r['valid_acc'] for r in rs])} "
                 f"| {mean_std([r['test_acc'] for r in rs])} | {', '.join(str(r['best_epoch']) for r in rs)} "
                 f"| {' / '.join('%.2f' % (100 * r['valid_acc']) for r in rs)} |")
lines += ['', '## Paired differences (same training seed), percentage points', '',
          '| Comparison | What else changes | Valid diff per seed | Valid mean | Test mean |', '|---|---|---|---|---|']
for a, b, note in PAIRS:
    seeds = sorted({r['seed'] for r in by_model[a]} & {r['seed'] for r in by_model[b]})
    get = lambda m, s, k: next(r[k] for r in by_model[m] if r['seed'] == s)  # noqa: E731
    dv = [100 * (get(a, s, 'valid_acc') - get(b, s, 'valid_acc')) for s in seeds]
    dt = [100 * (get(a, s, 'test_acc') - get(b, s, 'test_acc')) for s in seeds]
    lines.append(f"| {a} − {b} | {note} | {', '.join(f'{s}: {d:+.2f}' for s, d in zip(seeds, dv))} "
                 f"| {statistics.mean(dv):+.2f} | {statistics.mean(dt):+.2f} |" if seeds else f'| {a} − {b} | {note} | n/a | | |')
if missing:
    lines += ['', '**Missing runs:** ' + ', '.join(missing)]
(output / 'seeds_comparison.md').write_text('\n'.join(lines) + '\n')
(output / 'seeds_comparison.json').write_text(json.dumps({
    'runs': [{k: v for k, v in r.items() if k != 'history'} for r in records], 'missing': missing,
    'train_valid_ids_sha256': sorted(id_hashes), 'test_ids_sha256': test_reference}, indent=2))

fig, ax = plt.subplots(figsize=(7, 4.5))
for name, *_ in ARCHS:
    rs = by_model[name]
    if not rs:
        continue
    epochs = range(1, len(rs[0]['history']['val_acc']) + 1)
    per_epoch = list(zip(*[r['history']['val_acc'] for r in rs]))
    mean = [statistics.mean(v) for v in per_epoch]
    line, = ax.plot(epochs, mean, marker='o', label=f'{name} (n={len(rs)})')
    ax.fill_between(epochs, [min(v) for v in per_epoch], [max(v) for v in per_epoch], color=line.get_color(), alpha=0.2)
ax.set_xlabel('epoch')
ax.set_ylabel('validation accuracy (mean, min–max over seeds)')
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(output / 'seeds_val_acc.png', dpi=150)
print('\n'.join(lines))
if missing:
    raise SystemExit(f'{len(missing)} runs missing')
