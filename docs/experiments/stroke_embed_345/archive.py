"""Copy a finished run into docs/experiments/<name>/ in the team's archive layout.

    python delta_run/archive.py results/stroke_embed_345class_20261123_12345 \
        --name stroke_embed_345class_20261123 --logs results/stroke345-12345.log results/test345-12346.log

Copies: config.json, planned_config.json, history.json, curves.png, metrics_valid.json,
metrics_test.json (if present), training_runtime.json, {train,valid}_class_counts.json,
the sbatch/adapter scripts, the Slurm logs, and writes source_hashes.json (SHA-256 of every
src/ file, configs and delta_run/) plus a README.md skeleton.  Checkpoints and sample-ID lists
are NOT copied (too large / private), matching the other experiment folders.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ['config.json', 'planned_config.json', 'history.json', 'curves.png', 'metrics_valid.json',
        'metrics_test.json', 'training_runtime.json', 'train_class_counts.json', 'valid_class_counts.json']
SCRIPTS = ['run_processed.py', 'run_all345.py', 'train345.sbatch', 'evaluate_test.sbatch', 'archive.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir')
    p.add_argument('--name', required=True, help='docs/experiments/<name>')
    p.add_argument('--logs', nargs='*', default=[])
    a = p.parse_args()
    run, dest = Path(a.run_dir), ROOT / 'docs' / 'experiments' / a.name
    dest.mkdir(parents=True, exist_ok=False)
    copied = []
    for f in COPY:
        if (run / f).exists():
            shutil.copy2(run / f, dest / f); copied.append(f)
    for f in SCRIPTS:
        shutil.copy2(ROOT / 'delta_run' / f, dest / f)
    for log in a.logs:
        shutil.copy2(log, dest / Path(log).name)
    hashes = {str(f.relative_to(ROOT)): sha(f)
              for f in sorted(list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'configs').glob('*.json'))
                              + [f for f in (ROOT / 'delta_run').glob('*') if f.is_file()])}
    hashes['best.pt'] = sha(run / 'best.pt') if (run / 'best.pt').exists() else None
    hashes['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (dest / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    cfg = json.loads((run / 'config.json').read_text())
    hist = json.loads((run / 'history.json').read_text())
    mv = json.loads((run / 'metrics_valid.json').read_text()) if (run / 'metrics_valid.json').exists() else {}
    mt = json.loads((run / 'metrics_test.json').read_text()) if (run / 'metrics_test.json').exists() else None
    rt = json.loads((run / 'training_runtime.json').read_text()) if (run / 'training_runtime.json').exists() else {}
    best_ep = int(max(range(len(hist['val_acc'])), key=lambda i: hist['val_acc'][i])) + 1
    m = cfg['model']
    readme = f"""# {a.name}

Run directory on Delta: `{run}`  (job logs: {', '.join(Path(l).name for l in a.logs) or 'see results/'})
Git commit: `{hashes['git_commit']}`  ·  branch `stroke-base/embed`

| setting | value |
|---|---|
| representation | `{cfg['data'].get('representation', 'point_based')}` |
| encoder | `{m.get('encoder')}` |
| top LSTM | hidden {m.get('hidden_dim')} / dir, bidirectional={m.get('bidirectional')}, pooling={m.get('pooling')} |
| stroke token | dim {m.get('dim', '-')}, stroke_hidden {m.get('stroke_hidden', '-')}, max_strokes {m.get('max_strokes', '-')} |
| ablation flags | shape={m.get('use_shape', '-')}, location={m.get('use_location', '-')}, order={m.get('use_order', '-')} |
| data | 345 classes, 5,000 / 500 / 500 per class, >250 points excluded, seed {cfg['seed']} |
| training | {cfg['train']['epochs']} epochs, Adam lr {cfg['train']['lr']}, clip {cfg['train']['grad_clip']}, batch {cfg['data']['batch_size']} |

## Results

| split | top-1 | loss | note |
|---|---|---|---|
| valid (best, epoch {best_ep}) | {max(hist['val_acc']):.4f} | {hist['val_loss'][best_ep-1]:.4f} | checkpoint selection |
| test | {('%.4f' % mt['acc']) if mt else 'not evaluated'} | {('%.4f' % mt['loss']) if mt else '-'} | {'evaluated once on best.pt' if mt else 'run evaluate_test.sbatch after freezing settings'} |

Final-epoch train acc {hist['train_acc'][-1]:.4f}; training time {rt.get('training_seconds', float('nan'))/60:.1f} min on one A40.

Files: {', '.join(copied)}, scripts, Slurm logs, `source_hashes.json`.
Checkpoints and sample-ID lists stay in the run directory.

## Comparison baseline

BiLSTM-256 point_based (Anson, `lstm_345class_20261014`): 75.19% valid / 75.40% test, 711,513 params.
"""
    (dest / 'README.md').write_text(readme)
    print(f'archived -> {dest}\n' + readme)


if __name__ == '__main__':
    main()
