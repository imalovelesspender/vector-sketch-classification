"""Finalize docs/experiments/<EXP>/ after runs have written into its results/ subfolders.

    python delta_run/archive.py docs/experiments/stroke_embed_345

Layout (same as Mae's lstm_345_directionality):
    docs/experiments/<EXP>/
      README.md                 <- written here: settings + results table for every results/<TAG>
      source_hashes.json        <- SHA-256 of src/, configs/, delta_run/, each run's best.pt, git commit
      .gitignore                <- best.pt, last.pt, *_sample_ids.json
wandb/ (large/private, stay on Delta)
      train-<TAG>-<job>.log, test-<TAG>-<job>.log   (copied by the sbatch scripts)
      run_processed.py, run_all345.py, *.sbatch     <- copied here for provenance
      results/<TAG>/            <- config.json, history.json, curves.png, metrics_valid.json,
                                   metrics_test.json, training_runtime.json, class counts, ...
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# repo root = first ancestor that holds src/ and configs/ (works from delta_run/ or docs/experiments/<EXP>/delta_run/)
ROOT = next(p for p in HERE.parents if (p / 'src').is_dir() and (p / 'configs').is_dir())
SCRIPTS = ['run_processed.py', 'run_all345.py', 'train345.sbatch', 'evaluate_test.sbatch', 'smoke.sbatch', 'archive.py', 'README.md']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    exp = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent   # default: the experiment dir holding this delta_run/
    runs = sorted(d for d in (exp / 'results').iterdir() if (d / 'config.json').exists())
    assert runs, f'no finished runs under {exp}/results'
    if HERE.parent != exp:   # scripts kept elsewhere -> snapshot them into the experiment dir
        (exp / 'delta_run').mkdir(exist_ok=True)
        for f in SCRIPTS:
            if (HERE / f).exists():
                shutil.copy2(HERE / f, exp / 'delta_run' / f)
    (exp / '.gitignore').write_text('results/*/best.pt\nresults/*/last.pt\nresults/*/*_sample_ids.json\nresults/*/wandb/\n')

    hashes = {str(f.relative_to(ROOT)): sha(f) for f in sorted(
        list((ROOT / 'src').rglob('*.py')) + list((ROOT / 'configs').glob('*.json'))
        + [f for f in HERE.glob('*') if f.is_file()])}
    for r in runs:
        hashes[f'{r.name}/best.pt'] = sha(r / 'best.pt') if (r / 'best.pt').exists() else None
    hashes['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (exp / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))

    rows, settings = [], []
    for r in runs:
        cfg, hist = load(r / 'config.json'), load(r / 'history.json')
        mt, rt = load(r / 'metrics_test.json'), load(r / 'training_runtime.json') or {}
        m, d = cfg['model'], cfg['data']
        best = max(range(len(hist['val_acc'])), key=lambda i: hist['val_acc'][i])
        settings.append(f"| `{r.name}` | `{d.get('representation', 'point_based')}` | `{m.get('encoder')}` | "
                        f"{m.get('hidden_dim')} / bi={m.get('bidirectional')} / {m.get('pooling')} | "
                        f"{m.get('dim', '-')} / {m.get('stroke_hidden', '-')} / {m.get('max_strokes', '-')} | "
                        f"shape={m.get('use_shape', '-')}, loc={m.get('use_location', '-')}, order={m.get('use_order', '-')} | "
                        f"{cfg['train']['epochs']} |")
        rows.append(f"| `{r.name}` | {best + 1} | {hist['val_acc'][best]:.4f} | {hist['val_loss'][best]:.4f} | "
                    f"{('%.4f' % mt['acc']) if mt else 'not evaluated'} | {hist['train_acc'][-1]:.4f} | "
                    f"{rt.get('training_seconds', 0) / 60:.1f} |")
    c0 = load(runs[0] / 'config.json')
    readme = f"""# {exp.name}

Delta, `/projects/bjwy/sjin5/vector-sketch-classification`, branch `stroke-base/embed`, commit `{hashes['git_commit']}`.
Data: Sihui's `processed_full` (345 classes, 5,000 / 500 / 500 per class, >250 points excluded, seed {c0['seed']}).
Training: Adam lr {c0['train']['lr']}, clip {c0['train']['grad_clip']}, batch {c0['data']['batch_size']}; checkpoint by best validation top-1; test evaluated once.

## Runs

| run | representation | encoder | top LSTM (hidden / bi / pooling) | token (dim / stroke_hidden / max_strokes) | ablation flags | epochs |
|---|---|---|---|---|---|---|
{chr(10).join(settings)}

## Results

| run | best epoch | valid top-1 | valid loss | test top-1 | final train acc | train min |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Context: BiLSTM-256 point_based reference (Anson, `lstm_345class_20261014`) 75.19% valid / 75.40% test, 711,513 params.

## Files

`results/<run>/`: config.json, planned_config.json, history.json, curves.png, metrics_valid.json,
metrics_test.json, training_runtime.json, {{train,valid}}_class_counts.json.
Checkpoints and sample-ID lists stay on Delta (git-ignored). Slurm logs `train-*.log`, `test-*.log`.
`source_hashes.json`: SHA-256 of src/, configs/, delta_run/, each best.pt, and the git commit.
Scripts copied here: {', '.join(SCRIPTS)}.
"""
    (exp / 'README.md').write_text(readme)
    print(readme)


if __name__ == '__main__':
    main()
