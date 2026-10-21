"""Read-only checks before requesting GPUs for the three directionality runs (login node is fine).

    python psc_run/preflight_direction345.py --data-dir .../processed_full --selection-dir selection
"""
import argparse
import hashlib
import json
from pathlib import Path
import py_compile

from run_direction345 import EXPECTED, adapter, resolve_config

RUNS = {(True, 256): 'results/lstm345_dir_bi256', (False, 512): 'results/lstm345_dir_uni512',
        (False, 256): 'results/lstm345_dir_uni256'}
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--selection-dir', required=True)
args = parser.parse_args()

cfg, categories, manifest_path, locations = resolve_config(args.data_dir)
print('settings equal archived BiLSTM-256 config apart from:', json.dumps(locations, indent=1))
data_summary = json.loads((manifest_path.parent.parent / 'verify_summary.json').read_text())
manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert data_summary['manifest_sha256'] == manifest_sha, 'data changed since verify_processed.py'
selection = Path(args.selection_dir).resolve()
summary = json.loads((selection / 'selection_summary.json').read_text())
assert summary['manifest_sha256'] == manifest_sha, 'data changed since make_selection.py'
ids = {}
for split, count in [('train', 1725000), ('valid', 172500)]:
    raw = (selection / f'{split}_sample_ids.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == summary[split]['ids_sha256']
    ids[split] = json.loads(raw)
    assert len(ids[split]) == len(set(ids[split])) == count, split
    assert all(i.split('/')[0] in categories and i.split('/')[1:3] == ['full', split] for i in ids[split])
assert set(ids['train']).isdisjoint(ids['valid'])
for directory in RUNS.values():
    assert not (adapter.ROOT / directory).exists(), f'output exists: {directory}'
py_compile.compile(str(Path(__file__).with_name('run_direction345.py')), doraise=True)
for (bidirectional, hidden), expected in EXPECTED.items():
    model_cfg = dict(cfg['model'], bidirectional=bidirectional, hidden_dim=hidden)
    model = adapter.build_model(model_cfg, 3, 345)
    got = (model.encoder.embedding_dim, sum(p.numel() for p in model.parameters()))
    assert got == expected, (bidirectional, hidden, got)
    print(f'{RUNS[(bidirectional, hidden)]}: bidirectional={bidirectional} hidden={hidden} '
          f'embedding_dim={got[0]} parameters={got[1]}')
print('PREFLIGHT PASSED: source hashes, archived settings, verified data, shared sample IDs '
      '(1,725,000 / 172,500, disjoint), fresh output paths, runner syntax, model sizes')
