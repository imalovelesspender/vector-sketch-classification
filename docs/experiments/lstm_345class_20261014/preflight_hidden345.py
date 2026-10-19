"""Read-only checks before requesting GPU resources."""
import hashlib
import json
from pathlib import Path
import py_compile

root = Path(__file__).resolve().parents[1]
baseline = root / 'results/lstm_processed_345class_20261014'
cfg = json.loads((baseline / 'config.json').read_text())
assert cfg['model']['hidden_dim'] == 256
assert cfg['model']['bidirectional'] is True
assert cfg['model']['pooling'] == 'last'
assert cfg['train']['epochs'] == 10 and cfg['seed'] == 42
assert len(cfg['data']['categories']) == 345
manifest = Path(cfg['data']['preprocessing']['manifest'])
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == cfg['data']['preprocessing']['manifest_sha256']
hashes = json.loads((root / 'psc_run/historical_source_hashes.json').read_text())
for path, expected in hashes.items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
for split, count in [('train', 1725000), ('valid', 172500)]:
    ids = json.loads((baseline / f'{split}_sample_ids.json').read_text())
    assert len(ids) == count and len(set(ids)) == count
for hidden in [128, 512]:
    assert not (root / f'results/lstm_345_hidden{hidden}_20261018').exists()
py_compile.compile(str(root / 'psc_run/run_hidden345.py'), doraise=True)
print('PREFLIGHT PASSED: historical source hashes, manifest, 256 settings, sample IDs, fresh output paths, Python syntax')
