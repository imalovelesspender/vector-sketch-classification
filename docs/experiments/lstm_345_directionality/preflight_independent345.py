"""Read-only checks before requesting GPUs for the independent directionality runs.

Run from the historical checkout root on a login node:
    python psc_run/preflight_independent345.py --data-dir .../processed_full \
        --verification .../processed_full_verification.json
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import py_compile
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--verification', required=True)
args = parser.parse_args()
outputs = ['results/lstm_345_bi256', 'results/lstm_345_uni512', 'results/lstm_345_uni256']

protocol_bytes = (root / 'psc_run/protocol_config.json').read_bytes()
assert hashlib.sha256(protocol_bytes).hexdigest() == \
    'b62e06995f8db76ada5a890cdb63a7443e6a43d8b4d6a0fda4d323e41f64deb8', 'protocol_config.json'
protocol = json.loads(protocol_bytes)
hashes = json.loads((root / 'psc_run/historical_source_hashes.json').read_text())
for path, expected in hashes.items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
manifest_path = Path(args.data_dir).resolve() / 'manifest.json'
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
report = json.loads(Path(args.verification).read_text())
assert report['manifest_sha256'] == hashlib.sha256(manifest_bytes).hexdigest(), 'data changed since verification'
assert report['scale_matches_sihui'] is True
assert report['predicted_sample_counts'] == {'train': 1725000, 'valid': 172500}
categories = sorted(manifest['class_to_index'], key=manifest['class_to_index'].get)
for category in categories:
    for split in ('train', 'valid'):
        assert (manifest_path.parent / category / f'{split}.npz').is_file(), f'{category}/{split}'

# The reference config exactly as run_all345.py will resolve it.
cfg = json.loads((root / 'configs/lstm_baseline.json').read_text())
cfg.pop('_comment', None)
cfg['data'].update(data_dir=str(manifest_path.parent), categories=categories)
cfg['train']['device'] = 'cuda'
cfg['output_dir'] = outputs[0]
cfg['data']['preprocessing'] = {
    'format': 'sihui-processed-shards', 'already_normalized': True,
    'manifest': str(manifest_path),
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'label_mapping': {name: i for i, name in enumerate(categories)},
    'adapter': str((root / 'psc_run/run_processed.py').resolve()),
}
comparable = copy.deepcopy(cfg)
for keys in [('data', 'data_dir'), ('data', 'preprocessing', 'manifest'),
             ('data', 'preprocessing', 'adapter'), ('data', 'preprocessing', 'manifest_sha256')]:
    target = comparable
    for key in keys[:-1]:
        target = target[key]
    source = protocol
    for key in keys:
        source = source[key]
    target[keys[-1]] = source
comparable['output_dir'] = protocol['output_dir']
assert comparable == protocol, 'reference config would differ from the archived protocol'
print('REFERENCE CONFIG MATCHES ARCHIVED BILSTM-256 PROTOCOL (paths, output_dir, manifest hash aside)')

for directory in outputs:
    assert not (root / directory).exists(), f'output exists: {directory}'
for script in ('run_all345.py', 'run_direction345.py', 'run_processed.py'):
    py_compile.compile(str(root / 'psc_run' / script), doraise=True)

from src.models.classifier import build_model  # noqa: E402

for name, bidirectional, hidden, expected in [('bi256', True, 256, (512, 711513)),
                                               ('uni512', False, 512, (512, 1235801)),
                                               ('uni256', False, 256, (256, 355929))]:
    model = build_model(dict(protocol['model'], bidirectional=bidirectional, hidden_dim=hidden), 3, 345)
    got = (model.encoder.embedding_dim, sum(p.numel() for p in model.parameters()))
    assert got == expected, (name, got)
    print(f'{name}: embedding_dim={got[0]} parameters={got[1]}')
print('PREFLIGHT PASSED: protocol, source hashes, verified data, reference config, '
      'fresh outputs, syntax, model sizes')
