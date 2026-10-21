"""Fix the train/valid sample IDs shared by all three directionality runs.

Uses the archived adapter's PreparedDataset exactly as run_all345.py does
(<=250 points, 5000/500 per class, default_rng(42) per split, manifest order).
Run from the historical checkout root:
    python psc_run/make_selection.py --data-dir .../processed_full --output-dir selection
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import run_processed as adapter

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()

reference = json.loads((here / 'reference_bi256_config.json').read_text())
manifest_path = Path(args.data_dir).resolve() / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
categories = sorted(manifest['class_to_index'], key=manifest['class_to_index'].get)
assert categories == reference['data']['categories']
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=False)
summary = {'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
datasets = {}
for split, limit in [('train', 5000), ('valid', 500)]:
    ds = adapter.PreparedDataset(args.data_dir, categories, split, limit,
                                 reference['data']['max_seq_len'], reference['seed'])
    counts = np.bincount(ds.labels, minlength=345).tolist()
    assert len(ds) == 345 * limit and counts == [limit] * 345
    anson_counts = json.loads((here / f'anson_{split}_class_counts.json').read_text())
    assert counts == anson_counts, f'{split} class counts differ from the archived run'
    ids_bytes = json.dumps(ds.ids).encode()
    (output / f'{split}_sample_ids.json').write_bytes(ids_bytes)
    (output / f'{split}_class_counts.json').write_text(json.dumps(counts))
    summary[split] = {'samples': len(ds), 'ids_sha256': hashlib.sha256(ids_bytes).hexdigest(),
                      'points': int(sum(len(s) for s in ds.sketches)),
                      'first_ids': ds.ids[:3]}
    datasets[split] = ds
    print(f'SELECTED {split}: {len(ds)} samples', flush=True)
assert set(datasets['train'].ids).isdisjoint(datasets['valid'].ids)
(output / 'selection_summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print('SELECTION FIXED: 1,725,000 train / 172,500 valid IDs, disjoint, class counts match archive')
