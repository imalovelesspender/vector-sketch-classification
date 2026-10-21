"""Integrity checks for a regenerated processed_full dataset (reads every shard).

Also predicts the exact train/valid sample IDs that Anson's adapter selects
(seed 42, 5000/500 per class, 0 < length <= 250) and records their SHA-256, so
the later training runs can be checked against them.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

EXPECTED_SCALE = 0.12035210854952101  # printed by Sihui's notebook run
EXPECTED_COUNTS = {'train': 43988874, 'valid': 862500, 'test': 862500}
LIMITS = {'train': 5000, 'valid': 500}

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--protocol-config', required=True)
parser.add_argument('--report', required=True)
args = parser.parse_args()
data = Path(args.data_dir)
protocol = json.loads(Path(args.protocol_config).read_text())
manifest_bytes = (data / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
mapping = protocol['data']['preprocessing']['label_mapping']
categories = sorted(mapping, key=mapping.get)
assert manifest['archive_version'] == 'full'
assert manifest['representation'] == 'stroke-3'
assert manifest['preprocessing_version'] == 2
assert manifest['per_sketch_scaling'] == {'method': 'max_bbox_dimension', 'target': 1.0, 'degenerate': 'identity'}
assert manifest['normalization']['fit_split'] == 'train'
assert manifest['length_cap'] is None and manifest['subsampling'] is False
assert manifest['class_to_index'] == mapping, 'class order differs from the protocol'
assert protocol['data']['categories'] == categories
scale = manifest['normalization']['scale']
print(f'scale {scale!r} expected {EXPECTED_SCALE!r} exact_match={scale == EXPECTED_SCALE}')
assert manifest['split_counts'] == EXPECTED_COUNTS, manifest['split_counts']

rngs = {split: np.random.default_rng(protocol['seed']) for split in LIMITS}
predicted = {split: [] for split in LIMITS}
shard_hashes, eligible_min = {}, {split: None for split in LIMITS}
for label, category in enumerate(categories):
    for split in ('train', 'valid', 'test'):
        path = data / category / f'{split}.npz'
        shard_hashes[f'{category}/{split}.npz'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as shard:
            points, offsets, labels, ids = (shard[k] for k in ('points', 'offsets', 'labels', 'ids'))
        n = len(labels)
        assert n == manifest['statistics'][category][split]['samples'] == len(ids) == len(offsets) - 1
        assert offsets[0] == 0 and offsets[-1] == len(points) and np.all(np.diff(offsets) > 0)
        assert points.dtype == np.float32 and points.shape[1] == 3 and np.isfinite(points).all()
        assert np.isin(points[:, 2], [0, 1]).all() and np.all(labels == label)
        assert ids.tolist() == [f'{category}/full/{split}/{i}' for i in range(n)]
        if split in LIMITS:
            # Same selection as run_processed.PreparedDataset, categories in order.
            lengths = np.diff(offsets)
            selected = np.flatnonzero(lengths > 0)
            selected = selected[lengths[selected] <= protocol['data']['max_seq_len']]
            count = len(selected)
            eligible_min[split] = count if eligible_min[split] is None else min(eligible_min[split], count)
            assert count >= LIMITS[split], f'{category}/{split}: only {count} eligible'
            selected = np.sort(rngs[split].choice(selected, LIMITS[split], replace=False))
            predicted[split].extend(str(ids[i]) for i in selected)
    if label % 50 == 0:
        print(f'checked {label + 1}/345', flush=True)

report = {
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'scale': scale, 'scale_matches_sihui': scale == EXPECTED_SCALE,
    'split_counts': manifest['split_counts'], 'min_eligible_per_class': eligible_min,
    'predicted_sample_ids_sha256': {
        split: hashlib.sha256(json.dumps(ids).encode()).hexdigest() for split, ids in predicted.items()},
    'predicted_sample_counts': {split: len(ids) for split, ids in predicted.items()},
    'shard_sha256': shard_hashes,
}
Path(args.report).write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'shard_sha256'}, indent=2))
print('PROCESSED DATA VERIFIED')
