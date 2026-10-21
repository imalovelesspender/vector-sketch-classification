"""Integrity check of a processed_full/ directory produced by preprocess.ipynb.

    python verify_processed.py --data-dir .../data/processed_full --summary out.json

Checks the manifest metadata, every category shard, the training-fit normalization
(processed train coordinates must have unit standard deviation) and, for a few
categories, re-derives the processed sketches from the raw .full.npz archives.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--summary', required=True)
parser.add_argument('--reference-config', default=str(here / 'reference_bi256_config.json'))
args = parser.parse_args()

data = Path(args.data_dir).resolve()
raw_dir = data.parent / 'sketchrnn'
manifest_bytes = (data / 'manifest.json').read_bytes()
manifest = json.loads(manifest_bytes)
reference = json.loads(Path(args.reference_config).read_text())
SPLITS = ('train', 'valid', 'test')

# Manifest metadata.
assert manifest['archive_version'] == 'full'
assert manifest['representation'] == 'stroke-3'
assert manifest['preprocessing_version'] == 2
assert manifest['per_sketch_scaling'] == {'method': 'max_bbox_dimension', 'target': 1.0,
                                          'degenerate': 'identity'}
assert manifest['normalization']['fit_split'] == 'train'
scale = manifest['normalization']['scale']
assert np.isfinite(scale) and scale > 0
assert manifest['length_cap'] is None and manifest['subsampling'] is False
class_to_index = manifest['class_to_index']
categories = sorted(class_to_index, key=class_to_index.get)
assert len(categories) == 345 and categories == sorted(categories)
assert [class_to_index[c] for c in categories] == list(range(345))
# Same 345 classes, order and labels as Anson's archived BiLSTM-256 run.
assert categories == reference['data']['categories']
assert class_to_index == reference['data']['preprocessing']['label_mapping']
inventory = {Path(item['name']).name: item
             for item in json.loads((raw_dir / 'download_manifest.json').read_text())}
assert len(manifest['sources']) == 345
for record in manifest['sources']:
    assert record['file'] == record['category'] + '.full.npz'
    assert record['bytes'] == inventory[record['file']]['size'] == (raw_dir / record['file']).stat().st_size
assert sorted(r['category'] for r in manifest['sources']) == categories
for split in SPLITS:
    assert manifest['split_counts'][split] == sum(manifest['statistics'][c][split]['samples']
                                                  for c in categories)

# Every shard.
total = {s: {'samples': 0, 'points': 0} for s in SPLITS}
usable = {s: [] for s in SPLITS}  # per-class count of sketches with <= 250 points
xy_count, xy_sum, xy_square_sum = 0, 0.0, 0.0
for index, category in enumerate(categories):
    for split in SPLITS:
        with np.load(data / category / f'{split}.npz', allow_pickle=False) as shard:
            assert sorted(shard.files) == ['ids', 'labels', 'offsets', 'points']
            points, offsets, labels, ids = (shard[k] for k in ('points', 'offsets', 'labels', 'ids'))
        n = len(labels)
        stats = manifest['statistics'][category][split]
        assert points.dtype == np.float32 and points.ndim == 2 and points.shape[1] == 3
        assert offsets.dtype == np.int64 and labels.dtype == np.int64 and ids.dtype.kind == 'U'
        assert n == stats['samples'] == len(ids) == len(offsets) - 1 and len(points) == stats['points']
        lengths = np.diff(offsets)
        assert offsets[0] == 0 and offsets[-1] == len(points) and np.all(lengths > 0)
        assert np.all(labels == index)
        assert np.isfinite(points).all() and np.isin(points[:, 2], [0, 1]).all()
        assert ids.tolist() == [f'{category}/full/{split}/{i}' for i in range(n)]
        assert np.percentile(lengths, [0, 50, 90, 95, 99, 100]).tolist() == \
            list(stats['length_percentiles'].values())
        total[split]['samples'] += n
        total[split]['points'] += len(points)
        usable[split].append(int(np.sum(lengths <= reference['data']['max_seq_len'])))
        if split == 'train':
            xy = points[:, :2].astype(np.float64)
            xy_count += xy.size
            xy_sum += float(xy.sum())
            xy_square_sum += float(np.einsum('ij,ij->', xy, xy))
    if (index + 1) % 50 == 0:
        print(f'verified shards: {index + 1}/345', flush=True)
mean = xy_sum / xy_count
std = float(np.sqrt(xy_square_sum / xy_count - mean ** 2))
assert abs(std - 1.0) < 1e-3, f'processed train coordinate std {std}'


# Raw -> processed re-derivation for a few categories (independent of the notebook code).
def unit_scale(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    position = np.cumsum(sequence[:, :2], axis=0, dtype=np.float64)
    size = float(np.max(position.max(axis=0) - position.min(axis=0)))
    out = sequence.copy()
    out[:, :2] /= np.float32(size if size > 0 else 1.0)
    return out


spot = [categories[0], 'cat', categories[172], categories[-1]]
for category in spot:
    with np.load(raw_dir / f'{category}.full.npz', allow_pickle=True, encoding='latin1') as archive:
        for split in SPLITS:
            with np.load(data / category / f'{split}.npz', allow_pickle=False) as shard:
                points, offsets = shard['points'], shard['offsets']
            source = archive[split]
            assert len(source) == len(offsets) - 1
            for i in np.linspace(0, len(source) - 1, 25).astype(int):
                expected = unit_scale(source[i])
                expected[:, :2] /= scale
                assert np.allclose(points[offsets[i]:offsets[i + 1]], expected, rtol=1e-5, atol=1e-6), \
                    (category, split, i)

summary = {
    'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
    'anson_manifest_sha256': reference['data']['preprocessing']['manifest_sha256'],
    'normalization_scale': scale,
    'processed_train_xy_mean': mean, 'processed_train_xy_std': std,
    'split_counts': manifest['split_counts'], 'totals': total,
    'min_usable_per_class_le_250': {s: min(v) for s, v in usable.items()},
    'classes_below_limit': {'train': sum(v < 5000 for v in usable['train']),
                            'valid': sum(v < 500 for v in usable['valid']),
                            'test': sum(v < 500 for v in usable['test'])},
    'raw_spot_checked': spot,
}
Path(args.summary).write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print('PROCESSED DATA VERIFIED: manifest metadata, 345 classes/labels/order, 1035 shards, '
      'normalization (train std = 1), raw re-derivation spot check')
