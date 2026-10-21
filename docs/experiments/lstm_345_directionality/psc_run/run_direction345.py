"""Independent 345-class LSTM directionality ablation: BiLSTM-256x2, UniLSTM-512, UniLSTM-256.

Same stack as Anson's archived run_all345.py / run_hidden345.py (commit 5fe0e29 +
run_processed.py, hash-checked). All three runs use the data regenerated in
mcui3/ and the sample IDs fixed once by make_selection.py; their settings must
equal the archived BiLSTM-256 config except for absolute paths, the manifest
hash, model.bidirectional, model.hidden_dim and output_dir.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_processed as adapter

NUM_CLASSES = 345
TRAIN_PER_CLASS, VAL_PER_CLASS, EPOCHS = 5000, 500, 10
REFERENCE_CONFIG_SHA256 = 'b62e06995f8db76ada5a890cdb63a7443e6a43d8b4d6a0fda4d323e41f64deb8'
# (bidirectional, hidden_dim) -> (embedding_dim, parameter_count), stroke-3 input, 345-class head.
EXPECTED = {(True, 256): (512, 711513), (False, 512): (512, 1235801), (False, 256): (256, 355929)}
# Fields that only record where this run's files are.
LOCATION_FIELDS = [('data', 'data_dir'), ('data', 'preprocessing', 'manifest'),
                   ('data', 'preprocessing', 'manifest_sha256'), ('data', 'preprocessing', 'adapter')]
HERE = Path(__file__).resolve().parent


def get_field(cfg, keys):
    for key in keys:
        cfg = cfg[key]
    return cfg


def set_field(cfg, keys, value):
    for key in keys[:-1]:
        cfg = cfg[key]
    cfg[keys[-1]] = value


def resolve_config(data_dir):
    """Build the config the way run_all345.py does and check it against the archive."""
    reference_bytes = (HERE / 'reference_bi256_config.json').read_bytes()
    assert hashlib.sha256(reference_bytes).hexdigest() == REFERENCE_CONFIG_SHA256
    reference = json.loads(reference_bytes)
    for relative, expected in json.loads((HERE / 'historical_source_hashes.json').read_text()).items():
        actual = hashlib.sha256((adapter.ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, f'Historical source changed: {relative}'
    manifest_path = Path(data_dir).resolve() / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    categories = sorted(manifest['class_to_index'], key=manifest['class_to_index'].get)
    assert len(categories) == NUM_CLASSES
    cfg = json.loads((adapter.ROOT / 'configs/lstm_baseline.json').read_text())
    cfg.pop('_comment', None)
    cfg['data'].update(data_dir=str(manifest_path.parent), categories=categories)
    cfg['train']['device'] = 'cuda'
    cfg['output_dir'] = reference['output_dir']
    cfg['data']['preprocessing'] = {
        'format': 'sihui-processed-shards', 'already_normalized': True,
        'manifest': str(manifest_path),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'label_mapping': {name: i for i, name in enumerate(categories)},
        'adapter': str(Path(adapter.__file__).resolve()),
    }
    locations = {'.'.join(k): get_field(cfg, k) for k in LOCATION_FIELDS}
    comparable = copy.deepcopy(cfg)
    for keys in LOCATION_FIELDS:
        set_field(comparable, keys, get_field(reference, keys))
    assert comparable == reference, 'Settings differ from the archived BiLSTM-256 config'
    return cfg, categories, manifest_path, locations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--selection-dir', required=True)
    parser.add_argument('--bidirectional', choices=['true', 'false'], required=True)
    parser.add_argument('--hidden-dim', type=int, choices=[256, 512], required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    bidirectional = args.bidirectional == 'true'
    assert (bidirectional, args.hidden_dim) in EXPECTED
    assert torch.cuda.is_available(), 'GPU allocation required'
    cfg, categories, manifest_path, locations = resolve_config(args.data_dir)
    print('HISTORICAL SOURCE HASHES AND ARCHIVED SETTINGS VERIFIED', flush=True)
    selection = Path(args.selection_dir).resolve()
    cfg['model']['bidirectional'] = bidirectional
    cfg['model']['hidden_dim'] = args.hidden_dim
    cfg['output_dir'] = args.output_dir
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'planned_config.json').write_text(json.dumps(cfg, indent=2))
    adapter.set_seed(cfg['seed'])
    start = time.monotonic()
    datasets = {}
    for split, limit in [('train', TRAIN_PER_CLASS), ('valid', VAL_PER_CLASS)]:
        print(f'LOADING {split}: {NUM_CLASSES} classes, at most {limit} each', flush=True)
        datasets[split] = adapter.PreparedDataset(
            args.data_dir, categories, split, limit, cfg['data']['max_seq_len'], cfg['seed'])
        print(f'LOADED {split}: {len(datasets[split])} samples', flush=True)
    assert set(datasets['train'].ids).isdisjoint(datasets['valid'].ids)
    for split, ds in datasets.items():
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        assert len(counts) == NUM_CLASSES and np.all(counts > 0)
        assert ds.ids == json.loads((selection / f'{split}_sample_ids.json').read_bytes()), \
            f'{split} sample IDs differ from the shared selection'
        assert len(ds) == NUM_CLASSES * (TRAIN_PER_CLASS if split == 'train' else VAL_PER_CLASS)
        (output / f'{split}_sample_ids.json').write_text(json.dumps(ds.ids))
        (output / f'{split}_class_counts.json').write_text(json.dumps(counts.tolist()))
        print(f'EXACT {split} SAMPLE IDS VERIFIED AGAINST SHARED SELECTION', flush=True)
    print(f'DATA READY in {time.monotonic() - start:.1f}s; '
          f'train={len(datasets["train"])} valid={len(datasets["valid"])}', flush=True)

    def cached_build(directory, selected_categories, train_per_class=None,
                     val_per_class=None, test_per_class=None, max_seq_len=None,
                     stroke_format='stroke3', seed=0, scale_factor=None,
                     splits=('train', 'valid', 'test')):
        assert Path(directory).resolve() == manifest_path.parent
        assert list(selected_categories) == categories
        assert max_seq_len == cfg['data']['max_seq_len']
        assert stroke_format == 'stroke3' and seed == cfg['seed']
        assert scale_factor in (None, 1.0)
        assert train_per_class == TRAIN_PER_CLASS and val_per_class == VAL_PER_CLASS
        assert all(s in datasets for s in splits), 'Test evaluation is not enabled'
        return {s: datasets[s] for s in splits}

    # Same seeding sequence as run_hidden345.py.
    adapter.set_seed(cfg['seed'])
    model = adapter.build_model(cfg['model'], 3, NUM_CLASSES)
    shape = (model.encoder.embedding_dim, sum(p.numel() for p in model.parameters()))
    assert shape == EXPECTED[(bidirectional, args.hidden_dim)], shape
    assert model.encoder.num_directions == (2 if bidirectional else 1)
    del model
    adapter.set_seed(cfg['seed'])
    (output / 'ablation_metadata.json').write_text(json.dumps({
        'bidirectional': bidirectional, 'hidden_dim': args.hidden_dim,
        'embedding_dim': shape[0], 'parameter_count': shape[1],
        'reference_config_sha256': REFERENCE_CONFIG_SHA256, 'locations': locations,
        'selection_summary': json.loads((selection / 'selection_summary.json').read_text()),
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_hashes_verified': True, 'exact_train_valid_ids_verified': True,
        'test_evaluated': False,
    }, indent=2))
    adapter.training.build_datasets = cached_build
    adapter.evaluation.build_datasets = cached_build
    print(f'BEGIN DIRECTIONALITY ABLATION bidirectional={bidirectional} hidden={args.hidden_dim} '
          f'embedding={shape[0]} parameters={shape[1]}', flush=True)
    training_start = time.monotonic()
    history = adapter.training.train(cfg)
    elapsed = time.monotonic() - training_start
    (output / 'training_runtime.json').write_text(json.dumps({'training_seconds': elapsed}, indent=2))
    assert len(history['val_acc']) == EPOCHS
    assert all(np.isfinite(values).all() for values in history.values())
    metrics = adapter.evaluation.evaluate(str(output / 'best.pt'), split='valid', device='cuda')
    assert abs(metrics['acc'] - max(history['val_acc'])) < 1e-8
    print('345-CLASS TRAINING AND CHECKPOINT VALIDATION COMPLETED; TEST SPLIT UNTOUCHED', flush=True)


if __name__ == '__main__':
    main()
