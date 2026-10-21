"""345-class unidirectional LSTM runs against a BiLSTM-256 reference run.

Modeled on run_hidden345.py: same adapter, same historical model/loop, same
sample IDs. The reference run (--baseline-dir) must have exactly the protocol of
Anson's archived BiLSTM-256 config (protocol_config.json, next to this file),
apart from absolute paths and output_dir. Only model.bidirectional (-> false),
model.hidden_dim and the output directory then differ from the reference.
Absolute paths (data directory, manifest, adapter) are compared by content:
historical source hashes and the exact train/valid sample-ID lists.
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
PROTOCOL_CONFIG_SHA256 = 'b62e06995f8db76ada5a890cdb63a7443e6a43d8b4d6a0fda4d323e41f64deb8'
# hidden_dim -> (embedding_dim, parameter_count) for a 1-layer unidirectional LSTM
# on stroke-3 input with a 345-class linear head.
EXPECTED = {256: (256, 355929), 512: (512, 1235801)}
# Fields that hold absolute locations rather than settings.
PATH_FIELDS = [('data', 'data_dir'), ('data', 'preprocessing', 'manifest'),
               ('data', 'preprocessing', 'adapter')]


def get_field(cfg, keys):
    for key in keys:
        cfg = cfg[key]
    return cfg


def set_field(cfg, keys, value):
    for key in keys[:-1]:
        cfg = cfg[key]
    cfg[keys[-1]] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hidden-dim', type=int, choices=sorted(EXPECTED), required=True)
    parser.add_argument('--baseline-dir', required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'GPU allocation required'
    baseline_dir = Path(args.baseline_dir).resolve()
    protocol_bytes = (Path(__file__).parent / 'protocol_config.json').read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest() == PROTOCOL_CONFIG_SHA256, \
        'protocol_config.json is not the archived BiLSTM-256 config'
    protocol = json.loads(protocol_bytes)
    reference_bytes = (baseline_dir / 'config.json').read_bytes()
    reference = json.loads(reference_bytes)
    # Our own regenerated data has its own manifest file, so its hash may differ;
    # categories, label mapping and every other setting must match exactly.
    comparable = copy.deepcopy(reference)
    for keys in PATH_FIELDS + [('output_dir',), ('data', 'preprocessing', 'manifest_sha256')]:
        set_field(comparable, keys, get_field(protocol, keys))
    assert comparable == protocol, 'Reference run settings differ from the archived protocol'
    assert reference['model']['hidden_dim'] == 256 and reference['model']['bidirectional'] is True
    expected_hashes = json.loads((Path(__file__).parent / 'historical_source_hashes.json').read_text())
    for relative, expected in expected_hashes.items():
        actual = hashlib.sha256((adapter.ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, f'Historical source changed: {relative}'
    print('HISTORICAL SOURCE HASHES VERIFIED', flush=True)
    manifest_path = Path(args.data_dir).resolve() / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    categories = sorted(manifest['class_to_index'], key=manifest['class_to_index'].get)
    assert len(categories) == NUM_CLASSES
    cfg = json.loads((adapter.ROOT / 'configs/lstm_baseline.json').read_text())
    cfg.pop('_comment', None)
    cfg['data'].update(data_dir=str(manifest_path.parent), categories=categories)
    cfg['train']['device'] = 'cuda'
    cfg['output_dir'] = reference['output_dir']
    assert cfg['data']['train_per_class'] == TRAIN_PER_CLASS
    assert cfg['data']['val_per_class'] == VAL_PER_CLASS
    assert cfg['train']['epochs'] == EPOCHS
    cfg['data']['preprocessing'] = {
        'format': 'sihui-processed-shards', 'already_normalized': True,
        'manifest': str(manifest_path),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'label_mapping': {name: i for i, name in enumerate(categories)},
        'adapter': str(Path(adapter.__file__).resolve()),
    }
    # Compare settings with the baseline's paths substituted in, then restore ours.
    actual_paths = {keys: get_field(cfg, keys) for keys in PATH_FIELDS}
    path_differences = {
        '.'.join(keys): {'baseline': get_field(reference, keys), 'this_run': value}
        for keys, value in actual_paths.items() if value != get_field(reference, keys)}
    for keys in PATH_FIELDS:
        set_field(cfg, keys, get_field(reference, keys))
    assert cfg == reference, 'Settings differ from the existing 256 baseline'
    for keys, value in actual_paths.items():
        set_field(cfg, keys, value)
    for name, change in path_differences.items():
        print(f'PATH ONLY DIFFERS FROM BASELINE: {name}: {change["this_run"]}', flush=True)
    cfg['model']['bidirectional'] = False
    cfg['model']['hidden_dim'] = args.hidden_dim
    cfg['output_dir'] = args.output_dir
    expected = copy.deepcopy(reference)
    expected['model']['bidirectional'] = False
    expected['model']['hidden_dim'] = args.hidden_dim
    expected['output_dir'] = args.output_dir
    for keys in PATH_FIELDS:
        set_field(expected, keys, get_field(cfg, keys))
    assert cfg == expected
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'planned_config.json').write_text(json.dumps(cfg, indent=2))
    adapter.set_seed(cfg['seed'])
    start = time.monotonic()
    datasets = {}
    for split, limit in [('train', TRAIN_PER_CLASS), ('valid', VAL_PER_CLASS)]:
        print(f'LOADING {split}: {NUM_CLASSES} classes, at most {limit} each', flush=True)
        datasets[split] = adapter.PreparedDataset(
            args.data_dir, categories, split, limit,
            cfg['data']['max_seq_len'], cfg['seed'])
        print(f'LOADED {split}: {len(datasets[split])} samples', flush=True)
    assert set(datasets['train'].ids).isdisjoint(datasets['valid'].ids)
    for split, ds in datasets.items():
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        assert len(counts) == NUM_CLASSES and np.all(counts > 0)
        baseline_ids = (baseline_dir / f'{split}_sample_ids.json').read_bytes()
        assert ds.ids == json.loads(baseline_ids), f'{split} sample IDs differ from baseline'
        per_class = TRAIN_PER_CLASS if split == 'train' else VAL_PER_CLASS
        assert len(ds) == NUM_CLASSES * per_class
        (output / f'{split}_sample_ids.json').write_text(json.dumps(ds.ids))
        print(f'EXACT {split} SAMPLE IDS VERIFIED AGAINST BILSTM-256 BASELINE', flush=True)
        (output / f'{split}_class_counts.json').write_text(json.dumps(counts.tolist()))
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
    parameter_count = sum(p.numel() for p in model.parameters())
    embedding_dim = model.encoder.embedding_dim
    assert (embedding_dim, parameter_count) == EXPECTED[args.hidden_dim]
    assert model.encoder.num_directions == 1
    del model
    adapter.set_seed(cfg['seed'])
    (output / 'ablation_metadata.json').write_text(json.dumps({
        'bidirectional': False, 'hidden_dim': args.hidden_dim,
        'embedding_dim': embedding_dim, 'parameter_count': parameter_count,
        'baseline': {'bidirectional': True, 'hidden_dim': 256, 'embedding_dim': 512,
                     'parameter_count': 711513},
        'baseline_dir': str(baseline_dir),
        'baseline_config_sha256': hashlib.sha256(reference_bytes).hexdigest(),
        'protocol_config_sha256': PROTOCOL_CONFIG_SHA256,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'path_differences_from_baseline': path_differences,
        'protocol_matches_archived_bilstm256': True,
        'source_hashes_verified': True, 'exact_train_valid_ids_verified': True,
        'test_evaluated': False,
    }, indent=2))
    adapter.training.build_datasets = cached_build
    adapter.evaluation.build_datasets = cached_build
    print(f'BEGIN FORMAL DIRECTIONALITY ABLATION bidirectional=False hidden={args.hidden_dim} '
          f'embedding={embedding_dim} classes={NUM_CLASSES} parameters={parameter_count}', flush=True)
    training_start = time.monotonic()
    history = adapter.training.train(cfg)
    elapsed = time.monotonic() - training_start
    (output / 'training_runtime.json').write_text(json.dumps({'training_seconds': elapsed}, indent=2))
    assert len(history['val_acc']) == EPOCHS
    assert all(np.isfinite(values).all() for values in history.values())
    metrics = adapter.evaluation.evaluate(str(output / 'best.pt'), split='valid', device='cuda')
    assert abs(metrics['acc'] - max(history['val_acc'])) < 1e-8
    print(f'{NUM_CLASSES}-CLASS TRAINING AND CHECKPOINT VALIDATION COMPLETED; TEST SPLIT UNTOUCHED', flush=True)


if __name__ == '__main__':
    main()
