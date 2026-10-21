"""Seed-replicate / parameter-matched 345-class LSTM run with a fixed data sample.

Modeled on run_direction345.py, with one change: the data sample and the
training randomness use separate seeds. In the historical code a single
cfg['seed'] drives sample selection (via build_datasets), the train-loader
shuffle and the model initialization (set_seed inside train()), and
evaluate() re-draws the test subset with it. Here every split is selected once
with SELECTION_SEED (42, the archived protocol) and handed to train() and
evaluate() preloaded, so --train-seed only changes cfg['seed'], i.e. the
initialization and the shuffle order. Before training, the train/valid IDs must
equal the BiLSTM-256x2 seed-42 reference run and the test IDs must equal the
reference list written by preflight_seeds345.py.
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

HERE = Path(__file__).resolve().parent
NUM_CLASSES = 345
SELECTION_SEED = 42
PROTOCOL_PATH = HERE / 'protocol_config.json'
PROTOCOL_CONFIG_SHA256 = 'b62e06995f8db76ada5a890cdb63a7443e6a43d8b4d6a0fda4d323e41f64deb8'
DEVICE = 'cuda'
# (bidirectional, hidden_dim) -> (embedding_dim, parameter_count) on stroke-3 input, 345 classes.
EXPECTED = {(True, 256): (512, 711513), (False, 512): (512, 1235801),
            (False, 256): (256, 355929), (False, 378): (378, 709851)}
# Fields that hold absolute locations rather than settings.
LOCATION_FIELDS = [('data', 'data_dir'), ('data', 'preprocessing', 'manifest'),
                   ('data', 'preprocessing', 'manifest_sha256'), ('data', 'preprocessing', 'adapter'),
                   ('output_dir',)]


def get_field(cfg, keys):
    for key in keys:
        cfg = cfg[key]
    return cfg


def set_field(cfg, keys, value):
    for key in keys[:-1]:
        cfg = cfg[key]
    cfg[keys[-1]] = value


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--reference-dir', required=True, help='BiLSTM-256x2 seed-42 run directory')
    parser.add_argument('--reference-test-ids', required=True)
    parser.add_argument('--bidirectional', choices=['true', 'false'], required=True)
    parser.add_argument('--hidden-dim', type=int, required=True)
    parser.add_argument('--train-seed', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    bidirectional = args.bidirectional == 'true'
    assert (bidirectional, args.hidden_dim) in EXPECTED, 'architecture not in the plan'
    if DEVICE == 'cuda':
        assert torch.cuda.is_available(), 'GPU allocation required'

    protocol_bytes = PROTOCOL_PATH.read_bytes()
    assert sha256(protocol_bytes) == PROTOCOL_CONFIG_SHA256, 'protocol_config.json changed'
    protocol = json.loads(protocol_bytes)
    assert protocol['seed'] == SELECTION_SEED
    for relative, expected in json.loads((HERE / 'historical_source_hashes.json').read_text()).items():
        assert sha256((adapter.ROOT / relative).read_bytes()) == expected, f'Historical source changed: {relative}'
    reference_dir = Path(args.reference_dir).resolve()
    reference = json.loads((reference_dir / 'config.json').read_text())
    comparable = copy.deepcopy(reference)
    for keys in LOCATION_FIELDS:
        set_field(comparable, keys, get_field(protocol, keys))
    assert comparable == protocol, 'reference run settings differ from the archived protocol'
    assert reference['model']['bidirectional'] is True and reference['model']['hidden_dim'] == 256
    manifest_path = Path(args.data_dir).resolve() / 'manifest.json'
    assert Path(reference['data']['data_dir']).resolve() == manifest_path.parent, 'different data directory'
    assert reference['data']['preprocessing']['manifest_sha256'] == sha256(manifest_path.read_bytes()), \
        'processed data changed since the reference run'
    categories = reference['data']['categories']
    assert len(categories) == NUM_CLASSES
    print('PROTOCOL, SOURCE HASHES AND DATA MANIFEST VERIFIED', flush=True)

    # Only the model, the training seed and the output directory change.
    cfg = copy.deepcopy(reference)
    cfg['model']['bidirectional'] = bidirectional
    cfg['model']['hidden_dim'] = args.hidden_dim
    cfg['seed'] = args.train_seed
    cfg['output_dir'] = args.output_dir
    cfg['train']['device'] = DEVICE
    expected_cfg = copy.deepcopy(reference)
    for keys, value in [(('model', 'bidirectional'), bidirectional), (('model', 'hidden_dim'), args.hidden_dim),
                        (('seed',), args.train_seed), (('output_dir',), args.output_dir),
                        (('train', 'device'), DEVICE)]:
        set_field(expected_cfg, keys, value)
    assert cfg == expected_cfg
    data_cfg = cfg['data']
    limits = {'train': data_cfg['train_per_class'], 'valid': data_cfg['val_per_class'],
              'test': data_cfg['test_per_class']}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'planned_config.json').write_text(json.dumps(cfg, indent=2))
    start = time.monotonic()
    datasets, id_hashes = {}, {}
    for split in ('train', 'valid', 'test'):
        print(f'LOADING {split}: {NUM_CLASSES} classes, at most {limits[split]} each, '
              f'selection seed {SELECTION_SEED}', flush=True)
        ds = adapter.PreparedDataset(args.data_dir, categories, split, limits[split],
                                     data_cfg['max_seq_len'], SELECTION_SEED)
        ids_bytes = json.dumps(ds.ids).encode()
        reference_ids = (reference_dir / f'{split}_sample_ids.json' if split != 'test'
                         else Path(args.reference_test_ids))
        assert ids_bytes == reference_ids.read_bytes(), f'{split} sample IDs differ from the reference'
        assert len(ds) == NUM_CLASSES * limits[split]
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        assert counts.tolist() == [limits[split]] * NUM_CLASSES
        (output / f'{split}_sample_ids.json').write_bytes(ids_bytes)
        id_hashes[split] = sha256(ids_bytes)
        datasets[split] = ds
        print(f'EXACT {split} SAMPLE IDS VERIFIED ({len(ds)}, sha256 {id_hashes[split][:12]})', flush=True)
    assert set(datasets['train'].ids).isdisjoint(datasets['valid'].ids)
    assert set(datasets['train'].ids).isdisjoint(datasets['test'].ids)
    assert set(datasets['valid'].ids).isdisjoint(datasets['test'].ids)
    print(f'DATA READY in {time.monotonic() - start:.1f}s', flush=True)

    def preloaded_build(directory, selected_categories, train_per_class=None,
                        val_per_class=None, test_per_class=None, max_seq_len=None,
                        stroke_format='stroke3', seed=0, scale_factor=None,
                        splits=('train', 'valid', 'test')):
        # train() and evaluate() pass cfg['seed'] (the training seed) here; the
        # sample is fixed by SELECTION_SEED above, so it is deliberately ignored.
        assert Path(directory).resolve() == manifest_path.parent
        assert list(selected_categories) == categories
        assert max_seq_len == data_cfg['max_seq_len'] and stroke_format == 'stroke3'
        assert seed == args.train_seed and scale_factor in (None, 1.0)
        assert train_per_class == limits['train'] and val_per_class == limits['valid']
        assert test_per_class in (None, limits['test'])
        return {s: datasets[s] for s in splits}

    adapter.set_seed(args.train_seed)
    model = adapter.build_model(cfg['model'], 3, NUM_CLASSES)
    shape = (model.encoder.embedding_dim, sum(p.numel() for p in model.parameters()))
    assert shape == EXPECTED[(bidirectional, args.hidden_dim)], shape
    del model
    (output / 'run_metadata.json').write_text(json.dumps({
        'bidirectional': bidirectional, 'hidden_dim': args.hidden_dim,
        'embedding_dim': shape[0], 'parameter_count': shape[1],
        'train_seed': args.train_seed, 'selection_seed': SELECTION_SEED,
        'sample_ids_sha256': id_hashes, 'reference_dir': str(reference_dir),
        'protocol_config_sha256': PROTOCOL_CONFIG_SHA256,
        'runner_sha256': sha256(Path(__file__).read_bytes()),
        'differs_from_reference_in': ['model.bidirectional', 'model.hidden_dim', 'seed', 'output_dir'],
    }, indent=2))
    adapter.training.build_datasets = preloaded_build
    adapter.evaluation.build_datasets = preloaded_build
    print(f'BEGIN bidirectional={bidirectional} hidden={args.hidden_dim} train_seed={args.train_seed} '
          f'embedding={shape[0]} parameters={shape[1]}', flush=True)
    training_start = time.monotonic()
    history = adapter.training.train(cfg)
    elapsed = time.monotonic() - training_start
    (output / 'training_runtime.json').write_text(json.dumps({'training_seconds': elapsed}, indent=2))
    assert len(history['val_acc']) == cfg['train']['epochs']
    assert all(np.isfinite(values).all() for values in history.values())
    best = output / 'best.pt'
    valid = adapter.evaluation.evaluate(str(best), split='valid', device=DEVICE)
    assert abs(valid['acc'] - max(history['val_acc'])) < 1e-8
    # Test once, on the validation-selected checkpoint, after training is finished.
    before = sha256(best.read_bytes())
    adapter.evaluation.evaluate(str(best), split='test', device=DEVICE)
    assert sha256(best.read_bytes()) == before
    print('TRAINING, VALIDATION AND TEST EVALUATION COMPLETED', flush=True)


if __name__ == '__main__':
    main()
