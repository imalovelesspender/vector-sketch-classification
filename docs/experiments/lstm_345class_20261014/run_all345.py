"""345-class run using the same prepared-data adapter and Mae's model/loop."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import run_processed as adapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'GPU allocation required'
    manifest_path = Path(args.data_dir).resolve() / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    categories = sorted(manifest['class_to_index'], key=manifest['class_to_index'].get)
    assert len(categories) == 345
    cfg = json.loads((adapter.ROOT / 'configs/lstm_baseline.json').read_text())
    cfg.pop('_comment', None)
    cfg['data'].update(data_dir=str(manifest_path.parent), categories=categories)
    cfg['train']['device'] = 'cuda'
    cfg['output_dir'] = args.output_dir
    assert cfg['data']['train_per_class'] == 5000
    assert cfg['data']['val_per_class'] == 500
    assert cfg['train']['epochs'] == 10
    cfg['data']['preprocessing'] = {
        'format': 'sihui-processed-shards', 'already_normalized': True,
        'manifest': str(manifest_path),
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'label_mapping': {name: i for i, name in enumerate(categories)},
        'adapter': str(Path(adapter.__file__).resolve()),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'planned_config.json').write_text(json.dumps(cfg, indent=2))
    adapter.set_seed(cfg['seed'])
    start = time.monotonic()
    datasets = {}
    for split, limit in [('train', 5000), ('valid', 500)]:
        print(f'LOADING {split}: 345 classes, at most {limit} each', flush=True)
        datasets[split] = adapter.PreparedDataset(
            args.data_dir, categories, split, limit,
            cfg['data']['max_seq_len'], cfg['seed'])
        print(f'LOADED {split}: {len(datasets[split])} samples', flush=True)
    assert set(datasets['train'].ids).isdisjoint(datasets['valid'].ids)
    for split, ds in datasets.items():
        counts = np.bincount(ds.labels, minlength=345)
        assert len(counts) == 345 and np.all(counts > 0)
        (output / f'{split}_sample_ids.json').write_text(json.dumps(ds.ids))
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
        assert train_per_class == 5000 and val_per_class == 500
        assert all(s in datasets for s in splits), 'Test evaluation is not enabled'
        return {s: datasets[s] for s in splits}

    adapter.training.build_datasets = cached_build
    adapter.evaluation.build_datasets = cached_build
    print('BEGIN FORMAL 345-CLASS TRAINING', flush=True)
    history = adapter.training.train(cfg)
    assert len(history['val_acc']) == 10
    assert all(np.isfinite(values).all() for values in history.values())
    metrics = adapter.evaluation.evaluate(str(output / 'best.pt'), split='valid', device='cuda')
    assert abs(metrics['acc'] - max(history['val_acc'])) < 1e-8
    print('345-CLASS TRAINING AND CHECKPOINT VALIDATION COMPLETED; TEST SPLIT UNTOUCHED', flush=True)


if __name__ == '__main__':
    main()
