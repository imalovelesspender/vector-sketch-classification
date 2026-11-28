"""PSC-only adapter: use Sihui's prepared shards with Mae's unchanged model/loop."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.quickdraw import build_dataloaders, collate_sketches
import src.train as training
import src.evaluate as evaluation
from src.models.classifier import build_model
from src.utils import set_seed


class PreparedDataset(Dataset):
    def __init__(self, directory, categories, split, limit, max_length, seed):
        self.categories = list(categories)
        self.input_dim = 3
        self.num_classes = len(categories)
        self.scale_factor = 1.0  # Coordinates are already normalized; never divide again.
        self.sketches, self.labels, self.ids = [], [], []
        manifest = json.loads((Path(directory) / 'manifest.json').read_text())
        assert manifest['representation'] == 'stroke-3'
        assert manifest['normalization']['fit_split'] == 'train'
        rng = np.random.default_rng(seed)
        for local_label, category in enumerate(categories):
            with np.load(Path(directory) / category / f'{split}.npz', allow_pickle=False) as shard:
                offsets = shard['offsets']
                lengths = np.diff(offsets)
                labels = shard['labels']
                ids = shard['ids']
                assert len(offsets) == len(labels) + 1 == len(ids) + 1
                assert np.all(labels == manifest['class_to_index'][category])
                selected = np.flatnonzero(lengths > 0)
                if max_length is not None:
                    selected = selected[lengths[selected] <= max_length]
                if limit is not None and len(selected) > limit:
                    selected = np.sort(rng.choice(selected, limit, replace=False))
                if not len(selected):
                    raise ValueError(f'No usable samples: {category}/{split}')
                points = shard['points']
                assert offsets[-1] == len(points)
                for index in selected:
                    sample = np.array(points[offsets[index]:offsets[index + 1]], dtype=np.float32, copy=True)
                    assert sample.shape == (lengths[index], 3)
                    assert np.isfinite(sample).all()
                    assert np.isin(sample[:, 2], [0, 1]).all()
                    identifier = str(ids[index])
                    assert identifier.startswith(f'{category}/full/{split}/')
                    self.sketches.append(torch.from_numpy(sample))
                    self.labels.append(local_label)
                    self.ids.append(identifier)
        assert len(self.ids) == len(set(self.ids))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sketches[index], self.labels[index]


def build_prepared(directory, categories, train_per_class=None, val_per_class=None,
                   test_per_class=None, max_seq_len=None, stroke_format='stroke3',
                   seed=0, scale_factor=None, splits=('train', 'valid', 'test')):
    if stroke_format != 'stroke3':
        raise ValueError('This adapter only accepts already processed stroke3')
    if scale_factor not in (None, 1.0):
        raise ValueError('Refusing to normalize already normalized data again')
    limits = {'train': train_per_class, 'valid': val_per_class, 'test': test_per_class}
    result = {s: PreparedDataset(directory, categories, s, limits[s], max_seq_len, seed) for s in splits}
    if 'train' in result and 'valid' in result:
        assert set(result['train'].ids).isdisjoint(result['valid'].ids)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'This job requires its allocated GPU'
    training.build_datasets = build_prepared
    evaluation.build_datasets = build_prepared
    cfg = json.loads((ROOT / 'configs/lstm_baseline.json').read_text())
    cfg.pop('_comment', None)
    cfg['data']['data_dir'] = str(Path(args.data_dir).resolve())
    cfg['train']['device'] = 'cuda'
    cfg['output_dir'] = args.output_dir
    source = Path(args.data_dir) / 'manifest.json'
    cfg['data']['preprocessing'] = {
        'format': 'sihui-processed-shards', 'already_normalized': True,
        'manifest': str(source.resolve()),
        'manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'label_mapping': {name: i for i, name in enumerate(cfg['data']['categories'])},
        'adapter': str(Path(__file__).resolve()),
    }
    set_seed(cfg['seed'])
    probe = build_prepared(args.data_dir, cfg['data']['categories'], 32, 16,
                           max_seq_len=cfg['data']['max_seq_len'], seed=cfg['seed'],
                           splits=('train', 'valid'))
    batch = collate_sketches([probe['train'][i] for i in range(16)])
    model = build_model(cfg['model'], 3, len(cfg['data']['categories'])).cuda()
    model.eval()
    inputs = batch['strokes'].cuda()
    with torch.no_grad():
        logits = model(inputs, batch['lengths'])
        padded = torch.nn.functional.pad(inputs, (0, 0, 0, 7))
        assert torch.allclose(logits, model(padded, batch['lengths']), atol=1e-5, rtol=1e-5)
    model.train()
    loss = torch.nn.functional.cross_entropy(model(inputs, batch['lengths']), batch['labels'].cuda())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print('ADAPTER CHECKS PASSED: labels, IDs, splits, padding, forward/backward', flush=True)
    del model, inputs, logits, padded, probe
    torch.cuda.empty_cache()
    smoke = copy.deepcopy(cfg)
    smoke['data'].update(train_per_class=256, val_per_class=64)
    smoke['train']['epochs'] = 1
    smoke['output_dir'] = args.output_dir + '-smoke'
    training.train(smoke)
    print('REAL DATA SMOKE TRAINING COMPLETED', flush=True)
    datasets = build_prepared(args.data_dir, cfg['data']['categories'],
                              cfg['data']['train_per_class'], cfg['data']['val_per_class'],
                              max_seq_len=cfg['data']['max_seq_len'], seed=cfg['seed'],
                              splits=('train', 'valid'))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sample_ids.json').write_text(json.dumps({s: ds.ids for s, ds in datasets.items()}, indent=2))
    del datasets
    history = training.train(cfg)
    assert len(history['val_acc']) == cfg['train']['epochs']
    for values in history.values():
        assert np.isfinite(values).all()
    metrics = evaluation.evaluate(str(output / 'best.pt'), split='valid', device='cuda')
    assert abs(metrics['acc'] - max(history['val_acc'])) < 1e-8
    print('BASELINE TRAINING AND CHECKPOINT VALIDATION COMPLETED; TEST SPLIT UNTOUCHED', flush=True)


if __name__ == '__main__':
    main()
