"""Audited development cache. No test-array access, no image conversion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def geometry5(stroke3, coordinate_scale):
    a = np.asarray(stroke3, dtype=np.float32)
    xy = np.cumsum(a[:, :2], axis=0) * coordinate_scale
    xy -= xy.mean(axis=0, keepdims=True)
    return np.concatenate([a, xy], axis=1).astype(np.float32)


def prepare_standard(archive_dir, output, categories, train_n, valid_n, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = {'train': [], 'valid': []}
    ids = {'train': [], 'valid': []}
    archive_hashes = {}
    for label, category in enumerate(categories):
        file = Path(archive_dir) / f'{category}.npz'
        archive_hashes[category] = hashlib.sha256(file.read_bytes()).hexdigest()
        with np.load(file, allow_pickle=True, encoding='latin1') as archive:
            for split, n in [('train', train_n), ('valid', valid_n)]:
                sketches = archive[split]  # Never read test.
                selected = 0
                for idx in np.random.default_rng(seed + label).permutation(len(sketches)):
                    a = np.asarray(sketches[idx], dtype=np.float32).copy()
                    if a.ndim != 2 or a.shape[1] != 3 or not 2 <= len(a) <= 250:
                        continue
                    if not np.isfinite(a).all() or not np.isin(a[:, 2], [0, 1]).all():
                        continue
                    extent = np.ptp(np.cumsum(a[:, :2], axis=0), axis=0).max()
                    if extent <= 0:
                        continue
                    a[:, :2] /= extent
                    data[split].append((a, label))
                    ids[split].append(f'{category}/standard/{split}/{int(idx)}')
                    selected += 1
                    if selected == n:
                        break
                if selected != n:
                    raise ValueError(f'Not enough samples: {category}/{split}')
    sigma = float(np.concatenate([a[:, :2].reshape(-1) for a, _ in data['train']]).std())
    assert sigma > 0 and set(ids['train']).isdisjoint(ids['valid'])
    for split, rows in data.items():
        arrays = []
        for a, _ in rows:
            a[:, :2] /= sigma
            arrays.append(geometry5(a, sigma))
        np.savez_compressed(output / f'{split}.npz', points=np.concatenate(arrays),
                            offsets=np.r_[0, np.cumsum([len(a) for a in arrays])],
                            labels=np.array([label for _, label in rows], dtype=np.int64),
                            ids=np.asarray(ids[split]))
    meta = dict(categories=categories, coordinate_scale=sigma, seed=seed,
                train_per_class=train_n, valid_per_class=valid_n,
                archive_sha256=archive_hashes, test_used=False,
                input_columns=['dx','dy','pen_up','centered_x','centered_y'],
                token_unit='point, not a whole-stroke token',
                normalization_fit='selected development training sketches only',
                scope='standard .npz development pilot; not historical .full.npz samples')
    (output / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


class CachedDataset:
    def __init__(self, folder, split, input_dim):
        folder=Path(folder)
        if (folder/split/'points.npy').exists():
            self.points,self.offsets,self.labels,self.ids=[np.load(folder/split/f'{k}.npy',mmap_mode='r',allow_pickle=False)
                                                         for k in ['points','offsets','labels','ids']]
        else:
            with np.load(folder / f'{split}.npz', allow_pickle=False) as z:
                self.points, self.offsets, self.labels, self.ids = [z[k] for k in ['points','offsets','labels','ids']]
        if self.offsets[0]!=0 or self.offsets[-1]!=len(self.points) or len(self.offsets)!=len(self.labels)+1:
            raise ValueError('Invalid cache offsets')
        self.input_dim = input_dim
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        a = self.points[self.offsets[i]:self.offsets[i+1], :self.input_dim]
        return torch.from_numpy(a.copy()), int(self.labels[i])


def collate(batch):
    seqs, labels = zip(*batch)
    return (pad_sequence(seqs, batch_first=True), torch.tensor([len(a) for a in seqs]),
            torch.tensor(labels, dtype=torch.long))
