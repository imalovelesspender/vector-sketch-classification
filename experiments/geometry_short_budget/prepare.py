"""Select fixed stratified subsets of the already normalized full-data cache."""
import hashlib
import json
from pathlib import Path

import numpy as np


def prepare(source,target):
    source,target=Path(source),Path(target)
    parent=json.loads((source/'manifest.json').read_text())
    assert len(parent['categories'])==345
    if (target/'manifest.json').exists():
        ready=json.loads((target/'manifest.json').read_text())
        if ready['train_per_class']!=1000 or ready['valid_per_class']!=100:
            raise RuntimeError('Existing short cache uses a different budget')
        return ready
    target.mkdir(parents=True,exist_ok=True)
    for split,n in [('train',1000),('valid',100)]:
        arrays={k:np.load(source/split/f'{k}.npy',mmap_mode='r',allow_pickle=False)
                for k in ['points','offsets','labels','ids']}
        rng=np.random.default_rng(42)
        selected=[]
        for label in range(345):
            eligible=np.flatnonzero(arrays['labels']==label)
            if len(eligible)<n:raise RuntimeError('Insufficient samples in source cache')
            selected.extend(np.sort(rng.choice(eligible,n,replace=False)).tolist())
        selected=np.asarray(selected,dtype=np.int64)
        lengths=arrays['offsets'][selected+1]-arrays['offsets'][selected]
        folder=target/split;folder.mkdir(exist_ok=True)
        pts=np.lib.format.open_memmap(folder/'points.npy',mode='w+',dtype=np.float32,
                                     shape=(int(lengths.sum()),5))
        offsets=np.r_[0,np.cumsum(lengths)]
        for j,i in enumerate(selected):
            pts[offsets[j]:offsets[j+1]]=arrays['points'][arrays['offsets'][i]:arrays['offsets'][i+1]]
        pts.flush()
        np.save(folder/'offsets.npy',offsets)
        np.save(folder/'labels.npy',np.array(arrays['labels'][selected]))
        np.save(folder/'ids.npy',np.array(arrays['ids'][selected]))
        np.save(folder/'parent_indices.npy',selected)
        assert np.array_equal(np.bincount(arrays['labels'][selected],minlength=345),np.full(345,n))
        print(f'SHORT CACHE {split}: {len(selected)} sketches, all 345 classes',flush=True)
        del pts,arrays
    meta=dict(parent)
    meta.update(train_per_class=1000,valid_per_class=100,
        scope='USER-APPROVED SHORT BUDGET: 345 classes, 1000 train/100 valid per class, 5 epochs, seed 42',
        selection='fixed seed-42 stratified random subset of the historical-rule rebuilt cache',
        parent_manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
        normalization_fit='inherited unchanged: all original full-archive training sketches',
        protocol_change='User approved November 24 deadline screening; not the original 20-epoch experiment',
        test_used=False)
    (target/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return meta
