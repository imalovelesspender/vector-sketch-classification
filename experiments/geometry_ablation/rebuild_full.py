"""Rebuild a compact train/valid cache from the official full archives.

Uses Sihui's verified download and numerical preprocessing functions, not her
models or Sohyun's XAI encoder. Does not materialize the entire processed corpus.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from .data import geometry5


def json_save(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preprocess-worktree',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--train-per-class',type=int,default=5000)
    p.add_argument('--valid-per-class',type=int,default=500)
    a=p.parse_args()
    root=a.preprocess_worktree.resolve()
    cache=a.cache.resolve()
    raw=(root/'data_preprocessing/preprocess.ipynb').read_text(encoding='utf-8')
    nb=json.loads(raw)
    os.chdir(root)
    source=root/'data/sketchrnn'
    source.mkdir(parents=True,exist_ok=True)
    def progress(iterator,total,desc):
        for i,item in enumerate(iterator,1):
            if i%10==0 or i==total: print(f'{desc}: {i}/{total}',flush=True)
            yield item
    # Only the download and numerical functions are executed. Notebook plotting
    # and pandas imports are not needed for rebuilding this cache.
    ns=dict(Path=Path,json=json,np=np,ROOT=root,SOURCE=source,USE_FULL=True,
            DOWNLOAD_BOTH_VERSIONS=False,VERIFY_EXISTING_DOWNLOADS=True,WORKERS=4,
            ThreadPoolExecutor=ThreadPoolExecutor,as_completed=as_completed,tqdm=progress)
    exec(''.join(nb['cells'][3]['source']),ns)
    if a.download_only:
        print('345 FULL ARCHIVES DOWNLOADED AND MD5 VERIFIED',flush=True)
        return
    tree=ast.parse(''.join(nb['cells'][7]['source']))
    funcs=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),'<Sihui-preprocessing-functions>','exec'),ns)
    categories=ns['categories']
    assert len(categories)==345
    cache.mkdir(parents=True,exist_ok=True)
    normalization_file=cache/'normalization_progress.json'
    stats=json.loads(normalization_file.read_text()) if normalization_file.exists() else {}
    data_folder=root/'data/sketchrnn'
    for i,category in enumerate(categories,1):
        if category not in stats:
            with np.load(data_folder/f'{category}.full.npz',allow_pickle=True,encoding='latin1') as z:
                points,offsets,_=ns['pack_sequences'](z['train'])
                points=ns['uniform_scale'](points,offsets)
                xy=points[:,:2]
                stats[category]=dict(count=xy.size,total=float(np.sum(xy,dtype=np.float64)),
                    square_total=float(np.einsum('ij,ij->',xy,xy,dtype=np.float64)))
            json_save(normalization_file,stats)
            del points,xy,offsets
        if i%10==0 or i==345:
            print(f'TRAIN-ONLY NORMALIZATION {i}/345',flush=True)
    count=sum(stats[c]['count'] for c in categories)
    total=sum(stats[c]['total'] for c in categories)
    squares=sum(stats[c]['square_total'] for c in categories)
    sigma=float(np.sqrt(squares/count-(total/count)**2))
    assert sigma>0 and np.isfinite(sigma)
    print('FULL TRAINING COORDINATE SCALE',sigma,flush=True)
    # The historical adapter uses one RNG per split, in sorted category order.
    # Reproduce its selection rule; exact historical numerical identity still
    # requires archived preprocessing provenance / an external ID comparison.
    selections={}
    for split,limit in [('train',a.train_per_class),('valid',a.valid_per_class)]:
        rng=np.random.default_rng(42)
        selections[split]=[]
        total_points=0
        total_samples=0
        for category in categories:
            with np.load(data_folder/f'{category}.full.npz',allow_pickle=True,encoding='latin1') as z:
                seqs=z[split]
                lengths=np.fromiter((len(s) for s in seqs),dtype=np.int64,count=len(seqs))
                eligible=np.flatnonzero((lengths>0)&(lengths<=250))
                chosen=np.sort(rng.choice(eligible,limit,replace=False)) if len(eligible)>limit else eligible
                if len(chosen)!=limit:
                    raise RuntimeError(f'{category}/{split} has fewer than {limit} usable sketches')
                selections[split].append(chosen)
                total_points+=int(lengths[chosen].sum())
                total_samples+=len(chosen)
        folder=cache/split
        folder.mkdir(exist_ok=True)
        points=np.lib.format.open_memmap(folder/'points.npy',mode='w+',dtype=np.float32,shape=(total_points,5))
        offsets=np.lib.format.open_memmap(folder/'offsets.npy',mode='w+',dtype=np.int64,shape=(total_samples+1,))
        labels=np.lib.format.open_memmap(folder/'labels.npy',mode='w+',dtype=np.int64,shape=(total_samples,))
        ids=np.lib.format.open_memmap(folder/'ids.npy',mode='w+',dtype='<U100',shape=(total_samples,))
        cursor=row=0
        offsets[0]=0
        for label,category in enumerate(categories):
            with np.load(data_folder/f'{category}.full.npz',allow_pickle=True,encoding='latin1') as z:
                # Scale the complete original shard before selection, preserving
                # Sihui's per-shard cumulative/reduceat numerical procedure.
                packed,all_offsets,_=ns['pack_sequences'](z[split])
                packed=ns['uniform_scale'](packed,all_offsets)
                packed[:,:2]/=sigma
                for idx in selections[split][label]:
                    seq=packed[all_offsets[idx]:all_offsets[idx+1]]
                    features=geometry5(seq,sigma)
                    n=len(features)
                    points[cursor:cursor+n]=features
                    cursor+=n
                    offsets[row+1]=cursor
                    labels[row]=label
                    ids[row]=f'{category}/full/{split}/{int(idx)}'
                    row+=1
                del packed,all_offsets
            if (label+1)%10==0 or label==344:
                print(f'CACHE {split} {label+1}/345',flush=True)
        assert cursor==total_points and row==total_samples
        for arr in [points,offsets,labels,ids]: arr.flush()
        del points,offsets,labels,ids
    manifest=dict(categories=categories,coordinate_scale=sigma,seed=42,
        train_per_class=a.train_per_class,valid_per_class=a.valid_per_class,
        archive_sha256='Official MD5 hashes recorded in download_manifest.json',
        test_used=False,input_columns=['dx','dy','pen_up','centered_x','centered_y'],
        token_unit='point, not whole stroke',normalization_fit='all full-archive training sketches before filtering',
        scope='345-class full-archive rebuilt controls; historical equivalence requires verification',
        cache_format='npy_memmap',preprocessing_notebook_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        selection='historical PreparedDataset RNG rule: seed 42 independently per split, sorted categories',
        preprocessing_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    json_save(cache/'manifest.json',manifest)
    print('FULL 345-CLASS TRAIN/VALID CACHE READY; NO TEST PREDICTIONS COMPUTED',flush=True)


if __name__=='__main__':main()
