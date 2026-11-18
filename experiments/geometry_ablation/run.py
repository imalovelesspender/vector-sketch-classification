"""Run one-factor-at-a-time development ablations; never evaluate test data."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CachedDataset, collate, prepare_standard
from .models import configurations, make_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = correct = top5 = n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            top5 += (logits.topk(min(5, logits.size(-1)), dim=-1).indices == y[:, None]).any(-1).sum().item()
            n += len(y)
    return dict(loss=loss_sum/n, top1=correct/n, top5=top5/n, samples=n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive-dir', type=Path)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--train-per-class', type=int, default=500)
    p.add_argument('--valid-per-class', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--runs', nargs='*')
    p.add_argument('--resume', action='store_true', help='Resume an identical protocol from last.pt')
    a = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if a.prepare:
        prepare_standard(a.archive_dir, a.cache, ['airplane','apple','bicycle','cat','dog'],
                         a.train_per_class, a.valid_per_class, a.seed)
    meta = json.loads((a.cache / 'manifest.json').read_text())
    classes = len(meta['categories'])
    original_data={split:CachedDataset(a.cache,split,5) for split in ['train','valid']}
    # IDs are generated from unique source row indices, and include split names.
    # Check the loaded cache once, not once per architecture.
    for split,ds in original_data.items():
        if any(f'/{split}/' not in str(identifier) for identifier in ds.ids):
            raise ValueError('Cache ID split mismatch')
    cache_hashes={}
    for split in ['train','valid']:
        files=sorted((a.cache/split).glob('*.npy')) if (a.cache/split).is_dir() else [a.cache/f'{split}.npz']
        cache_hashes[split]={f.name:file_sha256(f) for f in files}
    matrix = configurations()
    if a.runs:
        unknown = set(a.runs) - {c['name'] for c in matrix}
        if unknown:
            raise ValueError(f'Unknown runs: {unknown}')
        matrix = [c for c in matrix if c['name'] in a.runs]
    a.output.mkdir(parents=True, exist_ok=a.resume)
    sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    protocol = dict(scope=meta['scope'], manifest=meta, matrix=matrix,
                    train=dict(epochs=a.epochs,batch_size=a.batch_size,lr=a.lr,optimizer='Adam',
                               weight_decay=0,clip=1,seed=a.seed,precision='float32'),
                    code_sha256=sources, cache_sha256=cache_hashes,
                    torch=torch.__version__, device=str(device),
                    gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                    selection='best validation top1; ties keep earliest epoch', test_used=False,
                    caveats=['Geometry-5 still uses point tokens, not one token per whole stroke.',
                             'Depth/width ablations change parameter count; heads ablation holds width fixed.',
                             'Transformer FF width is 3 x model width. LSTM width is per direction.',
                             'Historical comparison needs verified sample/preprocessing equivalence; validation is not test accuracy.',
                             'One seed does not establish statistical significance.'] +
                             (['Five-class standard-archive pilot is not the historical 345-class experiment.'] if classes==5 else []))
    if (a.output/'protocol.json').exists():
        existing=json.loads((a.output/'protocol.json').read_text())
        for key in ['matrix','train','cache_sha256','code_sha256']:
            if existing[key]!=protocol[key]:
                raise ValueError(f'Resume protocol mismatch: {key}')
    else:
        save_json(a.output/'protocol.json', protocol)
    source_folder=a.output/'source'
    source_folder.mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob('*.py'):
        (source_folder/f.name).write_bytes(f.read_bytes())
    results = []
    for config in matrix:
        seed_all(a.seed)
        name = config['name']
        folder = a.output/name
        folder.mkdir(exist_ok=a.resume)
        if a.resume and (folder/'metrics.json').exists():
            results.append(json.loads((folder/'metrics.json').read_text()))
            print('ALREADY COMPLETE', name, flush=True)
            continue
        model = make_model(config, classes).to(device)
        full_model = make_model(config, 345)
        params345 = sum(p.numel() for p in full_model.parameters())
        del full_model
        datasets={split:copy.copy(ds) for split,ds in original_data.items()}
        for ds in datasets.values(): ds.input_dim=config['input_dim']
        loaders = {split: DataLoader(ds, batch_size=a.batch_size, shuffle=split=='train',
                                     generator=torch.Generator().manual_seed(a.seed) if split=='train' else None,
                                     num_workers=0, collate_fn=collate)
                   for split,ds in datasets.items()}
        optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=0)
        history, best = [], -1.
        start_epoch=1
        prior_elapsed=0.
        if a.resume and (folder/'last.pt').exists():
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(last['model_state'])
            optimizer.load_state_dict(last['optimizer_state'])
            history,best=last['history'],last['best']
            start_epoch=last['epoch']+1
            prior_elapsed=last['elapsed_seconds']
            torch.set_rng_state(last['torch_rng'])
            if device.type=='cuda': torch.cuda.set_rng_state_all(last['cuda_rng'])
            loaders['train'].generator.set_state(last['loader_rng'])
            del last
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        print('BEGIN', name, 'parameters345=', params345, flush=True)
        for e in range(start_epoch,a.epochs+1):
            t = time.perf_counter()
            train = epoch(model, loaders['train'], device, optimizer)
            valid = epoch(model, loaders['valid'], device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            item = dict(epoch=e,train=train,valid=valid,seconds=time.perf_counter()-t)
            history.append(item)
            if valid['top1'] > best:
                best = valid['top1']
                torch.save(dict(model_state=model.state_dict(),config=config,classes=meta['categories'],
                                epoch=e,valid=valid,scope=meta['scope'],test_used=False,
                                optimizer_state=optimizer.state_dict(),protocol_sha256=
                                hashlib.sha256((a.output/'protocol.json').read_bytes()).hexdigest()), folder/'best.tmp')
                (folder/'best.tmp').replace(folder/'best.pt')
            save_json(folder/'history.json',history)
            torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
                            epoch=e,history=history,best=best,torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                            loader_rng=loaders['train'].generator.get_state(),
                            elapsed_seconds=prior_elapsed+time.perf_counter()-start),folder/'last.tmp')
            (folder/'last.tmp').replace(folder/'last.pt')
            print(name, 'epoch', e, 'train', round(train['top1'],4), 'valid', round(valid['top1'],4), flush=True)
        elapsed = prior_elapsed+time.perf_counter()-start
        ckpt = torch.load(folder/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        checked = epoch(model,loaders['valid'],device)
        assert checked['top1'] == best
        torch.save(dict(model_state=model.state_dict(),config=config,classes=meta['categories'],
                        best_epoch=ckpt['epoch'],epoch=a.epochs,scope=meta['scope']),folder/'selected_model.pt')
        row = dict(**config,best_epoch=ckpt['epoch'],valid_top1=checked['top1'],valid_loss=checked['loss'],
                   parameters_345_classes=params345,parameters_actual=sum(p.numel() for p in model.parameters()),
                   train_valid_seconds=elapsed,peak_gpu_memory_mb=
                   torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
                   checkpoint_reload_verified=True)
        results.append(row)
        save_json(folder/'metrics.json',row)
        del ckpt
        save_json(a.output/'summary.json',results)
        with (a.output/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]))
            writer.writeheader();writer.writerows(results)
    print('ALL DEVELOPMENT ABLATIONS COMPLETE; TEST SPLIT UNTOUCHED',flush=True)


if __name__ == '__main__':
    main()
