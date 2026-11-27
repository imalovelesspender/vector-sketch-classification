"""Audited full-cache Transformer follow-up; training and validation only."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from .models import Classifier, configurations


def save(path, data):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8'); tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Split:
    def __init__(self, cache, split, audit=True):
        folder=Path(cache)/split
        self.points,self.offsets,self.labels=[np.load(folder/f'{k}.npy',mmap_mode='r')
                                              for k in ['points','offsets','labels']]
        self.lengths=np.diff(self.offsets)
        if audit:
            per_class=5000 if split=='train' else 500
            if len(self.labels)!=345*per_class or not np.all(np.bincount(self.labels,minlength=345)==per_class):
                raise ValueError('Full-cache class counts differ')
            if self.points.dtype!=np.float32 or self.points.shape[1]!=5 or self.offsets[0]!=0 or self.offsets[-1]!=len(self.points):
                raise ValueError('Invalid cache format')
            if self.lengths.min()<1 or self.lengths.max()>250:
                raise ValueError('Invalid sequence lengths')
            for start in range(0,len(self.points),1_000_000):
                chunk=self.points[start:start+1_000_000]
                if not np.isfinite(chunk).all() or not np.isin(chunk[:,2],[0,1]).all():
                    raise ValueError('Nonfinite or invalid pen flags')
            ids=np.load(folder/'ids.npy',mmap_mode='r')
            for start in range(0,len(ids),10000):
                if not all(f'/full/{split}/' in str(x) for x in ids[start:start+10000]):
                    raise ValueError('Unexpected split IDs')
            del ids

    def __len__(self): return len(self.labels)

    def batch(self, indices, device):
        lengths=self.lengths[indices]
        steps=np.arange(int(lengths.max()))[None,:]
        valid=steps<lengths[:,None]
        positions=self.offsets[indices,None]+np.minimum(steps,lengths[:,None]-1)
        x=np.array(self.points[positions],copy=True); x[~valid]=0
        return (torch.from_numpy(x).to(device),torch.from_numpy(np.array(lengths)).long().to(device),
                torch.from_numpy(np.array(self.labels[indices])).long().to(device))


def epoch(model, data, device, batch_size, bf16, optimizer=None, epoch_index=1,
          seed=42, step=0, total_steps=1, log_every=500):
    training=optimizer is not None
    model.train(training)
    sums=torch.zeros(4,device=device,dtype=torch.float64)
    order=np.random.default_rng(seed+epoch_index).permutation(len(data)) if training else np.arange(len(data))
    start=time.perf_counter()
    if device.type=='cuda': torch.cuda.synchronize(); start=time.perf_counter()
    with torch.enable_grad() if training else torch.inference_mode():
        for batch,start_i in enumerate(range(0,len(data),batch_size),1):
            x,lengths,y=data.batch(order[start_i:start_i+batch_size],device)
            if training:
                warmup=max(1,round(total_steps*.05))
                factor=((step+1)/warmup if step<warmup else
                        .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_steps-warmup-1)))))
                for group in optimizer.param_groups: group['lr']=3e-4*factor
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else nullcontext():
                logits=model(x,lengths); loss=F.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step(); step+=1
            top=logits.topk(min(5,logits.shape[1]),dim=1).indices
            sums+=torch.stack([loss.detach().double()*len(y),(top[:,0]==y).sum().double(),
                               (top==y[:,None]).any(1).sum().double(),
                               torch.tensor(len(y),device=device,dtype=torch.float64)])
            if log_every and batch%log_every==0:
                print(json.dumps(dict(event='batch_progress',split='train' if training else 'valid',
                      epoch=epoch_index,batches=batch,total_batches=math.ceil(len(data)/batch_size),
                      seconds=round(time.perf_counter()-start,1))),flush=True)
    if device.type=='cuda': torch.cuda.synchronize()
    loss_sum,correct,top5,count=sums.tolist()
    return dict(loss=loss_sum/count,top1=correct/count,top5=top5/count,samples=int(count),
                seconds=time.perf_counter()-start),step


def checkpoint(path, payload):
    tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=15); p.add_argument('--runs',nargs='*')
    p.add_argument('--device',default='cuda'); a=p.parse_args()
    torch.set_num_threads(4); device=torch.device(a.device)
    if a.epochs<1: raise ValueError('Positive epoch budget required')
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    bf16=device.type=='cuda' and torch.cuda.is_bf16_supported()
    a.output.mkdir(parents=True,exist_ok=True)
    matrix=configurations()
    if a.runs:
        if set(a.runs)-{r['name'] for r in matrix}: raise ValueError('Unknown run')
        matrix=[r for r in matrix if r['name'] in a.runs]
    meta=json.loads((a.cache/'manifest.json').read_text())
    if len(meta['categories'])!=345 or meta['train_per_class']!=5000 or meta['valid_per_class']!=500:
        raise ValueError('Expected full 345-class rebuilt cache')
    print('AUDITING full train and validation cache',flush=True)
    data={split:Split(a.cache,split) for split in ['train','valid']}
    cache_hashes={split:{f.name:digest(f) for f in sorted((a.cache/split).glob('*.npy'))}
                  for split in ['train','valid']}
    source={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}
    protocol=dict(matrix=matrix,manifest=meta,cache_sha256=cache_hashes,source_sha256=source,
                  train=dict(epochs=a.epochs,batch_size=128,seed=42,optimizer='AdamW',lr=.0003,
                             weight_decay=.01,warmup_fraction=.05,scheduler='cosine',clip=1.,
                             precision='bfloat16 autocast' if bf16 else 'float32'),
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                  notebook_reference='transformer-baseline@22d4de978546f1603a55f4997cb5f76bfca856a5/notebooks/Transformer 5-features.ipynb',
                  point_position_start=0,ffn_width_fixed=384,test_used=False,
                  selection='best validation top1; ties retain earliest epoch',
                  caveats=['Rebuilt local cache; cross-teammate exact ordered IDs not yet verified.',
                           'Point3 control uses a five-column projection with XY zeroed; total parameter count matches Geometry5.',
                           'Zero-XY weights receive no data gradient, so equal parameter count does not mean equal active features.',
                           'Depth/width change capacity. Width256 retains FF384, unlike the short-budget FF768 variant.',
                           'Same budget within this Transformer study, not an architecture-only comparison to Adam-trained LSTM.',
                           'One seed; no claim of statistical significance or guaranteed convergence.'])
    if (a.output/'protocol.json').exists():
        if json.loads((a.output/'protocol.json').read_text())!=protocol: raise ValueError('Resume protocol differs')
    else: save(a.output/'protocol.json',protocol)
    snapshot=a.output/'source'; snapshot.mkdir(exist_ok=True)
    for f in Path(__file__).parent.glob('*.py'): (snapshot/f.name).write_bytes(f.read_bytes())
    results=[]
    for config in matrix:
        folder=a.output/config['name']; folder.mkdir(exist_ok=True)
        if (folder/'metrics.json').exists():
            results.append(json.loads((folder/'metrics.json').read_text())); continue
        seed_all(42); model=Classifier(config).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        history=[]; step=0; best=-1.; start_epoch=1
        if (folder/'last.pt').exists():
            last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(last['model_state']); optimizer.load_state_dict(last['optimizer_state'])
            history=last['history']; step=last['step']; best=last['best']; start_epoch=last['epoch']+1
            torch.set_rng_state(last['torch_rng'])
            if device.type=='cuda': torch.cuda.set_rng_state_all(last['cuda_rng'])
            del last
        parameters=sum(x.numel() for x in model.parameters())
        print(json.dumps(dict(event='FORMAL_TRAINING_BEGIN',name=config['name'],train=len(data['train']),
                              valid=len(data['valid']),epochs=a.epochs,start_epoch=start_epoch,parameters=parameters,
                              device=str(device),precision=protocol['train']['precision'])),flush=True)
        for e in range(start_epoch,a.epochs+1):
            train,step=epoch(model,data['train'],device,128,bf16,optimizer,e,42,step,
                              a.epochs*math.ceil(len(data['train'])/128))
            valid,_=epoch(model,data['valid'],device,128,bf16,epoch_index=e)
            history.append(dict(epoch=e,train=train,valid=valid))
            payload=dict(model_state=model.state_dict(),config=config,classes=meta['categories'],
                         coordinate_scale=meta['coordinate_scale'],epoch=e,valid=valid,
                         protocol_sha256=digest(a.output/'protocol.json'))
            if valid['top1']>best:
                best=valid['top1']; checkpoint(folder/'best.pt',payload)
            checkpoint(folder/'last.pt',dict(payload,optimizer_state=optimizer.state_dict(),history=history,
                       step=step,best=best,torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []))
            save(folder/'history.json',history)
            save(a.output/'status.json',dict(phase='training',name=config['name'],completed_epoch=e,
                 epochs=a.epochs,best_validation=best,updated_utc=datetime.now(timezone.utc).isoformat()))
            print(json.dumps(dict(event='epoch_complete',name=config['name'],**history[-1])),flush=True)
        selected=torch.load(folder/'best.pt',map_location=device,weights_only=False)
        model.load_state_dict(selected['model_state'])
        checked,_=epoch(model,data['valid'],device,128,bf16)
        if checked['top1']!=selected['valid']['top1']: raise AssertionError('Reload validation differs')
        result=dict(name=config['name'],parameters=parameters,best_epoch=selected['epoch'],valid=checked,
                    train_valid_seconds=sum(r['train']['seconds']+r['valid']['seconds'] for r in history),
                    checkpoint_reload_verified=True,epochs=a.epochs,test_used=False)
        save(folder/'metrics.json',result); results.append(result); save(a.output/'summary.json',results)
        del selected,optimizer,model
        if device.type=='cuda': torch.cuda.empty_cache()
    save(a.output/'status.json',dict(phase='completed',configs=len(results),test_used=False,
         updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__': main()
