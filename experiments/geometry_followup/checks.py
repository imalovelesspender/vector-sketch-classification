"""Check implementation against reference, input controls, masking and resume."""
import ast
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from .models import Classifier, configurations
from .run import Split, epoch, seed_all, save


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs=configurations(); counts={}
    for config in configs:
        seed_all(42); model=Classifier(config).to(device)
        counts[config['name']]=sum(p.numel() for p in model.parameters())
        x=torch.randn(3,9,5,device=device); lens=torch.tensor([9,5,1],device=device)
        model.eval()
        with torch.no_grad():
            a=model(x,lens); padded=F.pad(x,(0,0,0,5),value=777)
            torch.testing.assert_close(a,model(padded,lens),atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(a[1:2],model(x[1:2,:5],lens[1:2]),atol=3e-5,rtol=3e-5)
            changed=x.clone(); changed[:,:,3:]+=torch.randn_like(changed[:,:,3:])*10
            if config['zero_xy']: torch.testing.assert_close(a,model(changed,lens),atol=0,rtol=0)
            else: assert not torch.allclose(a,model(changed,lens))
            assert model(torch.zeros(1,250,5,device=device),torch.tensor([250],device=device)).shape==(1,345)
            packed=io.BytesIO(); torch.save(model.state_dict(),packed); packed.seek(0)
            clone=Classifier(config).to(device).eval()
            clone.load_state_dict(torch.load(packed,map_location=device,weights_only=True))
            torch.testing.assert_close(a,clone(x,lens),atol=0,rtol=0)
        for bad in [torch.tensor([0,5,1],device=device),torch.tensor([10,5,1],device=device)]:
            try: model(x,bad)
            except ValueError: pass
            else: raise AssertionError('Invalid length accepted')
        model.train(); F.cross_entropy(model(x,lens),torch.arange(3,device=device)).backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        del model,clone
    assert counts[configs[0]['name']]==counts[configs[1]['name']]==counts[configs[4]['name']]==707161
    # Execute only class declarations from the pinned notebook, no W&B or data cells.
    notebook=json.loads(subprocess.check_output(['git','show',
        '22d4de978546f1603a55f4997cb5f76bfca856a5:notebooks/Transformer 5-features.ipynb'],text=True,encoding='utf-8'))
    source=next(''.join(c['source']) for c in notebook['cells'] if 'class AttentionBlock' in ''.join(c['source']))
    tree=ast.parse(source)
    declarations=ast.Module(body=[n for n in tree.body if isinstance(n,ast.ClassDef)],type_ignores=[])
    env=dict(torch=torch,nn=nn,F=F,math=math)
    exec(compile(declarations,'pinned_reference_classes','exec'),env)
    reference=env['SketchClassifier']().to(device).eval()
    model=Classifier(configs[1]).to(device).eval()
    reference_state={('head.'+k[5:] if k.startswith('head.') else 'encoder.'+k):v
                     for k,v in model.state_dict().items()}
    reference.load_state_dict(reference_state,strict=True)
    x=torch.randn(3,17,5,device=device); lens=torch.tensor([17,9,1],device=device)
    with torch.no_grad():
        torch.testing.assert_close(model(x,lens),reference(x,lens),atol=0,rtol=0)
        if device.type=='cuda' and torch.cuda.is_bf16_supported():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                torch.testing.assert_close(model(x,lens),reference(x,lens),atol=0,rtol=0)
    # Verify geometry and gather batching on actual cache sketches.
    workspace=Path(__file__).resolve().parents[4]
    cache=workspace/'outputs/geometry5_full_cache_20261118'
    meta=json.loads((cache/'manifest.json').read_text())
    data=Split(cache,'valid',audit=False)
    indices=np.array([0,501,73001,len(data)-1])
    batch,lengths,labels=data.batch(indices,torch.device('cpu'))
    for j,i in enumerate(indices):
        a=np.array(data.points[data.offsets[i]:data.offsets[i+1]])
        xy=np.cumsum(a[:,:2],axis=0)*meta['coordinate_scale']; xy-=xy.mean(axis=0)
        np.testing.assert_allclose(a[:,3:],xy,atol=2e-6,rtol=2e-6)
        np.testing.assert_array_equal(batch[j,:len(a)].numpy(),a)
    # Exact CPU interrupted/resumed optimizer+RNG behavior, including dropout and LR schedule.
    tiny=copy.copy(data)
    tiny.labels=np.asarray(data.labels[:8]).copy()
    tiny.offsets=np.asarray(data.offsets[:9]).copy()
    tiny.lengths=np.diff(tiny.offsets)
    config=dict(configs[1],width=16,depth=1,heads=2,ff_dim=32)
    cpu=torch.device('cpu'); seed_all(123)
    first=Classifier(config); opt=torch.optim.AdamW(first.parameters(),lr=3e-4,weight_decay=.01)
    _,step=epoch(first,tiny,cpu,4,False,opt,1,42,0,4,0)
    buffer=io.BytesIO(); torch.save(dict(model=first.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),buffer)
    epoch(first,tiny,cpu,4,False,opt,2,42,step,4,0)
    buffer.seek(0); saved=torch.load(buffer,weights_only=False)
    resumed=Classifier(config); opt2=torch.optim.AdamW(resumed.parameters(),lr=3e-4,weight_decay=.01)
    resumed.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    epoch(resumed,tiny,cpu,4,False,opt2,2,42,step,4,0)
    assert all(torch.equal(v,resumed.state_dict()[k]) for k,v in first.state_dict().items())
    report=dict(passed=True,device=str(device),parameters=counts,
                checks=['mask/padding/mixed-length invariance','XY control counterfactual','finite gradients',
                        'strict checkpoint reload','reference float32 and supported bfloat16 exact outputs',
                        'real-cache geometry and batch identity','optimizer+dropout+schedule resume identity'])
    out=workspace/'outputs/geometry5_followup_checks_20261127.json';save(out,report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
