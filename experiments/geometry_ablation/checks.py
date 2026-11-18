"""Meaningful masking, geometry, gradients and reload checks before training."""
import copy
import json

import numpy as np
import torch

from .data import geometry5
from .models import configurations, make_model


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.mha.set_fastpath_enabled(False)
    a=np.array([[1,0,0],[0,1,1],[2,0,1]],dtype=np.float32)
    g=geometry5(a,.5)
    assert np.array_equal(g[:,:3],a)
    assert np.allclose(g[:,3:].mean(0),0,atol=1e-7)
    assert np.allclose(np.diff(g[:,3:],axis=0),a[1:,:2]*.5)
    counts={}
    for cfg in configurations():
        model=make_model(cfg,7).to(dev)
        x=torch.randn(3,13,cfg['input_dim'],device=dev)
        lengths=torch.tensor([1,7,13])
        model.eval()
        with torch.no_grad():
            y=model(x,lengths)
            z=x.clone();z[0,1:]=100;z[1,7:]=-100
            assert torch.allclose(y,model(z,lengths),atol=2e-5,rtol=1e-4),cfg['name']
            extra=torch.cat([x,torch.randn(3,4,cfg['input_dim'],device=dev)],1)
            assert torch.allclose(y,model(extra,lengths),atol=2e-5,rtol=1e-4),cfg['name']
            alone=model(x[:1,:1],lengths[:1])
            assert torch.allclose(y[:1],alone,atol=2e-5,rtol=1e-4),cfg['name']
            clone=make_model(cfg,7).to(dev)
            clone.load_state_dict(copy.deepcopy(model.state_dict()))
            clone.eval()
            assert torch.allclose(y,clone(x,lengths),atol=1e-6)
        model.train()
        loss=torch.nn.functional.cross_entropy(model(x,lengths),torch.tensor([0,1,2],device=dev))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        try:
            model(x,torch.tensor([0,7,13]))
            raise AssertionError('Invalid lengths accepted')
        except ValueError:
            pass
        counts[cfg['name']]=sum(p.numel() for p in make_model(cfg,345).parameters())
    assert len({counts[k] for k in counts if k.startswith('transformer_geometry5_heads') or k=='transformer_geometry5_base'})==1
    print(json.dumps(dict(device=str(dev),geometry_verified=True,masking_verified=True,
                          finite_gradients=True,reload_verified=True,parameters345=counts),indent=2))


if __name__ == '__main__':
    main()
