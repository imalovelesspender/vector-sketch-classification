"""Verify interrupted/resumed training matches an uninterrupted two-epoch run."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from . import run
from .data import CachedDataset


def main():
    root=Path.cwd().parents[1]
    cache=root/'outputs/geometry5_cache_20261118'
    with tempfile.TemporaryDirectory(prefix='anson_resume_check_') as tmp:
        temp=Path(tmp)
        # The new full-cache reader must preserve exactly the same samples.
        mem=temp/'mmap'/'valid'
        mem.mkdir(parents=True)
        original=CachedDataset(cache,'valid',5)
        for k in ['points','offsets','labels','ids']: np.save(mem/f'{k}.npy',getattr(original,k))
        mapped=CachedDataset(mem.parent,'valid',5)
        assert all(torch.equal(original[i][0],mapped[i][0]) and original[i][1]==mapped[i][1]
                   for i in [0,10,len(original)-1])
        for key in ['points','offsets','labels','ids']:
            getattr(mapped,key)._mmap.close()
        del mapped
        common=['--cache',str(cache),'--epochs','2','--device','cuda','--runs','transformer_geometry5_width64']
        argv=sys.argv
        try:
            sys.argv=['check',*common,'--output',str(temp/'continuous')]
            run.main()
            actual_epoch=run.epoch
            counter=0
            def interrupted(*args,**kwargs):
                nonlocal counter
                counter+=1
                if counter==3: raise RuntimeError('intentional checkpoint-resume check interruption')
                return actual_epoch(*args,**kwargs)
            run.epoch=interrupted
            sys.argv=['check',*common,'--output',str(temp/'resumed')]
            try:
                run.main()
                raise AssertionError('Expected an intentional interruption')
            except RuntimeError as e:
                if 'intentional checkpoint-resume' not in str(e): raise
            finally:
                run.epoch=actual_epoch
            sys.argv=['check',*common,'--output',str(temp/'resumed'),'--resume']
            run.main()
            a=torch.load(temp/'continuous/transformer_geometry5_width64/last.pt',map_location='cpu',weights_only=False)
            b=torch.load(temp/'resumed/transformer_geometry5_width64/last.pt',map_location='cpu',weights_only=False)
            assert all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in a['model_state'])
            for x,y in zip(a['history'],b['history']):
                assert x['train']==y['train'] and x['valid']==y['valid']
            print('PASS: memmap reader identity; resumed weights and metrics equal uninterrupted training',flush=True)
        finally:
            sys.argv=argv


if __name__=='__main__':main()
