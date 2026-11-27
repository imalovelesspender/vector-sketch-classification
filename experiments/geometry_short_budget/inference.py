"""Inference/embedding handoff for the completed Geometry5 screening.

Input must ALREADY have the experiment's stroke-3 preprocessing: bounding-box
scaling, then division of dx/dy by the stored full-training coordinate std.
Do not send raw canvas pixels to this interface.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.geometry_ablation.data import geometry5
from experiments.geometry_ablation.models import make_model


def load_candidate(checkpoint,device='cpu'):
    metadata=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if metadata.get('format_version')!=1:
        raise ValueError('Use a packaged inference.pt checkpoint')
    model=make_model(metadata['config'],len(metadata['classes']))
    model.load_state_dict(metadata['model_state'],strict=True)
    model.to(device).eval()
    return model,metadata


@torch.no_grad()
def extract_one(model,metadata,processed_stroke3):
    a=np.asarray(processed_stroke3,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=3 or not 1<=len(a)<=250:
        raise ValueError('Expected 1..250 valid points, shape [T,3]')
    if not np.isfinite(a).all() or not np.isin(a[:,2],[0,1]).all():
        raise ValueError('Non-finite coordinates or non-binary pen flags')
    x=geometry5(a,metadata['coordinate_scale'])
    device=next(model.parameters()).device
    z=model.embed(torch.from_numpy(x).unsqueeze(0).to(device),torch.tensor([len(x)]))
    logits=model.head(z)
    scores,indices=logits.softmax(-1).topk(min(5,len(metadata['classes'])))
    return dict(embedding=z[0].cpu().numpy(),
                top5=[dict(label=metadata['classes'][i],score=float(s))
                      for i,s in zip(indices[0].cpu().tolist(),scores[0].cpu().tolist())])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--processed-stroke3',required=True,help='Already normalized [T,3] .npy file')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    model,metadata=load_candidate(a.checkpoint,a.device)
    result=extract_one(model,metadata,np.load(a.processed_stroke3,allow_pickle=False))
    result['embedding']=result['embedding'].tolist()
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Saved {len(result["embedding"])}-dimensional embedding and top-5 model scores to {a.output}')


if __name__=='__main__':main()
