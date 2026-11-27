"""Audit completed records and package small, verified inference checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from experiments.geometry_ablation.data import CachedDataset,collate
from experiments.geometry_ablation.models import make_model
from .inference import load_candidate,extract_one


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    workspace=root.parents[1]
    source=workspace/'outputs/geometry5_short_20261124'
    cache=workspace/'outputs/geometry5_short_cache_20261124'
    rows=json.loads((source/'summary.json').read_text())
    protocol=json.loads((source/'protocol.json').read_text())
    expected={x['name'] for x in protocol['matrix']}
    assert len(rows)==14 and {x['name'] for x in rows}==expected
    audit=dict(completed_configs=14,epochs_per_config=5,train_samples=345000,valid_samples=34500,
               test_used=False,candidate_checks=[])
    for row in rows:
        history=json.loads((source/row['name']/'history.json').read_text())
        assert len(history)==5 and row['checkpoint_reload_verified']
        assert all(x['train']['samples']==345000 and x['valid']['samples']==34500 for x in history)
        assert row['valid_top1']==max(x['valid']['top1'] for x in history)
        assert (source/row['name']/'best.pt').is_file()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    bundle=out/'geometry5_mae_handoff';bundle.mkdir()
    selected=['lstm_geometry5_base','lstm_geometry5_depth2','lstm_geometry5_depth3',
              'transformer_geometry5_base','transformer_geometry5_depth6']
    ds=CachedDataset(cache,'valid',5)
    sample_indices=np.linspace(0,len(ds)-1,6,dtype=int).tolist()
    x,lengths,_=collate([ds[i] for i in sample_indices])
    torch.set_num_threads(2)
    lookup={row['name']:row for row in rows}
    for name in selected:
        original_path=source/name/'best.pt'
        original=torch.load(original_path,map_location='cpu',weights_only=False)
        assert original['valid']['top1']==lookup[name]['valid_top1']
        compact=dict(format_version=1,model_state=original['model_state'],config=original['config'],
            classes=original['classes'],epoch=original['epoch'],validation=original['valid'],
            coordinate_scale=protocol['manifest']['coordinate_scale'],
            input_columns=['dx','dy','pen_up','centered_x','centered_y'],max_seq_len=250,
            scope=protocol['scope'],source_checkpoint_sha256=sha(original_path))
        target=bundle/'checkpoints'/name/'inference.pt';target.parent.mkdir(parents=True)
        torch.save(compact,target)
        reference=make_model(original['config'],345).eval()
        reference.load_state_dict(original['model_state'],strict=True)
        packed,metadata=load_candidate(target)
        with torch.no_grad():
            z=reference.embed(x,lengths);logits=reference.head(z)
            actual_z=packed.embed(x,lengths);actual_logits=packed.head(actual_z)
        assert torch.allclose(logits,actual_logits,atol=1e-6,rtol=1e-5)
        assert torch.allclose(z,actual_z,atol=1e-6,rtol=1e-5)
        # Verify the public handoff entry point, not only raw state_dict loading.
        single=extract_one(packed,metadata,ds[sample_indices[0]][0].numpy()[:,:3])
        assert np.allclose(single['embedding'],actual_z[0].numpy(),atol=2e-5,rtol=1e-4)
        audit['candidate_checks'].append(dict(name=name,embedding_dim=z.shape[1],epoch=original['epoch'],
            best_validation=lookup[name]['valid_top1'],prediction_and_embedding_reload_verified=True,
            input_interface_verified=True,inference_checkpoint_sha256=sha(target),bytes=target.stat().st_size))
        print('VERIFIED',name,z.shape[1],flush=True)
        del reference,packed,original,compact,metadata
    for relative in ['experiments/geometry_ablation/__init__.py','experiments/geometry_ablation/models.py',
                     'experiments/geometry_ablation/data.py','experiments/geometry_short_budget/__init__.py',
                     'experiments/geometry_short_budget/inference.py']:
        dest=bundle/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/relative,dest)
    shutil.copy2(source/'protocol.json',bundle/'protocol.json')
    shutil.copy2(source/'summary.csv',bundle/'summary.csv')
    (bundle/'requirements.txt').write_text('numpy==2.3.5\ntorch==2.11.0\n',encoding='utf-8')
    readme='''# Geometry5 handoff to Mae

Five inference-only candidates are included for discussion, not a final model decision.
BiLSTM base/depth2/depth3 return 512-D embeddings. Transformer base/depth6 return
128-D embeddings. All classify 345 categories in the checkpoint's `classes` order.

These are the completed November 24 short-budget models: 1,000 train and 100
validation sketches per class, five epochs, seed 42. They are not the historical
20-epoch models. Validation scores are not test scores or retrieval metrics.

Input is normalized stroke-3 [T,3]: dx,dy,pen_up; lengths must be 1..250. Apply the
same per-sketch bounding-box scaling and stored full-training std exactly once.
The checkpoint records coordinate_scale. Raw canvas pixels are not accepted by
this inference helper. Centered XY is reconstructed as cumsum(normalized dx/dy)
times coordinate_scale, then centered on the sketch's mean point. Pen flags and
drawing order stay unchanged. Tokens remain POINTS; this is not an XAI hierarchy.

Tested on Python 3.12, PyTorch 2.11.0+cu128 and NumPy 2.3.5; CPU inference was
verified on six mixed-length validation sketches per candidate. Optimizer states
were stripped. Weights and class/config/preprocessing metadata were retained.

From this extracted folder:

python -m experiments.geometry_short_budget.inference --checkpoint checkpoints/transformer_geometry5_depth6/inference.pt --processed-stroke3 your_normalized_sketch.npy --output prediction.json

The output contains an embedding and top-five softmax model scores. These scores
are not calibrated confidence. For retrieval, extract reference and query vectors
with the SAME chosen checkpoint and preprocessing. Mae should build the reference
index from training sketches only; validation/test must not enter that index.
Any L2 normalization/cosine search is a subsequent backend choice, not a measured
retrieval result of this package.

See checksums.json and handoff_audit.json for integrity and packaging checks.
Raw datasets and API credentials are not included. The team still needs to agree
on the final checkpoint and frontend/backend request and response formats.
'''
    (bundle/'README.md').write_text(readme,encoding='utf-8')
    (bundle/'handoff_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    checksums={str(f.relative_to(bundle)).replace('\\','/'):sha(f) for f in sorted(bundle.rglob('*')) if f.is_file()}
    (bundle/'checksums.json').write_text(json.dumps(checksums,indent=2),encoding='utf-8')
    zip_path=out/'geometry5_mae_handoff.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(bundle.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
    with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
    (out/'handoff_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('PACKAGE READY',zip_path,'bytes',zip_path.stat().st_size,flush=True)


if __name__=='__main__':main()
