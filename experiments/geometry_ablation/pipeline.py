"""Continue the authorized full-data experiment and publish verified final records.

No test evaluation, shared-model edits, or main-branch merge is performed.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cache-process-id',type=int,required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--publish',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    workspace=root.parents[1]
    output=workspace/'outputs'
    state_dir=output/'geometry5_pipeline_20261118'
    state_dir.mkdir(exist_ok=True)
    cache=output/'geometry5_full_cache_20261118'
    results=output/'geometry5_full_20261118'
    branch='anson/stroke-experiments-20261118'
    def state(phase,**kwargs):
        data=dict(phase=phase,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
        temp=state_dir/'state.tmp'
        temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
        temp.replace(state_dir/'state.json')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=root,text=True,encoding='utf-8').strip()
    try:
        state('waiting_for_cache',cache_process_id=a.cache_process_id,epochs=a.epochs)
        if not (cache/'manifest.json').exists():
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
            kernel.CloseHandle.argtypes=[wintypes.HANDLE]
            handle=kernel.OpenProcess(0x100000,False,a.cache_process_id)
            if not handle: raise RuntimeError('Cache preparation process is not running; manifest is absent')
            try:
                deadline=time.monotonic()+6*60*60
                while True:
                    wait=kernel.WaitForSingleObject(handle,30000)
                    if wait==0: break
                    if wait!=258 or time.monotonic()>deadline:
                        raise RuntimeError(f'Cache wait failed or timed out: {wait}')
                    progress=cache/'normalization_progress.json'
                    n=len(json.loads(progress.read_text())) if progress.exists() else 0
                    state('waiting_for_cache',cache_process_id=a.cache_process_id,
                          epochs=a.epochs,normalization_categories=n)
            finally:
                kernel.CloseHandle(handle)
        meta=json.loads((cache/'manifest.json').read_text())
        if len(meta['categories'])!=345 or meta['train_per_class']!=5000 or meta['valid_per_class']!=500:
            raise RuntimeError('Full-data manifest does not match the agreed 345-class setup')
        if git('branch','--show-current')!=branch:
            raise RuntimeError('Worktree is on an unexpected branch')
        cmd=[sys.executable,'-u','-m','experiments.geometry_ablation.run','--cache',str(cache),
             '--output',str(results),'--epochs',str(a.epochs),'--device','cuda']
        if (results/'protocol.json').exists(): cmd.append('--resume')
        with (output/'geometry5_full_training_20261118.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            state('formal_training_process_started',training_pid=child.pid,epochs=a.epochs,
                  note='Confirm BEGIN/epoch records in training log before reporting actual training progress')
            code=child.wait()
        if code!=0: raise RuntimeError(f'Full training process exited with code {code}; see training log')
        rows=json.loads((results/'summary.json').read_text())
        if len(rows)!=14 or not all(r['checkpoint_reload_verified'] and 0<=r['valid_top1']<=1 for r in rows):
            raise RuntimeError('Expected 14 completed, validated result records')
        target=root/'docs/experiments/geometry5_345_20261118'
        target.mkdir(parents=True,exist_ok=True)
        for name in ['summary.csv','summary.json','protocol.json']:
            shutil.copy2(results/name,target/name)
        for row in rows:
            shutil.copy2(results/row['name']/'history.json',target/f"{row['name']}_history.json")
        best=max(rows,key=lambda r:r['valid_top1'])
        readme=(f'# Geometry input and architecture ablations: 345 classes\n\n'
                f'All 14 configurations completed {a.epochs} epochs on the same rebuilt full-archive '
                '1,725,000 training and 172,500 validation sketches. Models were selected on validation only. '
                '**No test-set evaluation was performed.**\n\n'
                f"Highest observed validation top-1: {best['valid_top1']:.2%} ({best['name']}). "
                'This is a single-seed result, not a significance claim. See summary.csv and per-run histories.\n\n'
                'Geometry5 adds centered absolute XY to dx, dy and pen state. Tokens remain points; '
                'this is not whole-stroke tokenization or the XAI shape/location/order encoder. '
                'Depth and width change model capacity. Transformer heads are varied at fixed width.\n\n'
                'Preprocessing uses the historical Sihui notebook source and adapter sampling rule, '
                'but the old processed-manifest/sample-ID artifacts were unavailable for independent '
                'comparison. Treat historical report numbers separately.\n\n'
                'Local checkpoint binaries and raw data are not committed. The complete protocol, '
                'source hashes, cache hashes and class order are in protocol.json.\n')
        (target/'README.md').write_text(readme,encoding='utf-8')
        status_file=root/'docs/ANSON_EXPERIMENT_STATUS.md'
        status_text=status_file.read_text(encoding='utf-8')
        banner=('# Update: full experiments completed\n\n'
                f'All 14 configurations completed {a.epochs} epochs. Final validation results: '
                '[geometry5_345_20261118](experiments/geometry5_345_20261118/README.md). '
                'No final test evaluation was performed. The notes below preserve the earlier development record.\n\n')
        status_file.write_text(banner+status_text,encoding='utf-8')
        state('completed_locally',best_validation=best['valid_top1'],best_config=best['name'])
        if a.publish:
            if git('diff','--cached','--name-only'):
                raise RuntimeError('Other staged changes exist; refusing to include unrelated work')
            git('add','--',str(target.relative_to(root)),str(status_file.relative_to(root)))
            git('diff','--cached','--check')
            git('commit','-m','Archive completed 345-class geometry-input and architecture ablations')
            subprocess.run(['git','push','-u','origin',branch],cwd=root,check=True)
            local=git('rev-parse','HEAD')
            remote=git('ls-remote','--heads','origin',f'refs/heads/{branch}').split()[0]
            if local!=remote: raise RuntimeError('Remote commit verification failed')
            state('completed_and_pushed',commit=local,
                  github=f'https://github.com/imalovelesspender/vector-sketch-classification/tree/{branch}')
    except Exception as e:
        state('failed',error=str(e))
        raise


if __name__=='__main__':main()
