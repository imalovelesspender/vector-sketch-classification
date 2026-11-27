"""User-approved reduced queue; keep the in-flight depth6 run, then heads8 only.

Lives outside geometry_followup so its frozen training-source hashes stay intact.
The original protocol/results are retained; a separately derived protocol records
the reduced configuration list. No optimizer, schedule, dataset or model changes.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[1]
sys.path.insert(0,str(WORKSPACE/'work/stroke-gpu-runtime'))
os.environ['PYTHONPATH']=str(WORKSPACE/'work/stroke-gpu-runtime')
from .geometry_followup import worker as shared
from .geometry_followup.run import digest,save

ORIGINAL=WORKSPACE/'outputs/geometry5_transformer_followup_20261127'
REDUCED=WORKSPACE/'outputs/geometry5_transformer_reduced_20261127'
NAMES=['transformer_point3_zero_xy_control','transformer_geometry5_base',
       'transformer_geometry5_depth6','transformer_geometry5_heads8']
DEFERRED=['transformer_geometry5_width256','lstm_geometry5_depth2','lstm_geometry5_depth3']
KERNEL=ctypes.WinDLL('kernel32',use_last_error=True)
KERNEL.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
KERNEL.OpenProcess.restype=wintypes.HANDLE
KERNEL.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
KERNEL.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]
KERNEL.CloseHandle.argtypes=[wintypes.HANDLE]


def alive(handle):
    code=wintypes.DWORD()
    return bool(handle and KERNEL.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259


def derive():
    protocol=json.loads((ORIGINAL/'protocol.json').read_text())
    if protocol['train']['epochs']!=15: raise ValueError('Original budget differs')
    reduced=dict(protocol,matrix=[c for c in protocol['matrix'] if c['name'] in NAMES])
    if [c['name'] for c in reduced['matrix']]!=NAMES: raise ValueError('Unexpected configuration list')
    for name,expected in protocol['source_sha256'].items():
        if digest(ROOT/'experiments/geometry_followup'/name)!=expected:
            raise ValueError('Frozen training source changed: '+name)
    REDUCED.mkdir(exist_ok=True)
    if (REDUCED/'protocol.json').exists():
        if json.loads((REDUCED/'protocol.json').read_text())!=reduced:
            raise ValueError('Reduced protocol differs; refusing overwrite')
    else: save(REDUCED/'protocol.json',reduced)
    inherited={}
    for name in NAMES[:3]:
        source=ORIGINAL/name
        row=json.loads((source/'metrics.json').read_text())
        history=json.loads((source/'history.json').read_text())
        if len(history)!=15 or not row['checkpoint_reload_verified']:
            raise ValueError('Cannot inherit unfinished/unverified run: '+name)
        if row['valid']['top1']!=max(h['valid']['top1'] for h in history):
            raise ValueError('Inherited best-validation mismatch')
        target=REDUCED/name;target.mkdir(exist_ok=True)
        inherited[name]={}
        for filename in ['metrics.json','history.json','best.pt','last.pt']:
            original=source/filename; copied=target/filename
            if not copied.exists(): shutil.copy2(original,copied)
            original_hash=digest(original)
            if digest(copied)!=original_hash: raise ValueError('Inherited file copy differs')
            inherited[name][filename]=original_hash
    save(REDUCED/'derivation.json',dict(reason='User approved simpler queue on November 27',
         parent_protocol=str(ORIGINAL/'protocol.json'),parent_protocol_sha256=digest(ORIGINAL/'protocol.json'),
         retained=NAMES,deferred=DEFERRED,inherited_files_sha256=inherited,
         epoch_budget_unchanged=15,model_and_optimizer_unchanged=True,
         original_checkpoint_protocol_hashes_preserved=True))


def publish():
    rows=json.loads((REDUCED/'summary.json').read_text())
    if [r['name'] for r in rows]!=NAMES: raise ValueError('Reduced result set incomplete')
    target=ROOT/'docs/experiments/geometry5_reduced_20261127';target.mkdir(parents=True,exist_ok=True)
    for row in rows:
        history=json.loads((REDUCED/row['name']/'history.json').read_text())
        if len(history)!=15 or not row['checkpoint_reload_verified']:
            raise ValueError('Unverified reduced result')
        if any(h['train']['samples']!=1725000 or h['valid']['samples']!=172500 for h in history):
            raise ValueError('Reduced sample counts differ')
        if row['valid']['top1']!=max(h['valid']['top1'] for h in history):
            raise ValueError('Reduced best-validation mismatch')
        shutil.copy2(REDUCED/row['name']/'history.json',target/(row['name']+'_history.json'))
    for filename in ['summary.json','protocol.json','derivation.json']:
        shutil.copy2(REDUCED/filename,target/filename)
    lines=['# Geometry5 reduced full-data follow-up','',
           'User-approved simpler queue: retain Point3 control, Geometry5 base, depth6 and heads8. '
           'Defer width256 and both deeper LSTMs. All four retained configurations use 15 epochs, '
           'the same 345-class rebuilt data (1,725,000 train / 172,500 validation), seed42, '
           'AdamW 0.0003 with 5% warmup + cosine decay, batch128 and BF16 autocast. No test evaluation.','',
           '| Configuration | Best validation top-1 | Best epoch | Parameters |',
           '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['valid']['top1']:.2%} | {r['best_epoch']} | {r['parameters']:,} |")
    lookup={r['name']:r for r in rows};base=lookup[NAMES[1]]['valid']['top1']
    lines+=['','## 给组员的结论','',
            f"- 加入中心化位置特征后，准确率由 {lookup[NAMES[0]]['valid']['top1']:.2%} 变为 {base:.2%}，变化 {(base-lookup[NAMES[0]]['valid']['top1'])*100:+.2f} 个百分点。",
            f"- 深度从4层增加到6层：变化 {(lookup[NAMES[2]]['valid']['top1']-base)*100:+.2f} 个百分点，同时参数量增加。",
            f"- heads从4变8：变化 {(lookup[NAMES[3]]['valid']['top1']-base)*100:+.2f} 个百分点，总参数量相同。",
            '- Point3对照将XY两列置零，保持模型总参数量一致；并非历史的三输入投影模型。',
            '- 这是单个seed、固定15轮预算下的验证结果。组员历史实验样本一致性尚未确认，不直接横向排名。',
            '- 本轮没有补跑宽度或深层LSTM，不对这几项作完整训练后的结论。']
    (target/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if shared.git('branch','--show-current')!=shared.BRANCH: raise RuntimeError('Unexpected branch')
    if shared.git('diff','--cached','--name-only'): raise RuntimeError('Other staged changes exist')
    shared.git('add','--',str(target.relative_to(ROOT)));shared.git('diff','--cached','--check')
    shared.git('commit','-m','Archive reduced full-data Geometry5 follow-up results')
    shared.git('push','origin',shared.BRANCH)
    commit=shared.git('rev-parse','HEAD')
    if shared.git('ls-remote','--heads','origin','refs/heads/'+shared.BRANCH).split()[0]!=commit:
        raise RuntimeError('Remote publication SHA differs')
    shared.state('completed_and_pushed',scope='user-approved reduced queue',commit=commit,results=str(target))


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-training-pid',type=int,default=6576)
    a=p.parse_args()
    import msvcrt
    lock=(WORKSPACE/'outputs/geometry5_followup_20261127.lock').open('a+b')
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    handle=None
    try:
        depth=ORIGINAL/NAMES[2]/'metrics.json'
        if not depth.exists():
            # This handle remains bound to the original process, even if its PID is reused later.
            handle=KERNEL.OpenProcess(0x100000|0x1000|1,False,a.original_training_pid)
            if not alive(handle): raise RuntimeError('In-flight depth6 process is absent; checkpoint resume needed')
            while not depth.exists():
                if not alive(handle): raise RuntimeError('In-flight training stopped before depth6 completed')
                shared.state('running',scope='reduced queue: finishing depth6, then heads8 only',
                             training_pid=a.original_training_pid,output=str(ORIGINAL),deferred=DEFERRED)
                time.sleep(5)
            if alive(handle):
                # Depth6 has completed reload verification; stop before the deferred width run proceeds.
                if not KERNEL.TerminateProcess(handle,0): raise ctypes.WinError(ctypes.get_last_error())
                for _ in range(50):
                    if not alive(handle): break
                    time.sleep(.1)
                if alive(handle): raise RuntimeError('Original training did not stop')
        shared.state('preparing_reduced_results',deferred=DEFERRED);derive()
        shared.execute('experiments.geometry_followup.run',REDUCED,15,NAMES)
        shared.state('checking_and_publishing',scope='reduced queue');publish()
    except Exception as error:
        shared.state('failed',scope='reduced queue',error=str(error))
        (WORKSPACE/'outputs/geometry5_reduced_worker_error_20261127.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        if handle: KERNEL.CloseHandle(handle)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':main()
