"""Independent Windows worker: formal experiments, audit, archive own branch."""
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
import traceback

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parents[1]
RUNTIME=WORKSPACE/'work/stroke-gpu-runtime'
if RUNTIME.is_dir():
    sys.path.insert(0,str(RUNTIME))
    os.environ['PYTHONPATH']=str(RUNTIME)
from .run import save, digest

OUT=WORKSPACE/'outputs'
CACHE=OUT/'geometry5_full_cache_20261118'
TRANSFORMER=OUT/'geometry5_transformer_followup_20261127'
LSTM=OUT/'geometry5_lstm_followup_20261127'
LOG=OUT/'geometry5_followup_training_20261127.log'
STATE=OUT/'geometry5_followup_worker_20261127.json'
BRANCH='anson/stroke-experiments-20261118'
ENV=os.environ.copy()
ENV.update(PYTHONPATH=str(WORKSPACE/'work/stroke-gpu-runtime'),PYTHONIOENCODING='utf-8',
           GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
ENV['PATH']='C:\\Program Files\\Git\\cmd;'+ENV.get('PATH','')


class Power(ctypes.Structure):
    _fields_=[('ac',wintypes.BYTE),('flag',wintypes.BYTE),('percent',wintypes.BYTE),
              ('reserved',wintypes.BYTE),('seconds',wintypes.DWORD),('full',wintypes.DWORD)]


def state(phase,**fields):
    power=Power(); kernel=ctypes.windll.kernel32
    kernel.SetThreadExecutionState.argtypes=[wintypes.DWORD]
    kernel.SetThreadExecutionState.restype=wintypes.DWORD
    ac=bool(kernel.GetSystemPowerStatus(ctypes.byref(power))) and power.ac==1
    kernel.SetThreadExecutionState(0x80000001 if ac else 0x80000000)
    payload=dict(phase=phase,worker_pid=os.getpid(),on_ac=ac,
                 updated_utc=datetime.now(timezone.utc).isoformat(),**fields)
    save(STATE,payload); save(OUT/'current_training_status.json',payload)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=ENV,text=True,encoding='utf-8').strip()


def audit_previous_lstm():
    previous=OUT/'geometry5_full_20261118'
    protocol=json.loads((previous/'protocol.json').read_text())
    for file in ['models.py','data.py','run.py']:
        if digest(ROOT/'experiments/geometry_ablation'/file)!=protocol['code_sha256'][file]:
            raise ValueError('Historical full LSTM implementation differs: '+file)
    for split,files in protocol['cache_sha256'].items():
        for filename,expected in files.items():
            if digest(CACHE/split/filename)!=expected:
                raise ValueError('Historical full LSTM cache differs: '+split+'/'+filename)
    rows=json.loads((previous/'summary.json').read_text())
    selected=[]
    for name in ['lstm_point3_control','lstm_geometry5_base']:
        row=next(r for r in rows if r['name']==name)
        history=json.loads((previous/name/'history.json').read_text())
        if len(history)!=20 or not row['checkpoint_reload_verified']:
            raise ValueError('Existing full-budget LSTM control incomplete')
        if any(h['train']['samples']!=1725000 or h['valid']['samples']!=172500 for h in history):
            raise ValueError('Existing full-budget LSTM counts differ')
        if row['valid_top1']!=max(h['valid']['top1'] for h in history):
            raise ValueError('Historical LSTM summary differs from history')
        selected.append(dict(row,source_folder=str(previous/name),source='reused completed 20-epoch full-data run'))
    save(OUT/'geometry5_reused_lstm_controls_20261127.json',dict(verified=True,rows=selected,
         implementation_hashes_verified=True,full_cache_hashes_verified=True,test_used=False))


def execute(module, output, epochs, names=None):
    args=[sys.executable,'-u','-m',module,'--cache',str(CACHE),'--output',str(output),
          '--epochs',str(epochs),'--device','cuda']
    if names: args+=['--runs',*names]
    if module.endswith('geometry_ablation.run') and (output/'protocol.json').exists(): args+=['--resume']
    startup=subprocess.STARTUPINFO(); startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow=0
    with LOG.open('a',encoding='utf-8') as log:
        log.write('\nStarting '+module+'\n');log.flush()
        process=subprocess.Popen(args,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
        while process.poll() is None:
            state('running',module=module,training_pid=process.pid,output=str(output),epochs=epochs)
            time.sleep(15)
    if process.returncode: raise RuntimeError(f'{module} exited {process.returncode}; see {LOG.name}')


def publish():
    target=ROOT/'docs/experiments/geometry5_followup_20261127'
    target.mkdir(parents=True,exist_ok=True)
    rows_t=json.loads((TRANSFORMER/'summary.json').read_text())
    rows_l=json.loads((LSTM/'summary.json').read_text())
    if len(rows_t)!=5 or len(rows_l)!=2: raise ValueError('Follow-up configurations incomplete')
    for folder,rows,epochs in [(TRANSFORMER,rows_t,15),(LSTM,rows_l,20)]:
        for row in rows:
            h=json.loads((folder/row['name']/'history.json').read_text())
            if len(h)!=epochs or not row['checkpoint_reload_verified']:
                raise ValueError('Unverified checkpoint/history')
            if any(x['train']['samples']!=1725000 or x['valid']['samples']!=172500 for x in h):
                raise ValueError('Incomplete epoch counts')
            score=row['valid']['top1'] if folder==TRANSFORMER else row['valid_top1']
            if score!=max(x['valid']['top1'] for x in h): raise ValueError('Best checkpoint mismatch')
        sub=target/('transformer_15epochs' if folder==TRANSFORMER else 'lstm_20epochs')
        sub.mkdir(exist_ok=True)
        for name in ['summary.json','protocol.json']:
            shutil.copy2(folder/name,sub/name)
        for row in rows: shutil.copy2(folder/row['name']/'history.json',sub/(row['name']+'_history.json'))
    shutil.copy2(OUT/'geometry5_reused_lstm_controls_20261127.json',target/'reused_lstm_controls.json')
    shutil.copy2(OUT/'geometry5_followup_checks_20261127.json',target/'checks.json')
    lines=['# Geometry5 full-data follow-up results','',
           'All 345 categories: 5,000 train / 500 validation per class (1,725,000 / 172,500).',
           'No test-set evaluation; a single seed, 42. Rebuilt samples are identical within this study; '
           'cross-teammate sample identity remains unverified.','',
           '## Transformer: 15 epochs, AdamW 0.0003, 5% warmup + cosine decay','',
           '| Configuration | Best validation top-1 | Best epoch | Parameters | Train + validation seconds |',
           '|---|---:|---:|---:|---:|']
    for r in rows_t:
        lines.append(f"| {r['name']} | {r['valid']['top1']:.2%} | {r['best_epoch']} | {r['parameters']:,} | {r['train_valid_seconds']:.1f} |")
    lines+=['','The Point3 control uses zero XY columns in the same five-input projection. '
            'Base and heads8 have the same total parameter count. FFN width stays 384 when model width changes. '
            'Depth/width comparisons include capacity changes.','',
            '## LSTM: 20 epochs, Adam 0.001','',
            '| Configuration | Best validation top-1 | Best epoch | Parameters | Source |',
            '|---|---:|---:|---:|---|']
    reused=json.loads((target/'reused_lstm_controls.json').read_text())['rows']
    for r in [*reused,*rows_l]:
        source='verified prior full-data control' if r in reused else 'new full-data follow-up'
        lines.append(f"| {r['name']} | {r['valid_top1']:.2%} | {r['best_epoch']} | {r['parameters_actual']:,} | {source} |")
    delta=(rows_t[1]['valid']['top1']-rows_t[0]['valid']['top1'])*100
    best_t=max(rows_t,key=lambda r:r['valid']['top1']);best_l=max([*reused,*rows_l],key=lambda r:r['valid_top1'])
    lines+=['','## 给组员的结论','',
            f"- 相同参数总量、相同数据与训练预算下，Transformer 加入位置特征后的验证准确率变化为 {delta:+.2f} 个百分点。",
            f"- 本轮 Transformer 候选中验证表现最高的是 {best_t['name']}：{best_t['valid']['top1']:.2%}。",
            f"- 完整 20 轮 LSTM 候选中最高的是 {best_l['name']}：{best_l['valid_top1']:.2%}。",
            '- 两个架构的优化器、轮数和精度不同，因此跨架构比较只能解释为当前训练方案的表现。',
            '- 不能据此断言已收敛、具有统计显著性，或检索效果最好；最终选择需与 Mae 核对样本、验证协议和 demo 需求。']
    (target/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if git('branch','--show-current')!=BRANCH: raise RuntimeError('Unexpected publication branch')
    if git('diff','--cached','--name-only'): raise RuntimeError('Unrelated staged changes; refusing publication')
    git('add','--',str(target.relative_to(ROOT)))
    git('diff','--cached','--check')
    git('commit','-m','Archive verified full-data Geometry5 follow-up results')
    git('push','origin',BRANCH)
    commit=git('rev-parse','HEAD')
    if git('ls-remote','--heads','origin','refs/heads/'+BRANCH).split()[0]!=commit:
        raise RuntimeError('Remote SHA verification failed')
    state('completed_and_pushed',commit=commit,results=str(target))


def main():
    # Lock held for worker lifetime prevents two launches racing on checkpoints.
    import msvcrt
    lock=(OUT/'geometry5_followup_20261127.lock').open('a+b')
    lock.seek(0); lock.write(b'0');lock.flush();lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise RuntimeError('Another formal follow-up worker holds the lock')
    try:
        checks=json.loads((OUT/'geometry5_followup_checks_20261127.json').read_text())
        if not checks['passed']: raise RuntimeError('Checks did not pass')
        state('auditing_reused_full_lstm_controls');audit_previous_lstm()
        execute('experiments.geometry_followup.run',TRANSFORMER,15)
        execute('experiments.geometry_ablation.run',LSTM,20,
                ['lstm_geometry5_depth2','lstm_geometry5_depth3'])
        state('checking_and_publishing');publish()
    except Exception as error:
        state('failed',error=str(error))
        (OUT/'geometry5_followup_worker_error_20261127.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__': main()
