"""Run and publish the user-approved deadline experiment in an independent worker."""
import ctypes
from ctypes import wintypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

from .prepare import prepare

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parents[1]
OUT=WORKSPACE/'outputs'
CACHE=OUT/'geometry5_short_cache_20261124'
RESULTS=OUT/'geometry5_short_20261124'
LOG=OUT/'geometry5_short_training_20261124.log'
STATE=OUT/'geometry5_short_supervisor_live.json'
ENV=os.environ.copy()
ENV.update(USERPROFILE='C:\\Users\\zenin',PYTHONPATH=str(WORKSPACE/'work/stroke-gpu-runtime'),
           PYTHONIOENCODING='utf-8',GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='never')
ENV['PATH']='C:\\Program Files\\Git\\cmd;'+ENV.get('PATH','')

class Power(ctypes.Structure):
    _fields_=[('ac',wintypes.BYTE),('flag',wintypes.BYTE),('percent',wintypes.BYTE),
              ('reserved',wintypes.BYTE),('seconds',wintypes.DWORD),('full',wintypes.DWORD)]

KERNEL=ctypes.WinDLL('kernel32',use_last_error=True)
KERNEL.GetSystemPowerStatus.argtypes=[ctypes.POINTER(Power)]
KERNEL.SetThreadExecutionState.argtypes=[wintypes.DWORD]
KERNEL.SetThreadExecutionState.restype=wintypes.DWORD

def state(phase,**fields):
    power=Power()
    ac=bool(KERNEL.GetSystemPowerStatus(ctypes.byref(power))) and power.ac==1
    KERNEL.SetThreadExecutionState(0x80000001 if ac else 0x80000000)
    data=dict(phase=phase,scope='approved 345-class short budget',worker_pid=os.getpid(),
              updated_utc=datetime.now(timezone.utc).isoformat(),on_ac=ac,epochs=5,
              train_per_class=1000,valid_per_class=100,**fields)
    if LOG.exists():
        with LOG.open('rb') as f:
            f.seek(max(0,LOG.stat().st_size-65536))
            text=f.read().decode('utf-8',errors='replace')
        names=re.findall(r'^BEGIN (\S+)',text,re.M)
        if names:
            data['current_config']=names[-1]
            h=RESULTS/names[-1]/'history.json'
            try:
                hist=json.loads(h.read_text())
                data.update(completed_epoch=hist[-1]['epoch'],best_validation=max(x['valid']['top1'] for x in hist),
                            last_epoch_seconds=hist[-1]['seconds'])
            except (OSError,ValueError,IndexError):pass
    if (RESULTS/'summary.json').exists():
        data['completed_configs']=len(json.loads((RESULTS/'summary.json').read_text()))
    for file in [STATE,OUT/'geometry5_supervisor_live.json',OUT/'current_training_status.json']:
        temp=file.with_suffix('.tmp')
        temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(file)

def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=ENV,text=True,encoding='utf-8').strip()

def publish():
    rows=json.loads((RESULTS/'summary.json').read_text())
    if len(rows)!=14:raise RuntimeError('Expected 14 completed configurations')
    for row in rows:
        hist=json.loads((RESULTS/row['name']/'history.json').read_text())
        if len(hist)!=5 or not row['checkpoint_reload_verified']:
            raise RuntimeError('Incomplete or unverified result')
        if any(x['train']['samples']!=345000 or x['valid']['samples']!=34500 for x in hist):
            raise RuntimeError('Sample count mismatch')
        if row['valid_top1']!=max(x['valid']['top1'] for x in hist):
            raise RuntimeError('Best-validation mismatch')
    target=ROOT/'docs/experiments/geometry5_short_20261124'
    target.mkdir(parents=True,exist_ok=True)
    for name in ['summary.csv','summary.json','protocol.json']:
        shutil.copy2(RESULTS/name,target/name)
    for row in rows:
        shutil.copy2(RESULTS/row['name']/'history.json',target/f"{row['name']}_history.json")
    lookup={r['name']:r for r in rows}
    best=max(rows,key=lambda r:r['valid_top1'])
    lines=['# 345-class short-budget architecture screening', '',
           '**User-approved November 24 deadline protocol. This is not the original 20-epoch experiment.**','',
           'All 14 configurations: 345,000 train / 34,500 validation sketches, all 345 classes, '
           '5 epochs, seed 42, Adam 0.001, batch 128, float32. Same sampled IDs and '
           'training-fitted normalization for every model. No test-set predictions.','',
           'Inputs remain point tokens: point3 = dx/dy/pen; geometry5 adds centered XY. '
           'This is not whole-stroke tokenization or an XAI hierarchical encoder.','',
           '| Configuration | Best val. top-1 | Best epoch | Parameters | Train + valid seconds |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['valid_top1']:.2%} | {r['best_epoch']} | {r['parameters_actual']:,} | {r['train_valid_seconds']:.1f} |")
    lines+=['','Depth/width change parameter count. Transformer FF width stays 3x model width. '
            'Heads are varied at fixed width. A single seed and five epochs do not establish convergence, '
            'statistical significance or a universally best architecture. Historical full-budget and '
            'test scores must not be mixed into this table.','',
            'Source: team slides and Sihui preprocessing/model references documented in '
            '`experiments/geometry_ablation/README.md`. Cache source hashes, source-code hashes, '
            'sampling and the approved budget change are recorded in protocol.json.']
    (target/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    conclusions=['# 给组员的结论：统一短预算筛选','',
        '范围：345 类，每类 1,000 张训练、100 张验证，14 组各训练 5 轮。'
        '这些是验证成绩，不是 test 成绩，也不是原来计划的 20 轮完整实验。','']
    for family in ['lstm','transformer']:
        a=lookup[f'{family}_point3_control'];b=lookup[f'{family}_geometry5_base']
        delta=(b['valid_top1']-a['valid_top1'])*100
        conclusions.append(f"- {family.upper()}：加入中心化位置特征后，验证准确率从 {a['valid_top1']:.2%} 变为 {b['valid_top1']:.2%}，变化 {delta:+.2f} 个百分点。")
        family_best=max((r for r in rows if r['family']==family),key=lambda r:r['valid_top1'])
        conclusions.append(f"- {family.upper()} 本轮最高配置是 {family_best['name']}，验证准确率 {family_best['valid_top1']:.2%}。")
    conclusions+=['',f"本轮整体最高：{best['name']}，{best['valid_top1']:.2%}。",'',
        '解释时要说“在这次数据量和 5 轮预算下的观察”。更大、更深、更多头是否有效，'
        '应对照表中同一因素的配置，不能把增加参数量的收益都归因于结构设计。'
        '还不能推出充分训练后的最优配置。原来的长任务和 checkpoint 已保留，未与本表混用。']
    (target/'CONCLUSION_ZH.md').write_text('\n'.join(conclusions)+'\n',encoding='utf-8')
    team=(f'I finished the 14-config screening on all 345 classes using a shared short budget: '
          f'1,000 training and 100 validation sketches per class, five epochs per run. '
          f'The highest validation accuracy was {best["valid_top1"]:.2%} with {best["name"]}. '
          'The table separates the input-feature comparison from depth, width and head-count ablations. '
          'These are short-budget validation results, not final test scores or the original 20-epoch results. '
          'I have included the configurations, parameter counts, timing and reproducible records on my branch.\n')
    (target/'TEAM_UPDATE.txt').write_text(team,encoding='utf-8')
    if git('branch','--show-current')!='anson/stroke-experiments-20261118':
        raise RuntimeError('Unexpected branch; refusing to push')
    if git('diff','--cached','--name-only'):raise RuntimeError('Other staged changes exist')
    git('add','--','experiments/geometry_short_budget',str(target.relative_to(ROOT)))
    git('diff','--cached','--check')
    git('commit','-m','Archive approved 345-class five-epoch architecture screening and conclusions')
    git('push','-u','origin','anson/stroke-experiments-20261118')
    commit=git('rev-parse','HEAD')
    remote=git('ls-remote','--heads','origin','refs/heads/anson/stroke-experiments-20261118').split()[0]
    if commit!=remote:raise RuntimeError('Remote commit verification failed')
    state('completed_and_pushed',commit=commit,
          github='https://github.com/imalovelesspender/vector-sketch-classification/tree/anson/stroke-experiments-20261118')

def main():
    try:
        state('preparing_short_cache')
        prepare(OUT/'geometry5_full_cache_20261118',CACHE)
        startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        for attempt in range(1,5):
            cmd=[sys.executable,'-u','-m','experiments.geometry_ablation.run','--cache',str(CACHE),
                 '--output',str(RESULTS),'--epochs','5','--device','cuda']
            if (RESULTS/'protocol.json').exists():cmd.append('--resume')
            with LOG.open('a',encoding='utf-8') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,
                    stderr=subprocess.STDOUT,startupinfo=startup,creationflags=subprocess.CREATE_NO_WINDOW)
                while process.poll() is None:
                    state('running',training_pid=process.pid,attempt=attempt)
                    time.sleep(15)
            if process.returncode==0:
                state('checking_and_publishing');publish();return
            tail=LOG.read_text(encoding='utf-8',errors='replace')[-6000:]
            if 'Traceback' in tail or attempt==4:
                raise RuntimeError(f'Training exited {process.returncode}; inspect {LOG.name}')
            state('retrying_unexpected_exit',exit_code=process.returncode,attempt=attempt)
            time.sleep(15)
    except Exception as e:
        state('failed',error=str(e))
        (OUT/'geometry5_short_worker_error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:KERNEL.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
