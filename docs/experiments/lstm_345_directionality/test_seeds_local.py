"""CPU end-to-end test of the seed / parameter-matched pipeline on synthetic data.

    python test_seeds_local.py <empty work dir>

Builds a checkout of 5fe0e29 plus this folder's psc_run files, synthetic processed
shards for the real 345 categories (tiny counts, some sketches > 250 points), and a
reduced protocol (5/2/2 per class, 2 epochs, CPU). Only the protocol hash/path and
the device are patched. Then: seed-42 reference runs, preflight, all 9 planned runs,
summary, determinism checks and negative tests.
"""
import copy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
work = Path(sys.argv[1]).resolve()
assert not work.exists() or not any(work.iterdir()), 'use an empty directory'
root = work / 'checkout'
root.mkdir(parents=True)
subprocess.run(f'git -C {REPO} archive 5fe0e29 | tar x -C {root}', shell=True, check=True)
(root / 'psc_run').mkdir()
for name in ['run_seeded345.py', 'preflight_seeds345.py', 'summarize_seeds345.py', 'seed_runs.tsv',
             'protocol_config.json']:
    shutil.copy(HERE / name, root / 'psc_run' / name)
for name in ['run_processed.py', 'historical_source_hashes.json']:
    shutil.copy(HERE.parent / 'lstm_345class_20261014' / name, root / 'psc_run' / name)
sha = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731

# Synthetic processed_full for the real 345 categories.
real_protocol = json.loads((HERE / 'protocol_config.json').read_text())
categories = real_protocol['data']['categories']
data = work / 'processed_full'
rng = np.random.default_rng(0)
counts = {'train': 8, 'valid': 4, 'test': 4}
for label, category in enumerate(categories):
    (data / category).mkdir(parents=True)
    for split, n in counts.items():
        lengths = rng.integers(3, 12, n)
        lengths[0] = 300  # must be excluded by the 250-point filter
        points = rng.normal(size=(int(lengths.sum()), 3)).astype(np.float32)
        points[:, 2] = rng.integers(0, 2, len(points))
        np.savez(data / category / f'{split}.npz', points=points,
                 offsets=np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
                 labels=np.full(n, label, dtype=np.int64),
                 ids=np.asarray([f'{category}/full/{split}/{i}' for i in range(n)]))
(data / 'manifest.json').write_text(json.dumps({
    'class_to_index': {c: i for i, c in enumerate(categories)}, 'representation': 'stroke-3',
    'normalization': {'fit_split': 'train', 'scale': 1.0}}))
manifest_sha = sha((data / 'manifest.json').read_bytes())
verification = work / 'verification.json'
verification.write_text(json.dumps({'manifest_sha256': manifest_sha, 'scale_matches_sihui': True}))

# Reduced protocol; only counts, epochs, batch size and device differ from the real one.
protocol = copy.deepcopy(real_protocol)
protocol['data'].update(train_per_class=5, val_per_class=2, test_per_class=2, batch_size=64)
protocol['train'].update(epochs=2, device='cpu')
protocol_path = work / 'protocol_test.json'
protocol_path.write_bytes(json.dumps(protocol, indent=1).encode())

sys.path[:0] = [str(root / 'psc_run'), str(root)]
import os  # noqa: E402
os.chdir(root)
import run_seeded345 as runner  # noqa: E402
import run_processed as adapter  # noqa: E402
runner.DEVICE = 'cpu'
runner.PROTOCOL_PATH = protocol_path
runner.PROTOCOL_CONFIG_SHA256 = sha(protocol_path.read_bytes())

# Hand-made seed-42 BiLSTM reference, as run_all345.py would leave it.
reference = root / 'results/lstm_345_bi256'
reference.mkdir(parents=True)
cfg = copy.deepcopy(protocol)
cfg['data']['data_dir'] = str(data)
cfg['data']['preprocessing'].update(manifest=str(data / 'manifest.json'), manifest_sha256=manifest_sha,
                                    adapter=str(root / 'psc_run/run_processed.py'))
cfg['output_dir'] = 'results/lstm_345_bi256'
(reference / 'config.json').write_text(json.dumps(cfg, indent=2))
for split, limit in [('train', 5), ('valid', 2), ('test', 2)]:
    ds = adapter.PreparedDataset(str(data), categories, split, limit, 250, 42)
    assert all(not i.endswith('/0') for i in ds.ids), 'the 300-point sketch was not excluded'
    target = reference / f'{split}_sample_ids.json' if split != 'test' else work / 'test_ids_handmade.json'
    target.write_bytes(json.dumps(ds.ids).encode())
ids_sha = [sha((reference / f'{s}_sample_ids.json').read_bytes()) for s in ('train', 'valid')]


def run(output, bidirectional, hidden, seed, test_ids=None, reference_dir=reference):
    runner.main(['--data-dir', str(data), '--reference-dir', str(reference_dir),
                 '--reference-test-ids', str(test_ids or root / 'results/seed_reference/test_sample_ids.json'),
                 '--bidirectional', bidirectional, '--hidden-dim', str(hidden),
                 '--train-seed', str(seed), '--output-dir', output])


def expect_failure(label, function):
    try:
        function()
    except (AssertionError, FileExistsError, SystemExit) as error:
        print(f'NEGATIVE TEST OK ({label}): {type(error).__name__}: {str(error)[:80]}')
        return
    raise AssertionError(f'negative test did not fail: {label}')


# Seed-42 runs of the three original architectures (stand-ins for the PSC runs).
for name, bi, h in [('bi256', 'true', 256), ('uni512', 'false', 512), ('uni256', 'false', 256)]:
    run(f'results/tmp42_{name}', bi, h, 42, test_ids=work / 'test_ids_handmade.json')
    target = root / f'results/lstm_345_{name}'
    target.mkdir(exist_ok=True)
    for f in (root / f'results/tmp42_{name}').iterdir():
        if f.name != 'test_sample_ids.json':  # the PSC seed-42 runs have no test ID file
            shutil.copy(f, target / f.name)
    shutil.rmtree(root / f'results/tmp42_{name}')

argv_preflight = ['preflight_seeds345.py', '--data-dir', str(data), '--verification', str(verification),
                  '--reference-dir', 'results/lstm_345_bi256',
                  '--reference-test-ids', 'results/seed_reference/test_sample_ids.json',
                  '--expected-ids-sha256', *ids_sha]
sys.argv = argv_preflight
runpy.run_path(str(root / 'psc_run/preflight_seeds345.py'), run_name='__main__')
assert (root / 'results/seed_reference/test_sample_ids.json').read_bytes() == \
    (work / 'test_ids_handmade.json').read_bytes()

# All nine planned runs, exactly as the array would run them.
plan = [line.split('\t') for line in (root / 'psc_run/seed_runs.tsv').read_text().splitlines()[1:]]
for task, output, bi, hidden, seed in plan:
    run(output, bi, int(hidden), int(seed))

# Identical sample IDs everywhere; different seeds change training, same seed reproduces it.
for task, output, *_ in plan:
    for split in ('train', 'valid'):
        assert (root / output / f'{split}_sample_ids.json').read_bytes() == \
            (reference / f'{split}_sample_ids.json').read_bytes()
    assert (root / output / 'test_sample_ids.json').read_bytes() == \
        (root / 'results/seed_reference/test_sample_ids.json').read_bytes()
    saved = json.loads((root / output / 'config.json').read_text())
    assert saved['seed'] == int(plan[int(task)][4])
h = lambda d: json.loads((root / d / 'history.json').read_text())  # noqa: E731
assert h('results/lstm_345_bi256_s43') != h('results/lstm_345_bi256_s44') != h('results/lstm_345_bi256')
run('results/repeat_bi256_s43', 'true', 256, 43)
assert h('results/repeat_bi256_s43') == h('results/lstm_345_bi256_s43'), 'same seed must reproduce'
import torch  # noqa: E402
a = torch.load(root / 'results/repeat_bi256_s43/last.pt', weights_only=False)['model_state']
b = torch.load(root / 'results/lstm_345_bi256_s43/last.pt', weights_only=False)['model_state']
assert all(torch.equal(a[k], b[k]) for k in a)
c = torch.load(root / 'results/lstm_345_bi256_s44/last.pt', weights_only=False)['model_state']
assert not all(torch.equal(a[k], c[k]) for k in a)
print('DETERMINISM OK: same training seed -> identical weights/history; seeds 43 vs 44 differ; IDs identical')
shutil.rmtree(root / 'results/repeat_bi256_s43')

sys.argv = ['summarize_seeds345.py', '--reference-test-ids', 'results/seed_reference/test_sample_ids.json',
            '--output', 'results/lstm_345_seeds_summary']
runpy.run_path(str(root / 'psc_run/summarize_seeds345.py'), run_name='__main__')

# Negative tests.
expect_failure('output exists', lambda: run('results/lstm_345_uni378', 'false', 378, 42))
expect_failure('architecture not planned', lambda: run('results/x1', 'true', 512, 43))
bad_test = work / 'bad_test_ids.json'
ids = json.loads((root / 'results/seed_reference/test_sample_ids.json').read_text())
bad_test.write_text(json.dumps(ids[:-1] + [ids[-1].rsplit('/', 1)[0] + '/0']))
expect_failure('test IDs differ', lambda: run('results/x2', 'false', 378, 45, test_ids=bad_test))
bad_reference = work / 'bad_reference'
shutil.copytree(reference, bad_reference)
train_ids = json.loads((bad_reference / 'train_sample_ids.json').read_text())
(bad_reference / 'train_sample_ids.json').write_text(json.dumps(train_ids[1:] + train_ids[:1]))
expect_failure('train IDs differ', lambda: run('results/x3', 'false', 378, 45, reference_dir=bad_reference))
changed = json.loads((bad_reference / 'config.json').read_text())
changed['train']['lr'] = 0.01
(bad_reference / 'config.json').write_text(json.dumps(changed))
expect_failure('reference settings differ', lambda: run('results/x4', 'false', 378, 45, reference_dir=bad_reference))
sys.argv = argv_preflight
expect_failure('preflight: planned outputs exist',
               lambda: runpy.run_path(str(root / 'psc_run/preflight_seeds345.py'), run_name='__main__'))
print('ALL LOCAL TESTS PASSED')
