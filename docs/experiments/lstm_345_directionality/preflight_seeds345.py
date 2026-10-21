"""Gate before the seed / parameter-matched runs (needs a GPU for the re-evaluation).

Run from the historical checkout root:
    python psc_run/preflight_seeds345.py --data-dir .../processed_full \
        --verification .../processed_full_verification.json --reference-dir results/lstm_345_bi256 \
        --reference-test-ids results/seed_reference/test_sample_ids.json

1. Protocol, source hashes, data manifest and the existing seed-42 runs are unchanged.
2. Writes the reference test-ID list (selection seed 42, 500 per class, <= 250 points).
3. Re-evaluates the three existing seed-42 best.pt checkpoints on exactly those test
   samples and requires the accuracy recorded in their metrics_test.json, which ties
   the earlier test results to the same test IDs. Nothing in the run directories is written.
4. Planned outputs are fresh; every planned architecture has its expected size.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import py_compile

import torch
from torch.utils.data import DataLoader

import run_processed as adapter
import run_seeded345 as runner
from src.data.quickdraw import collate_sketches
from src.engine import run_epoch

TRAIN_IDS_SHA256 = '67e6b8afdbd1d09ece950dd56fa1f03e1495aa12423763a3e6baf9165552bf0b'
VALID_IDS_SHA256 = '97160becfbfc1101c3a212d64ff5b57d3ce90b337091de026a7a0565e4e21da7'
SEED42_RUNS = ['results/lstm_345_bi256', 'results/lstm_345_uni512', 'results/lstm_345_uni256']

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--verification', required=True)
parser.add_argument('--reference-dir', required=True)
parser.add_argument('--reference-test-ids', required=True)
parser.add_argument('--plan', default=str(runner.HERE / 'seed_runs.tsv'))
parser.add_argument('--expected-ids-sha256', nargs=2, default=[TRAIN_IDS_SHA256, VALID_IDS_SHA256],
                    metavar=('TRAIN', 'VALID'), help='only for the synthetic local test')
args = parser.parse_args()
TRAIN_IDS_SHA256, VALID_IDS_SHA256 = args.expected_ids_sha256
sha = lambda data: hashlib.sha256(data).hexdigest()  # noqa: E731
root = adapter.ROOT

assert sha(runner.PROTOCOL_PATH.read_bytes()) == runner.PROTOCOL_CONFIG_SHA256
for relative, expected in json.loads((runner.HERE / 'historical_source_hashes.json').read_text()).items():
    assert sha((root / relative).read_bytes()) == expected, relative
manifest_path = Path(args.data_dir).resolve() / 'manifest.json'
manifest_sha = sha(manifest_path.read_bytes())
report = json.loads(Path(args.verification).read_text())
assert report['manifest_sha256'] == manifest_sha and report['scale_matches_sihui'] is True
reference_dir = Path(args.reference_dir)
assert (sha((reference_dir / 'train_sample_ids.json').read_bytes()),
        sha((reference_dir / 'valid_sample_ids.json').read_bytes())) == (TRAIN_IDS_SHA256, VALID_IDS_SHA256)
for run in SEED42_RUNS:
    cfg = json.loads((root / run / 'config.json').read_text())
    assert cfg['data']['preprocessing']['manifest_sha256'] == manifest_sha, run
    assert cfg['seed'] == runner.SELECTION_SEED, run
    assert sha((root / run / 'train_sample_ids.json').read_bytes()) == TRAIN_IDS_SHA256, run
    assert sha((root / run / 'valid_sample_ids.json').read_bytes()) == VALID_IDS_SHA256, run
print('PROTOCOL, SOURCES, DATA AND SEED-42 TRAIN/VALID IDS UNCHANGED', flush=True)

plan = list(csv.DictReader(open(args.plan), delimiter='\t'))
assert [int(row['task']) for row in plan] == list(range(len(plan)))
assert len({row['output_dir'] for row in plan}) == len(plan)
for row in plan:
    assert not (root / row['output_dir']).exists(), f"output exists: {row['output_dir']}"
    key = (row['bidirectional'] == 'true', int(row['hidden_dim']))
    model = adapter.build_model(dict(json.loads(runner.PROTOCOL_PATH.read_text())['model'],
                                     bidirectional=key[0], hidden_dim=key[1]), 3, runner.NUM_CLASSES)
    got = (model.encoder.embedding_dim, sum(p.numel() for p in model.parameters()))
    assert got == runner.EXPECTED[key], (row, got)
    print(f"task {row['task']}: {row['output_dir']} train_seed={row['train_seed']} "
          f"embedding={got[0]} parameters={got[1]}")
py_compile.compile(str(runner.HERE / 'run_seeded345.py'), doraise=True)

# Reference test sample, drawn exactly as evaluate() did for the seed-42 runs.
protocol = json.loads(runner.PROTOCOL_PATH.read_text())
categories = json.loads((root / SEED42_RUNS[0] / 'config.json').read_text())['data']['categories']
test = adapter.PreparedDataset(args.data_dir, categories, 'test', protocol['data']['test_per_class'],
                               protocol['data']['max_seq_len'], runner.SELECTION_SEED)
assert len(test) == runner.NUM_CLASSES * protocol['data']['test_per_class']
assert not set(test.ids) & set(json.loads((reference_dir / 'train_sample_ids.json').read_text()))
test_path = Path(args.reference_test_ids)
test_path.parent.mkdir(parents=True, exist_ok=True)
test_bytes = json.dumps(test.ids).encode()
if test_path.exists():
    assert test_path.read_bytes() == test_bytes, 'existing reference test IDs differ'
else:
    test_path.write_bytes(test_bytes)
print(f'REFERENCE TEST IDS: {len(test)} sha256 {sha(test_bytes)}', flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = DataLoader(test, batch_size=256, shuffle=False, collate_fn=collate_sketches)
for run in SEED42_RUNS:
    checkpoint = adapter.evaluation.load_checkpoint(str(root / run / 'best.pt'))
    model = adapter.build_model(checkpoint['config']['model'], checkpoint['input_dim'], len(checkpoint['categories']))
    model.load_state_dict(checkpoint['model_state'])
    metrics = run_epoch(model.to(device), loader, device)
    recorded = json.loads((root / run / 'metrics_test.json').read_text())
    assert recorded['num_samples'] == len(test)
    # Correct-prediction counts; allow 2 sketches of GPU nondeterminism out of 172,500.
    difference = abs(round(metrics['acc'] * len(test)) - round(recorded['acc'] * len(test)))
    assert difference <= 2, (run, metrics, recorded)
    print(f"{run}: test acc {metrics['acc']:.6f} vs recorded {recorded['acc']:.6f} "
          f"({difference} sketches apart) on the reference test IDs")
print('PREFLIGHT PASSED: protocol, sources, data, seed-42 IDs, reference test IDs, '
      'seed-42 test metrics reproduced, fresh outputs, model sizes, runner syntax')
