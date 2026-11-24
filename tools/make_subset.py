"""Write a small copy of Sihui's processed_full that contains exactly the sketches the
345-class protocol selects (seed 42, lengths <= 250, 5000/500/500 per class).

The selection reproduces PreparedDataset in delta_run/run_processed.py step by step (one
default_rng per split, categories in manifest order), so loading the subset with the same
protocol yields the same sample ids: every category then has <= limit sketches, the loader
never draws from the rng, and sorted selection order is preserved.

    python tools/make_subset.py --src /projects/bjwy/shared/processed_full --dst /u/sjin5/processed_subset
    tar -C /u/sjin5 -czf processed_subset.tgz processed_subset       # ~1-2 GB, upload this
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np

LIMITS = {'train': 5000, 'valid': 500, 'test': 500}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--dst', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-length', type=int, default=250)
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    manifest = json.loads((src / 'manifest.json').read_text())
    categories = sorted(manifest['class_to_index'], key=manifest['class_to_index'].get)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / 'manifest.json', dst / 'manifest.json')
    report = {'source': str(src.resolve()), 'seed': a.seed, 'max_length': a.max_length,
              'limits': LIMITS, 'counts': {}}
    for split, limit in LIMITS.items():
        rng = np.random.default_rng(a.seed)
        total = 0
        for category in categories:
            with np.load(src / category / f'{split}.npz', allow_pickle=False) as shard:
                offsets, labels, ids, points = shard['offsets'], shard['labels'], shard['ids'], shard['points']
                lengths = np.diff(offsets)
                selected = np.flatnonzero(lengths > 0)
                selected = selected[lengths[selected] <= a.max_length]
                if len(selected) > limit:
                    selected = np.sort(rng.choice(selected, limit, replace=False))
                new_points = np.concatenate([points[offsets[i]:offsets[i + 1]] for i in selected]).astype(np.float32)
                new_offsets = np.concatenate([[0], np.cumsum(lengths[selected])]).astype(offsets.dtype)
                (dst / category).mkdir(exist_ok=True)
                np.savez(dst / category / f'{split}.npz', points=new_points, offsets=new_offsets,
                         labels=labels[selected], ids=ids[selected])
                total += len(selected)
        report['counts'][split] = total
        print(f'{split}: {total} sketches', flush=True)
    (dst / 'subset_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
