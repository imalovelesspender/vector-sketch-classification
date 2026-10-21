"""Execute Sihui's data_preprocessing/preprocess.ipynb headlessly, unchanged except
for one setting: DOWNLOAD_BOTH_VERSIONS=False, so only the 345 .full.npz archives
used by USE_FULL=True are downloaded (the standard archives are never read).

Run from the root of a checkout of the data-preprocessing branch:
    python run_notebook.py --stop-after-download   # login node: download + MD5 only
    python run_notebook.py                         # whole notebook (inside a job)
Plots are rendered with the Agg backend and discarded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

NOTEBOOK = Path('data_preprocessing/preprocess.ipynb')
NOTEBOOK_SHA256 = 'af55d144d5bee88d56fa4d3e71a76fd525612a51db1a4290f15dd28083cdf11c'
ORIGINAL = 'DOWNLOAD_BOTH_VERSIONS = True'
REPLACEMENT = 'DOWNLOAD_BOTH_VERSIONS = False'
DOWNLOAD_CELL_MARKER = 'def download_archive(item):'

parser = argparse.ArgumentParser()
parser.add_argument('--stop-after-download', action='store_true')
args = parser.parse_args()

raw = NOTEBOOK.read_bytes()
assert hashlib.sha256(raw).hexdigest() == NOTEBOOK_SHA256, 'preprocess.ipynb is not the recorded version'
cells = [''.join(c['source']) for c in json.loads(raw)['cells'] if c['cell_type'] == 'code']
assert sum(ORIGINAL in c for c in cells) == 1
cells = [c.replace(ORIGINAL, REPLACEMENT) for c in cells]

import matplotlib.pyplot as plt  # noqa: E402

plt.show = lambda *a, **k: plt.close('all')
namespace = {'__name__': '__main__', 'display': print}
for number, source in enumerate(cells, 1):
    print(f'===== notebook code cell {number}/{len(cells)} =====', flush=True)
    exec(compile(source, f'preprocess.ipynb:code{number}', 'exec'), namespace)
    if args.stop_after_download and DOWNLOAD_CELL_MARKER in source:
        print('STOPPED AFTER DOWNLOAD CELL', flush=True)
        break
else:
    assert not args.stop_after_download
    print('NOTEBOOK COMPLETED', flush=True)
