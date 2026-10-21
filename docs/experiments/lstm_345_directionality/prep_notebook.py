"""Turn Sihui's preprocess.ipynb into a plain script without changing its code.

The only edit is DOWNLOAD_BOTH_VERSIONS=False, so only the 345 .full.npz
archives used by USE_FULL=True are downloaded. --download-only keeps just the
setup and download cells (for the login node); otherwise every code cell runs.
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('notebook')
parser.add_argument('output')
parser.add_argument('--download-only', action='store_true')
args = parser.parse_args()
cells = [''.join(c['source']) for c in json.loads(Path(args.notebook).read_text())['cells']
         if c['cell_type'] == 'code']
assert len(cells) == 9, len(cells)
setup = cells[0]
old = 'DOWNLOAD_BOTH_VERSIONS = True'
assert setup.count(old) == 1 and 'USE_FULL = True' in setup
cells[0] = setup.replace(old, 'DOWNLOAD_BOTH_VERSIONS = False')
if args.download_only:
    cells = cells[:2]
prelude = ('import matplotlib\nmatplotlib.use("Agg")\n'
           'display = print  # notebook-only helper used by one summary cell\n')
body = '\n\n'.join(f'# ---- notebook code cell {i} ----\n{src}' for i, src in enumerate(cells))
Path(args.output).write_text(prelude + '\n' + body + '\n')
print(f'wrote {args.output}: {len(cells)} code cells')
