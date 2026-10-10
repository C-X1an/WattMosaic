"""Execute the public walkthrough; no competition data or trained weights."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument('--write', action='store_true', help='Save the executed notebook outputs')
args = parser.parse_args()
path = Path(__file__).resolve().parents[1] / 'notebooks/WattMosaic.ipynb'
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
NotebookClient(notebook, timeout=60, kernel_name='python3',
               resources={'metadata': {'path': str(path.parent)}}).execute()
nbformat.validate(notebook)
if args.write:
    nbformat.write(notebook, path)
print('Public notebook executed top-to-bottom. Event metrics were displayed, not recomputed.')
