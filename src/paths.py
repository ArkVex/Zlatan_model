"""Central path resolver so scripts work from the repo root after the src/ reorg.

Layout:
  <ROOT>/src/        source (this package)
  <ROOT>/data/       downloaded IO-VNBD CSVs        (gitignored)
  <ROOT>/splits*/    cached train/val/test .npy     (gitignored)
  <ROOT>/results/    metrics json                   (committed)
  <ROOT>/exports/    app-team deliverables          (committed)
  <ROOT>/demo/       demo html + data               (committed)
  trained .keras/.pt models live at <ROOT> (gitignored, regenerated)
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(ROOT, "splits")
SPLITS_V2 = os.path.join(ROOT, "splits_v2")
RESULTS = os.path.join(ROOT, "results")
EXPORTS = os.path.join(ROOT, "exports")
DEMO = os.path.join(ROOT, "demo")
MODELS = ROOT   # trained model files sit at the repo root

for _d in (RESULTS, EXPORTS, DEMO):
    os.makedirs(_d, exist_ok=True)
