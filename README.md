# SIH26168 — Speed-Estimation Model (IO-VNBD)

Smartphone dead-reckoning speed model for the SIH26168 problem statement
(Intelligent Dead Reckoning with GNSS/NavIC fusion). Trains a compact TCN that predicts
vehicle forward speed (m/s) from phone IMU sensors, plus the fusion / drift analysis and the
app-team deliverables.

## Repository layout
```
src/            all source
  paths.py            central path resolver (repo root + folders below)
  download_data.py    pull IO-VNBD smartphone (S-*) CSVs from GitHub LFS
  dataset.py          round-1 features (9 raw channels, 2 s window)
  dataset_v2.py       round-2 features (7 gravity-removed, rotation-invariant, 5 s)
  prepare_splits*.py  cache identical train/val/test .npy -> splits/ , splits_v2/
  common_eval.py      shared hyperparameters + metrics (no framework bias)
  train_pytorch.py    speed model in PyTorch   (SPLIT_DIR + TAG env-configurable)
  train_tf.py         same architecture in Keras
  train_tcn_tf.py     the shipped TCN speed model
  compare.py [TAG]    print the PyTorch vs TensorFlow table
  evaluate_drift.py   along-track distance drift over simulated blackouts
  fusion.py           dead-reckoning + 2D position drift (speed vs heading decomposition)
  heading.py          gyro + magnetometer heading (magnetometer path rejected — see notes)
  decimate.py         reference high-rate -> 10 Hz decimator for the app (gate G2a)
  preprocess.py       standalone feature spec the Kotlin port must match
  export_artifacts.py build all app-team deliverables (-> exports/)
  make_stub.py        interface stub .tflite (-> exports/)
  demo.py             build the tunnel-demo trajectory (-> demo/)
exports/        app-team deliverables: engine_*.tflite, norm.json, manifest.json,
                testset.npz, engine_stub.tflite, CHANGELOG.md
results/        metrics json (results_*, drift_*, fusion_drift)
demo/           nav-demo.html (self-contained) + demo_data.json
data/ splits*/  regenerated, git-ignored
```

## Reproduce
Run scripts from the repo root; each resolves its own paths via `src/paths.py`.
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py 40           # ~33 unique drives, ~160 MB
# Round 1
python src/prepare_splits.py
python src/train_pytorch.py && python src/train_tf.py
python src/compare.py
# Round 2 (engineered features)
python src/prepare_splits_v2.py
SPLIT_DIR=splits_v2 TAG=_v2 python src/train_pytorch.py
SPLIT_DIR=splits_v2 TAG=_v2 python src/train_tf.py
python src/compare.py _v2
# Shipped TCN + deliverables
python src/train_tcn_tf.py
python src/export_artifacts.py           # writes exports/
python src/fusion.py                     # 2D drift analysis -> results/
python src/demo.py                        # demo trajectory -> demo/
```

## Model
Shipped: a TCN (dilated causal 1-D convs), ~115k params, builtins-only TFLite. Input a
50×7 window of rotation-invariant IMU features @10 Hz, output speed (m/s). Supervised
regression, MSE, Adam. Split is by drive (whole file to one split) to prevent window leakage.

## Results (held-out test)
| | Round 1 (raw) | Round 2 (engineered) | TCN |
| --- | --- | --- | --- |
| PyTorch MAE / R² | 4.23 / 0.66 | 3.16 / 0.82 | — |
| TensorFlow MAE / R² | 4.40 / 0.60 | 3.12 / 0.83 | 4.06 / 0.66* |
*TCN test split is larger/harder (see `results/`); its win shows up in drift, not raw MAE:
along-track distance drift median dropped 18% → 12%.

Framework accuracy is effectively tied; **TensorFlow chosen** for faster training and
direct TFLite export. Heading (not speed) is the KPI bottleneck — see the vault notes.

## Deliverables for the app team (`exports/`)
`engine_tcn_v1_*.tflite` (+ `engine_stub.tflite`), `norm.json`, `manifest.json` (with a
self-test vector), `testset.npz` (parity answer key, hard cases). G1 export parity passes
at 5.7e-6. Regenerate all with `python src/export_artifacts.py`.

## Data note
The IO-VNBD `GPS SPEED (Kmh)` column is actually **m/s** (verified against GPS lat/lon-derived
speed, ratio ~1.01). Files are Git LFS — real content is served from
`media.githubusercontent.com/media/...`, not `raw.githubusercontent.com`. Dataset:
https://github.com/onyekpeu/IO-VNBD
