# SIH26168 — Speed-Estimation Model (IO-VNBD)

Smartphone dead-reckoning speed model for the SIH26168 problem statement
(Intelligent Dead Reckoning with GNSS/NavIC fusion). Trains a small CNN+GRU that
predicts vehicle forward speed (m/s) from phone IMU sensors, benchmarked fairly in
**both PyTorch and TensorFlow** on identical data.

## Pipeline
```
download_data.py      # pull IO-VNBD smartphone (S-*) CSVs from GitHub LFS
dataset.py            # round-1 features (9 raw channels, 2 s window)
dataset_v2.py         # round-2 features (7 gravity-removed, rotation-invariant, 5 s)
prepare_splits.py     # cache identical train/val/test .npy (round 1) -> splits/
prepare_splits_v2.py  # same for round 2 -> splits_v2/
train_pytorch.py      # SPLIT_DIR + TAG env-configurable
train_tf.py           # identical architecture in Keras
common_eval.py        # shared hyperparameters + metrics (no bias)
compare.py [TAG]      # print side-by-side table
```

## Reproduce
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python download_data.py 40           # ~33 unique drives, ~160 MB
# Round 1
python prepare_splits.py
python train_pytorch.py && python train_tf.py
python compare.py
# Round 2 (engineered features)
python prepare_splits_v2.py
SPLIT_DIR=splits_v2 TAG=_v2 python train_pytorch.py
SPLIT_DIR=splits_v2 TAG=_v2 python train_tf.py
python compare.py _v2
```

## Model
`Conv1D(32) -> Conv1D(64) -> GRU(64) -> Dense(1)`, ~37k params. Input a window of
IMU features, output speed (m/s). Supervised regression, MSE loss, Adam, 12 epochs,
seed 42. Split is by drive (whole file to one split) to prevent window leakage.

## Results (test set, identical cached arrays)
| | Round 1 (raw) | Round 2 (engineered) |
| --- | --- | --- |
| PyTorch MAE / R² | 4.23 m/s / 0.66 | 3.16 m/s / 0.82 |
| TensorFlow MAE / R² | 4.40 m/s / 0.60 | 3.12 m/s / 0.83 |

Framework accuracy is effectively tied; **TensorFlow chosen** for ~2× faster training
and direct TensorFlow Lite export to the Android app.

## Data note
The IO-VNBD `GPS SPEED (Kmh)` column is actually **m/s** (verified against GPS
lat/lon-derived speed, ratio ~1.01). Files are Git LFS — real content is served from
`media.githubusercontent.com/media/...`, not `raw.githubusercontent.com`.

## Not in this repo
`data/`, `splits*/`, trained weights, and logs are git-ignored (regenerate via the
scripts above). Dataset: https://github.com/onyekpeu/IO-VNBD
