"""Load IO-VNBD smartphone CSVs into windowed (X, y) arrays.

Input channels (9): accelerometer XYZ, gyroscope yaw/pitch/roll, magnetometer XYZ.
Label: vehicle forward speed in m/s (from GPS SPEED, km/h -> m/s).
Sampling rate: 10 Hz. A window of WIN samples predicts the speed at the window's end.
"""
import glob
import os
import numpy as np
import pandas as pd

from paths import DATA as DATA_DIR
WIN = 20          # 20 samples @10Hz = 2.0 s of context
STRIDE = 2        # hop between windows (0.2 s) -> more training samples

# Match columns by keyword so minor header differences don't break loading.
COL_KEYS = {
    "speed":  ["GPS SPEED"],
    "acc_x":  ["ACCELEROMETER X"],
    "acc_y":  ["ACCELEROMETER Y"],
    "acc_z":  ["ACCELEROMETER Z"],
    "gyr_y":  ["GYROSCOPE YAW"],
    "gyr_p":  ["GYROSCOPE PITCH"],
    "gyr_r":  ["GYROSCOPE ROLL"],
    "mag_x":  ["MAGNETIC FIELD X"],
    "mag_y":  ["MAGNETIC FIELD Y"],
    "mag_z":  ["MAGNETIC FIELD Z"],
    "acc_gps": ["GPS ACCURACY"],
}
FEATURES = ["acc_x", "acc_y", "acc_z", "gyr_y", "gyr_p", "gyr_r", "mag_x", "mag_y", "mag_z"]


def _resolve(cols):
    """Map our short names -> actual dataframe column names."""
    up = {c.upper().strip(): c for c in cols}
    out = {}
    for short, keys in COL_KEYS.items():
        found = None
        for want in keys:
            for uc, orig in up.items():
                if want in uc:
                    found = orig
                    break
            if found:
                break
        out[short] = found
    return out


def load_file(path):
    """Return (X_windows, y_speed) for one CSV, or (None, None) if unusable."""
    try:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: read error {e}")
        return None, None
    cmap = _resolve(df.columns)
    if any(cmap[k] is None for k in FEATURES + ["speed"]):
        print(f"  skip {os.path.basename(path)}: missing columns")
        return None, None

    def num(col):
        return pd.to_numeric(df[col], errors="coerce")

    feat = np.stack([num(cmap[f]).to_numpy(dtype=np.float32) for f in FEATURES], axis=1)
    # NOTE: header says "(Kmh)" but the values are actually m/s (verified against
    # speed computed from GPS lat/lon deltas: ratio ~1.01). Do NOT divide by 3.6.
    speed_ms = num(cmap["speed"]).to_numpy(dtype=np.float32)

    # Drop rows with NaNs in features or label.
    good = np.isfinite(feat).all(axis=1) & np.isfinite(speed_ms)
    # Optional GPS-accuracy gate: keep rows with a reasonable fix when available.
    if cmap["acc_gps"] is not None:
        acc = num(cmap["acc_gps"]).to_numpy(dtype=np.float32)
        good &= (~np.isfinite(acc)) | (acc <= 20.0)
    feat, speed_ms = feat[good], speed_ms[good]
    if len(feat) < WIN + 5:
        return None, None

    # Sliding windows; label = speed at the last sample of the window.
    idx = range(0, len(feat) - WIN, STRIDE)
    X = np.stack([feat[i:i + WIN] for i in idx]).astype(np.float32)
    y = np.array([speed_ms[i + WIN - 1] for i in idx], dtype=np.float32)
    return X, y


def load_all(limit=None):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "S-*.csv")))
    if limit:
        files = files[:limit]
    Xs, ys, used = [], [], []
    for f in files:
        X, y = load_file(f)
        if X is None:
            continue
        Xs.append(X)
        ys.append(y)
        used.append(os.path.basename(f))
    if not Xs:
        raise SystemExit("No usable files loaded.")
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    print(f"Loaded {len(used)} files -> X{X.shape} y{y.shape}  "
          f"speed range {y.min():.1f}-{y.max():.1f} m/s")
    return X, y, used


if __name__ == "__main__":
    X, y, used = load_all()
    print("files used:", used)
