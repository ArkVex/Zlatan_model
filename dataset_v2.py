"""Round-2 features: gravity-removed, rotation-invariant, energy-aware.

Why v2: raw phone accelerometer carries gravity + phone-tilt, so absolute axes
don't map to speed and don't generalize across how the phone is mounted. Here we
use the GRAVITY vector to split linear acceleration into a vertical component and
a horizontal magnitude (both invariant to phone orientation), plus gyro activity
and magnitudes — features that actually correlate with vehicle speed.

Channels (7, all orientation-invariant):
  a_horiz     horizontal linear-accel magnitude (forward+lateral)
  a_vert      vertical linear-accel component
  a_lin_mag   |linear accel|
  gyr_y/p/r   gyroscope yaw/pitch/roll
  gyro_mag    |gyro|
Label: forward speed in m/s (GPS SPEED column, already m/s).
Sampling: 10 Hz. Window WIN=50 (5.0 s), stride 5 (0.5 s).
"""
import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WIN = 50
STRIDE = 5

RAW_KEYS = {
    "speed": "GPS SPEED", "acc_gps": "GPS ACCURACY",
    "ax": "ACCELEROMETER X", "ay": "ACCELEROMETER Y", "az": "ACCELEROMETER Z",
    "grx": "GRAVITY X", "gry": "GRAVITY Y", "grz": "GRAVITY Z",
    "wy": "GYROSCOPE YAW", "wp": "GYROSCOPE PITCH", "wr": "GYROSCOPE ROLL",
}
FEATURES = ["a_horiz", "a_vert", "a_lin_mag", "gyr_y", "gyr_p", "gyr_r", "gyro_mag"]


def _resolve(cols):
    up = {c.upper().strip(): c for c in cols}
    out = {}
    for short, want in RAW_KEYS.items():
        found = None
        for uc, orig in up.items():
            if want in uc:
                found = orig
                break
        out[short] = found
    return out


def load_file(path):
    try:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    except Exception:
        return None, None
    c = _resolve(df.columns)
    need = ["speed", "ax", "ay", "az", "grx", "gry", "grz", "wy", "wp", "wr"]
    if any(c[k] is None for k in need):
        return None, None

    def num(k):
        return pd.to_numeric(df[c[k]], errors="coerce").to_numpy(dtype=np.float64)

    acc = np.stack([num("ax"), num("ay"), num("az")], axis=1)
    grav = np.stack([num("grx"), num("gry"), num("grz")], axis=1)
    gyro = np.stack([num("wy"), num("wp"), num("wr")], axis=1)
    speed = num("speed")

    # Linear acceleration = measured - gravity.
    a_lin = acc - grav
    g_norm = np.linalg.norm(grav, axis=1, keepdims=True) + 1e-6
    g_hat = grav / g_norm                                   # unit "up" vector
    a_vert = np.sum(a_lin * g_hat, axis=1)                  # vertical component
    a_vert_vec = a_vert[:, None] * g_hat
    a_horiz = np.linalg.norm(a_lin - a_vert_vec, axis=1)    # horizontal magnitude
    a_lin_mag = np.linalg.norm(a_lin, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    feat = np.stack([
        a_horiz, a_vert, a_lin_mag,
        gyro[:, 0], gyro[:, 1], gyro[:, 2], gyro_mag,
    ], axis=1).astype(np.float32)

    good = np.isfinite(feat).all(axis=1) & np.isfinite(speed)
    if c["acc_gps"] is not None:
        gp = pd.to_numeric(df[c["acc_gps"]], errors="coerce").to_numpy()
        good &= (~np.isfinite(gp)) | (gp <= 20.0)
    feat, speed = feat[good], speed[good].astype(np.float32)
    if len(feat) < WIN + 5:
        return None, None

    idx = range(0, len(feat) - WIN, STRIDE)
    X = np.stack([feat[i:i + WIN] for i in idx]).astype(np.float32)
    y = np.array([speed[i + WIN - 1] for i in idx], dtype=np.float32)
    return X, y


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(DATA_DIR, "S-*.csv")))
    X, y = load_file(files[0])
    print("sample", os.path.basename(files[0]), "->", X.shape, y.shape,
          "speed", round(float(y.min()), 1), "-", round(float(y.max()), 1))
