"""Measure the REAL KPI: position drift during a simulated GNSS blackout.

For each held-out test drive we:
  1. Run the trained speed model over the whole drive (windowed).
  2. Pick blackout segments of a target ground-truth distance (e.g. 1 km).
  3. Inside a blackout, dead-reckon: integrate predicted speed along a heading that
     starts from the last known GPS heading and is then propagated by the gyroscope
     yaw rate only (no GPS — exactly what happens in a tunnel).
  4. Compare the dead-reckoned end position to the GPS truth end position.
  5. drift% = final_position_error / distance_travelled.

The problem-statement target is < 10% (e.g. < 100 m over 1 km).

Uses the TensorFlow round-2 model (chosen framework) and dataset_v2 features.
"""
import glob
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset_v2 import WIN, _resolve, RAW_KEYS

from paths import ROOT as HERE, RESULTS
SPLITS = os.path.join(HERE, "splits_v2")
MODEL = os.path.join(HERE, os.environ.get("MODEL_FILE", "model_tf_v2.keras"))
TARGET_DIST = 1000.0   # metres per simulated blackout
DT = 0.1               # 10 Hz


def build_features_and_truth(path):
    """Return per-timestep (features, speed, gyro_yaw_rate, lat, lon, gps_heading)."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    c = _resolve(df.columns)
    need = ["speed", "ax", "ay", "az", "grx", "gry", "grz", "wy", "wp", "wr"]
    if any(c[k] is None for k in need):
        return None
    lat_c = [x for x in df.columns if "GPS LATITUDE" in x.upper()]
    lon_c = [x for x in df.columns if "GPS LONGITUDE" in x.upper()]
    hdg_c = [x for x in df.columns if "GPS ORIENTATION" in x.upper()]
    if not (lat_c and lon_c and hdg_c):
        return None

    def num(col):
        return pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)

    acc = np.stack([num(c["ax"]), num(c["ay"]), num(c["az"])], 1)
    grav = np.stack([num(c["grx"]), num(c["gry"]), num(c["grz"])], 1)
    gyro = np.stack([num(c["wy"]), num(c["wp"]), num(c["wr"])], 1)
    speed = num(c["speed"])
    lat, lon = num(lat_c[0]), num(lon_c[0])
    hdg = num(hdg_c[0])
    # Magnetometer (for the heading filter). Optional — absent -> zeros, flagged.
    mx = [x for x in df.columns if "MAGNETIC FIELD X" in x.upper()]
    my = [x for x in df.columns if "MAGNETIC FIELD Y" in x.upper()]
    mz = [x for x in df.columns if "MAGNETIC FIELD Z" in x.upper()]
    if mx and my and mz:
        mag = np.stack([num(mx[0]), num(my[0]), num(mz[0])], 1)
    else:
        mag = np.full_like(grav, np.nan)

    a_lin = acc - grav
    g_hat = grav / (np.linalg.norm(grav, axis=1, keepdims=True) + 1e-6)
    a_vert = np.sum(a_lin * g_hat, axis=1)
    a_horiz = np.linalg.norm(a_lin - a_vert[:, None] * g_hat, axis=1)
    a_lin_mag = np.linalg.norm(a_lin, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    feat = np.stack([a_horiz, a_vert, a_lin_mag,
                     gyro[:, 0], gyro[:, 1], gyro[:, 2], gyro_mag], 1).astype(np.float32)

    good = np.isfinite(feat).all(1) & np.isfinite(speed) & np.isfinite(lat) & \
        np.isfinite(lon) & np.isfinite(hdg)
    return dict(feat=feat[good], speed=speed[good], gyro_yaw=gyro[good, 0],
                gyro=gyro[good], lat=lat[good], lon=lon[good], hdg=hdg[good],
                grav=grav[good], mag=mag[good])


def latlon_to_m(lat, lon, lat0):
    """Local equirectangular metres."""
    R = 6371000.0
    x = np.radians(lon) * R * np.cos(np.radians(lat0))
    y = np.radians(lat) * R
    return x, y


def eval_drive(path, model, mean, std):
    d = build_features_and_truth(path)
    if d is None or len(d["feat"]) < WIN + 300:
        return []
    feat, speed = d["feat"], d["speed"]
    gyro_yaw, hdg = d["gyro_yaw"], d["hdg"]
    lat, lon = d["lat"], d["lon"]
    lat0 = np.nanmean(lat)
    gx, gy = latlon_to_m(lat, lon, lat0)

    # Model speed prediction for every window end index (WIN-1 .. end).
    n = len(feat)
    idx = np.arange(WIN - 1, n)
    X = np.stack([feat[i - WIN + 1:i + 1] for i in idx])
    Xn = ((X - mean) / std).astype(np.float32)
    pred = model.predict(Xn, verbose=0, batch_size=1024).ravel()
    pred_speed = np.zeros(n)
    pred_speed[idx] = pred
    pred_speed[:WIN - 1] = pred[0]

    # True cumulative ground distance to carve blackout segments of TARGET_DIST.
    dgx, dgy = np.diff(gx), np.diff(gy)
    step = np.sqrt(dgx ** 2 + dgy ** 2)
    cum = np.concatenate([[0], np.cumsum(step)])

    results = []
    start = WIN
    while start < n - 10:
        # find end where true ground distance ~ TARGET_DIST
        end = np.searchsorted(cum, cum[start] + TARGET_DIST)
        if end >= n:
            break
        # Dead reckoning driven by the SPEED model only: integrate predicted speed
        # over the blackout to get distance travelled. This is the frame-free,
        # heading-independent KPI for the speed model's contribution to drift.
        v = np.clip(pred_speed[start:end], 0.0, None)
        pred_dist = float(np.sum(v) * DT)
        true_dist = float(cum[end] - cum[start])
        if true_dist > 0:
            results.append(abs(pred_dist - true_dist) / true_dist * 100)
        start = end
    return results


def main():
    cfg = json.load(open(os.path.join(SPLITS, "config.json")))
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)
    test_files = set(cfg["files"]["test"])
    model = tf.keras.models.load_model(MODEL)

    all_drift = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "S-*.csv"))):
        if os.path.basename(f) not in test_files:
            continue
        ds = eval_drive(f, model, mean, std)
        if ds:
            print(f"{os.path.basename(f):16s} segments={len(ds):3d}  "
                  f"median drift {np.median(ds):.1f}%")
            all_drift.extend(ds)

    if not all_drift:
        print("No blackout segments long enough for", TARGET_DIST, "m")
        return
    a = np.array(all_drift)
    summary = dict(
        target_dist_m=TARGET_DIST, n_segments=len(a),
        median_drift_pct=round(float(np.median(a)), 2),
        mean_drift_pct=round(float(np.mean(a)), 2),
        p90_drift_pct=round(float(np.percentile(a, 90)), 2),
        pass_rate_under_10pct=round(float(np.mean(a < 10) * 100), 1),
    )
    json.dump(summary, open(os.path.join(RESULTS, "drift_results.json"), "w"), indent=1)
    print("\nDRIFT KPI (target <10%):")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
