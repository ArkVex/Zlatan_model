"""Part C: dead-reckoning + fusion reference, measuring REAL 2D position drift.

Pipeline per drive:
  - model predicts forward speed each tick (TCN)
  - heading is propagated from the gyro yaw rate, seeded from GPS course at blackout start
  - the gyro->heading gain k is CALIBRATED on GPS-good data (auto-handles sign/scale/units)
  - during a simulated GNSS blackout we integrate position open-loop (this is exactly what
    a fused EKF does during a blackout — no GPS updates arrive, so DR == the filter)
  - at blackout end we compare the dead-reckoned position to GPS truth: drift = err / distance

We measure THREE variants to decompose the error:
  A. model speed + gyro heading    -> the real system
  B. TRUE speed  + gyro heading    -> isolates HEADING error
  C. model speed + TRUE heading    -> isolates SPEED error
Comparing A/B/C tells us whether to invest in a heading model or a better speed model.

Usage: MODEL_FILE=model_tcn_tf.keras python fusion.py
"""
import glob
import json
import os
import numpy as np
import tensorflow as tf

from dataset_v2 import WIN
from evaluate_drift import build_features_and_truth, latlon_to_m, TARGET_DIST, DT
from heading import magnetic_heading, circular_offset, complementary_heading

HERE = os.path.dirname(__file__)
SPLITS = os.path.join(HERE, "splits_v2")
MODEL = os.path.join(HERE, os.environ.get("MODEL_FILE", "model_tcn_tf.keras"))


def gps_course(gx, gy, w=10):
    """Heading (rad, clockwise from north) from GPS position deltas over a w-sample
    (~1 s) baseline. A 1-sample delta is far too noisy at low speed (tiny dx,dy ->
    random angle); a 1 s baseline gives a stable true heading. East=x, North=y."""
    de = np.zeros_like(gx)
    dn = np.zeros_like(gy)
    de[w:] = gx[w:] - gx[:-w]
    dn[w:] = gy[w:] - gy[:-w]
    de[:w] = gx[w] - gx[0] if len(gx) > w else 0.0
    dn[:w] = gy[w] - gy[0] if len(gy) > w else 0.0
    return np.arctan2(de, dn)   # atan2(east, north) = clockwise-from-north


def calibrate_gain(course, gyro_yaw, speed):
    """Least-squares gain k so that d(course) ~= k * gyro_yaw * dt, on moving samples."""
    dcourse = np.diff(np.unwrap(course))
    g = gyro_yaw[1:] * DT
    move = speed[1:] > 3.0
    g, dcourse = g[move], dcourse[move]
    denom = float(np.sum(g * g))
    return float(np.sum(g * dcourse) / denom) if denom > 1e-9 else 0.0


def dead_reckon(px, py, h, speeds, yaws, k):
    """Integrate a segment with gyro-propagated heading: returns end (x, y)."""
    for v, w in zip(speeds, yaws):
        h += k * w * DT
        px += max(v, 0.0) * np.sin(h) * DT
        py += max(v, 0.0) * np.cos(h) * DT
    return px, py


def dead_reckon_headings(px, py, speeds, headings):
    """Integrate a segment using a precomputed per-sample heading array."""
    for v, h in zip(speeds, headings):
        px += max(v, 0.0) * np.sin(h) * DT
        py += max(v, 0.0) * np.cos(h) * DT
    return px, py


def eval_drive(path, model, mean, std):
    d = build_features_and_truth(path)
    if d is None or len(d["feat"]) < WIN + 300:
        return None
    feat, tspeed = d["feat"], d["speed"]
    gyro_yaw, lat, lon = d["gyro_yaw"], d["lat"], d["lon"]
    n = len(feat)

    idx = np.arange(WIN - 1, n)
    Xn = ((np.stack([feat[i - WIN + 1:i + 1] for i in idx]) - mean) / std).astype(np.float32)
    pred = model.predict(Xn, verbose=0, batch_size=1024).ravel()
    mspeed = np.zeros(n)
    mspeed[idx] = pred
    mspeed[:WIN - 1] = pred[0]

    gx, gy = latlon_to_m(lat, lon, np.nanmean(lat))
    course = gps_course(gx, gy)
    k = calibrate_gain(course, gyro_yaw, tspeed)

    # Magnetometer heading: per-sample device-frame heading, offset-calibrated to GPS course.
    mag_head = magnetic_heading(d["grav"], d["mag"])
    have_mag = np.isfinite(mag_head).any()
    if have_mag:
        offset = circular_offset(course, mag_head, tspeed > 3.0)
        mag_head_abs = mag_head + offset
    else:
        mag_head_abs = np.full(n, np.nan)

    step = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
    cum = np.concatenate([[0], np.cumsum(step)])

    out = {"A": [], "B": [], "E": [], "F": [], "dist_true": [], "dist_model": []}
    start = WIN
    while start < n - 10:
        end = np.searchsorted(cum, cum[start] + TARGET_DIST)
        if end >= n:
            break
        true_dist = float(cum[end] - cum[start])
        if true_dist > 0:
            seg = slice(start, end)
            gyaw = gyro_yaw[seg]
            h0 = course[start]
            gx0, gy0 = gx[start], gy[start]
            # A: model speed + gyro heading -> the REAL system's 2D drift (gyro-only).
            ax, ay = dead_reckon(gx0, gy0, h0, mspeed[seg], gyaw, k)
            out["A"].append(np.hypot(ax - gx[end], ay - gy[end]) / true_dist * 100)
            # B: TRUE speed + the same gyro-heading pipeline. A - B isolates the speed
            #    contribution; B is the drift floor from gyro-heading error alone.
            bx, by = dead_reckon(gx0, gy0, h0, tspeed[seg], gyaw, k)
            out["B"].append(np.hypot(bx - gx[end], by - gy[end]) / true_dist * 100)
            # E / F: same as A / B but heading from the gyro+magnetometer complementary
            #        filter instead of gyro alone. E = improved real system, F = new floor.
            if have_mag:
                comp = complementary_heading(h0, gyaw, k, mag_head_abs[seg])
                ex, ey = dead_reckon_headings(gx0, gy0, mspeed[seg], comp)
                out["E"].append(np.hypot(ex - gx[end], ey - gy[end]) / true_dist * 100)
                fx, fy = dead_reckon_headings(gx0, gy0, tspeed[seg], comp)
                out["F"].append(np.hypot(fx - gx[end], fy - gy[end]) / true_dist * 100)
            # Along-track distance drift (speed-only, heading-free), for reference.
            out["dist_true"].append(true_dist)
            out["dist_model"].append(float(np.sum(np.clip(mspeed[seg], 0, None)) * DT))
        start = end
    return out


def summarize(a):
    a = np.array(a)
    if len(a) == 0:
        return None
    return dict(n=len(a), median=round(float(np.median(a)), 2),
                mean=round(float(np.mean(a)), 2),
                p90=round(float(np.percentile(a, 90)), 2),
                pass_under_10=round(float(np.mean(a < 10) * 100), 1))


def main():
    cfg = json.load(open(os.path.join(SPLITS, "config.json")))
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)
    test = set(cfg["files"]["test"])
    model = tf.keras.models.load_model(MODEL)

    agg = {"A": [], "B": [], "E": [], "F": [], "dist_true": [], "dist_model": []}
    for f in sorted(glob.glob(os.path.join(HERE, "data", "S-*.csv"))):
        if os.path.basename(f) not in test:
            continue
        r = eval_drive(f, model, mean, std)
        if r:
            for kk in agg:
                agg[kk] += r[kk]

    dt = np.array(agg["dist_true"]); dm = np.array(agg["dist_model"])
    dist_drift = np.abs(dm - dt) / dt * 100
    result = dict(
        model=os.path.basename(MODEL), target_dist_m=TARGET_DIST, metric="2D position drift %",
        A_model_speed_gyro_heading=summarize(agg["A"]),
        B_true_speed_gyro_heading=summarize(agg["B"]),
        E_model_speed_magcomp_heading=summarize(agg["E"]),
        F_true_speed_magcomp_heading=summarize(agg["F"]),
        along_track_distance_drift=summarize(list(dist_drift)),
    )
    json.dump(result, open(os.path.join(HERE, "fusion_drift.json"), "w"), indent=1)
    print("2D POSITION DRIFT over 1 km blackouts (target < 10%):")
    print("  A  model speed + GYRO heading (real, gyro-only): ", result["A_model_speed_gyro_heading"])
    print("  B  TRUE  speed + GYRO heading (gyro floor):      ", result["B_true_speed_gyro_heading"])
    print("  E  model speed + MAG+GYRO heading (real, filter):", result["E_model_speed_magcomp_heading"])
    print("  F  TRUE  speed + MAG+GYRO heading (new floor):   ", result["F_true_speed_magcomp_heading"])
    print("  along-track distance drift (speed only):         ", result["along_track_distance_drift"])
    print("\n  Interpretation: A->E shows the heading-filter gain; B->F shows how much the")
    print("  filter lowers the heading floor. If E << A, the magnetometer filter works.")


if __name__ == "__main__":
    main()
