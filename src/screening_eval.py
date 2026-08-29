"""W1 evaluation harness + Gap-6 screening plots (per Aneesh/SIH-IDR-architecture.md).

Adds what the drift eval was missing:
  - the two naive baselines judges will ask for:
      * constant-velocity: hold the last known GNSS speed through the outage
      * double-integration: integrate horizontal linear acceleration in the world frame
        (the classic "why naive INS fails" reference)
  - outage-duration buckets {10, 30, 60, 180 s} instead of a single 1 km slice
  - along-track / cross-track decomposition of the 2D endpoint error
  - mean SIGNED bias (the error type that kills the drift KPI), not just magnitude
  - the four screening plots (trajectory overlay, drift-vs-duration, along/cross
    time series, drift CDF with the 10% target)

Outputs: results/screening/*.png + results/screening/summary.json
Model under test: the shipped TCN v2 (exports/norm.json constants).
"""
import glob
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import paths
from dataset_v2 import _resolve, WIN
from demo import destaircase
from fusion import gps_course, calibrate_gain, dead_reckon

DT = 0.1
BUCKETS_S = [10, 30, 60, 180]
OUT = os.path.join(paths.RESULTS, "screening")
MODEL = os.path.join(paths.EXPORTS, "engine_tcn_v2_2026-08-30_ft1.tflite")
NORM = os.path.join(paths.EXPORTS, "norm.json")


# ---------------------------------------------------------------- data loading
def load_drive(path):
    """Everything the harness needs from one drive, at 10 Hz."""
    import pandas as pd
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    c = _resolve(df.columns)
    need = ["speed", "ax", "ay", "az", "grx", "gry", "grz", "wy", "wp", "wr"]
    if any(c[k] is None for k in need):
        return None
    lat_c = [x for x in df.columns if "GPS LATITUDE" in x.upper()]
    lon_c = [x for x in df.columns if "GPS LONGITUDE" in x.upper()]
    if not (lat_c and lon_c):
        return None

    def num(col):
        s = df[col]
        if getattr(s, "ndim", 1) > 1:
            s = s.iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").to_numpy(np.float64).ravel()

    cols = {k: num(c[k]) for k in need}
    lat, lon = num(lat_c[0]), num(lon_c[0])
    m = min(min(len(v) for v in cols.values()), len(lat), len(lon))
    cols = {k: v[:m] for k, v in cols.items()}
    lat, lon = lat[:m], lon[:m]

    acc = np.stack([cols["ax"], cols["ay"], cols["az"]], 1)
    grav = np.stack([cols["grx"], cols["gry"], cols["grz"]], 1)
    gyro = np.stack([cols["wy"], cols["wp"], cols["wr"]], 1)
    speed = cols["speed"]

    a_lin = acc - grav
    g_hat = grav / (np.linalg.norm(grav, axis=1, keepdims=True) + 1e-6)
    a_vert = np.sum(a_lin * g_hat, axis=1)
    a_vert_vec = a_vert[:, None] * g_hat
    a_h_vec = a_lin - a_vert_vec                       # horizontal accel vector (device-ish frame)
    a_horiz = np.linalg.norm(a_h_vec, axis=1)
    feat = np.stack([a_horiz, a_vert, np.linalg.norm(a_lin, axis=1),
                     gyro[:, 0], gyro[:, 1], gyro[:, 2],
                     np.linalg.norm(gyro, axis=1)], 1).astype(np.float32)

    good = np.isfinite(feat).all(1) & np.isfinite(speed) & np.isfinite(lat) & np.isfinite(lon)
    feat, speed = feat[good], speed[good]
    lat, lon = lat[good], lon[good]
    gyro_yaw, a_h_mag = gyro[good, 0], a_horiz[good]
    if len(feat) < WIN + 200:
        return None

    lat0 = float(np.nanmean(lat))
    R = 6371000.0
    gx = np.radians(lon) * R * math.cos(math.radians(lat0))
    gy = np.radians(lat) * R
    gx, gy = destaircase(gx, gy)
    course = gps_course(gx, gy)
    k = calibrate_gain(course, gyro_yaw, speed)
    step = np.hypot(np.diff(gx), np.diff(gy))
    cum = np.concatenate([[0], np.cumsum(step)])
    return dict(feat=feat, speed=speed, gyro_yaw=gyro_yaw, a_h=a_h_mag,
                gx=gx, gy=gy, cum=cum, course=course, k=k,
                name=os.path.basename(path))


def model_speeds(d, itp, mean, std):
    feat = d["feat"]
    n = len(feat)
    di, do = itp.get_input_details()[0], itp.get_output_details()[0]
    pred = np.zeros(n)
    for i in range(WIN - 1, n):
        w = ((feat[i - WIN + 1:i + 1] - mean) / std).astype(np.float32)[None]
        itp.set_tensor(di["index"], w)
        itp.invoke()
        pred[i] = itp.get_tensor(do["index"])[0][0]
    pred[:WIN - 1] = pred[WIN - 1]
    return np.clip(pred, 0, None)


# ---------------------------------------------------------------- per-segment eval
def eval_segment(d, pred, s, e):
    """All estimators over one simulated outage [s, e). Returns dict of drifts (%)."""
    true_dist = float(d["cum"][e] - d["cum"][s])
    if true_dist < 5:
        return None
    T = (e - s) * DT
    # our model
    model_dist = float(np.sum(pred[s:e]) * DT)
    # baseline 1: constant velocity (hold GNSS speed at outage entry)
    cv_dist = float(d["speed"][s]) * T
    # baseline 2: double integration of horizontal linear acceleration magnitude
    #   v(t) = v0 + integral(a_horiz) — the naive INS approach; expected to blow up,
    #   which is the pedagogical point (error grows ~t^2..t^3 with bias)
    v = float(d["speed"][s])
    di_dist = 0.0
    for j in range(s, e):
        v = max(v + d["a_h"][j] * DT, 0.0)
        di_dist += v * DT
    # 2D endpoint (model speed + calibrated gyro heading) -> along/cross split
    ex, ey = dead_reckon(d["gx"][s], d["gy"][s], d["course"][s],
                         pred[s:e], d["gyro_yaw"][s:e], d["k"])
    err = np.array([ex - d["gx"][e], ey - d["gy"][e]])
    # travel direction at outage end (unit vector along true displacement)
    disp = np.array([d["gx"][e] - d["gx"][s], d["gy"][e] - d["gy"][s]])
    nrm = np.linalg.norm(disp)
    if nrm < 1:
        along = cross = np.nan
    else:
        u = disp / nrm
        along = float(np.dot(err, u))
        cross = float(err[0] * -u[1] + err[1] * u[0])
    pct = lambda x: abs(x - true_dist) / true_dist * 100
    return dict(true_dist=true_dist,
                model=pct(model_dist), cv=pct(cv_dist), di=pct(di_dist),
                model_signed=(model_dist - true_dist) / true_dist * 100,
                along_pct=abs(along) / true_dist * 100 if np.isfinite(along) else np.nan,
                cross_pct=abs(cross) / true_dist * 100 if np.isfinite(cross) else np.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(os.path.join(paths.SPLITS_V2, "config.json")))
    test_files = [os.path.join(paths.DATA, f) for f in cfg["files"]["test"]]
    norm = json.load(open(NORM))["input_norm"]
    mean, std = np.array(norm["mean"]), np.array(norm["std"])
    itp = tf.lite.Interpreter(model_path=MODEL)
    itp.allocate_tensors()

    drives = []
    for f in test_files:
        if not os.path.exists(f):
            continue
        d = load_drive(f)
        if d is None:
            continue
        d["pred"] = model_speeds(d, itp, mean, std)
        drives.append(d)
        print(f"loaded {d['name']}  ({len(d['feat'])} samples)")

    # segments per duration bucket (non-overlapping, moving)
    results = {b: [] for b in BUCKETS_S}
    for d in drives:
        n = len(d["feat"])
        for b in BUCKETS_S:
            L = int(b / DT)
            s = WIN
            while s + L < n:
                if d["speed"][s:s + L].mean() > 3.0:      # moving segment
                    r = eval_segment(d, d["pred"], s, s + L)
                    if r:
                        results[b].append(r)
                s += L
    summary = {}
    for b in BUCKETS_S:
        rs = results[b]
        if not rs:
            continue
        g = lambda k: np.array([r[k] for r in rs])
        summary[f"{b}s"] = dict(
            n=len(rs),
            model_median=round(float(np.median(g("model"))), 2),
            model_mean=round(float(np.mean(g("model"))), 2),
            model_signed_bias=round(float(np.mean(g("model_signed"))), 2),
            model_pass10=round(float(np.mean(g("model") < 10) * 100), 1),
            cv_median=round(float(np.median(g("cv"))), 2),
            di_median=round(float(np.median(g("di"))), 2),
            along_median=round(float(np.nanmedian(g("along_pct"))), 2),
            cross_median=round(float(np.nanmedian(g("cross_pct"))), 2),
        )
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

    # ---------------- plot 1: trajectory overlay with shaded outage ----------------
    d = max(drives, key=lambda x: len(x["feat"]))
    n = len(d["feat"])
    L = int(60 / DT)
    s = n // 3
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(d["gx"] - d["gx"][0], d["gy"] - d["gy"][0], color="#888", lw=1.2, label="GNSS truth")
    ex, ey = d["gx"][s], d["gy"][s]
    h = d["course"][s]
    xs, ys = [ex], [ey]
    for j in range(s, min(s + L, n - 1)):
        h += d["k"] * d["gyro_yaw"][j] * DT
        ex += d["pred"][j] * math.sin(h) * DT
        ey += d["pred"][j] * math.cos(h) * DT
        xs.append(ex); ys.append(ey)
    ax.plot(np.array(xs) - d["gx"][0], np.array(ys) - d["gy"][0],
            color="#d62728", lw=2, label="dead-reckoned (60 s outage)")
    ax.plot(d["gx"][s] - d["gx"][0], d["gy"][s] - d["gy"][0], "k^", ms=9, label="outage start")
    ax.set_title(f"Trajectory overlay — {d['name']}, 60 s simulated outage")
    ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    ax.axis("equal"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(os.path.join(OUT, "1_trajectory_overlay.png"), dpi=130, bbox_inches="tight")

    # ---------------- plot 2: drift vs outage duration vs baselines ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [b for b in BUCKETS_S if f"{b}s" in summary]
    for key, label, color in [("model_median", "TCN v2 (ours)", "#1f77b4"),
                              ("cv_median", "constant-velocity baseline", "#2ca02c"),
                              ("di_median", "double-integration baseline", "#d62728")]:
        ax.plot(xs, [summary[f"{b}s"][key] for b in xs], "o-", label=label, color=color)
    ax.axhline(10, ls="--", color="k", lw=1, label="10% target")
    ax.set_yscale("log")
    ax.set_xlabel("outage duration (s)"); ax.set_ylabel("median along-track drift (%)")
    ax.set_title("Drift vs outage duration — held-out drives")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.savefig(os.path.join(OUT, "2_drift_vs_duration.png"), dpi=130, bbox_inches="tight")

    # ---------------- plot 3: along/cross decomposition over one outage ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    s = n // 3
    L = int(60 / DT)
    ex, ey = d["gx"][s], d["gy"][s]
    h = d["course"][s]
    t_axis, alongs, crosses = [], [], []
    for j in range(s, min(s + L, n - 1)):
        h += d["k"] * d["gyro_yaw"][j] * DT
        ex += d["pred"][j] * math.sin(h) * DT
        ey += d["pred"][j] * math.cos(h) * DT
        err = np.array([ex - d["gx"][j + 1], ey - d["gy"][j + 1]])
        disp = np.array([d["gx"][j + 1] - d["gx"][s], d["gy"][j + 1] - d["gy"][s]])
        nrm = np.linalg.norm(disp)
        if nrm > 1:
            u = disp / nrm
            alongs.append(float(np.dot(err, u)))
            crosses.append(float(err[0] * -u[1] + err[1] * u[0]))
            t_axis.append((j - s) * DT)
    ax.plot(t_axis, alongs, label="along-track error (m)", color="#1f77b4")
    ax.plot(t_axis, crosses, label="cross-track error (m)", color="#ff7f0e")
    ax.set_xlabel("time into outage (s)"); ax.set_ylabel("error (m)")
    ax.set_title("Error decomposition during a 60 s outage")
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(os.path.join(OUT, "3_along_cross.png"), dpi=130, bbox_inches="tight")

    # ---------------- plot 4: drift CDF with the 10% target ----------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for b, color in zip(BUCKETS_S, ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]):
        vals = np.sort([r["model"] for r in results[b]])
        if len(vals) < 5:
            continue
        ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals) * 100,
                label=f"{b} s outages (n={len(vals)})", color=color)
    ax.axvline(10, ls="--", color="k", lw=1, label="10% target")
    ax.set_xlim(0, 60)
    ax.set_xlabel("along-track drift (%)"); ax.set_ylabel("segments under this drift (%)")
    ax.set_title("Cumulative distribution of drift")
    ax.legend(); ax.grid(alpha=.3)
    fig.savefig(os.path.join(OUT, "4_drift_cdf.png"), dpi=130, bbox_inches="tight")

    print("wrote 4 plots + summary.json ->", OUT)


if __name__ == "__main__":
    main()
