"""Turn a GPS-live ride into labelled training windows for the target vehicle.

Why this exists
---------------
`engine_running_negatives.py` extracts one missing CLASS (stopped, engine running). This extracts
the whole ride, because the 2026-09-02 session showed the problem is bigger than a missing class.

On 238 seconds of continuous motorbike riding with GNSS live throughout, the shipped `tcn_v2`
correlated with true speed at **r = 0.17, r-squared = 0.03**. Not a scale error, not a lag — checked
at every offset from -3 s to +3 s and after 3 s smoothing, and it never rose above 0.18. The model
explains three percent of the variance in speed. Its regression against truth is
`v_model = 0.28 * v_gnss + 4.68`: it over-predicts by 3.7 m/s while crawling and under-predicts by
1.1 m/s above 7 m/s, which is what a model does when it has almost no signal and falls back on its
own mean.

The same model reports R-squared 0.66 on the IO-VNBD test set. IO-VNBD is cars. This is a
motorbike — different vibration spectrum, different mount, and the machine leans into turns. **The
model does not transfer to a two-wheeler**, and no amount of extra negatives fixes that, because the
problem is not a missing class but a missing domain.

So: collect labelled speed from the vehicle we actually intend to run on, and fine-tune on it.

What counts as a usable label
-----------------------------
Only ticks where GNSS was genuinely live. During dead reckoning `v_gnss` is a stale value from the
last fix and would teach the model that whatever the IMU was doing means whatever the vehicle was
doing minutes ago.

Usage:
    python src/ride_training_data.py trip_*.csv tel_*.csv -o rides/ride_<id>.npz
"""
import argparse
import csv
import os

import numpy as np

WIN = 50            # samples per window at 10 Hz, matching dataset_v2
RATE_HZ = 10.0
AIDED_MODES = {"GNSS", "NAVIC"}

# Reject a window whose label moved more than this within the window. The label is the speed at the
# window's END, and a hard acceleration means the earlier samples describe a different speed.
MAX_LABEL_SPREAD_MPS = 3.0


def imu_rows(path):
    keep = []
    with open(path) as f:
        for line in f:
            if not line.startswith("IMU"):
                continue
            p = line.strip().split(",")
            if len(p) < 11:
                continue
            try:
                keep.append([float(x) for x in p[1:11]])
            except ValueError:
                continue
    if not keep:
        raise SystemExit(f"{path}: no IMU rows")
    return np.asarray(keep)


def labels(tel_path):
    """[(t_nanos, speed_mps)] for ticks where GNSS was actually live."""
    with open(tel_path) as f:
        first = f.readline()
        if not first.startswith("#schema"):
            f.seek(0)
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get("mode") not in AIDED_MODES:
            continue
        try:
            out.append((int(float(r["t_nanos"])), float(r["v_gnss_mps"])))
        except (ValueError, KeyError):
            continue
    return out


def features(acc, grav, gyro):
    """The 7 channels FeatureExtractor.kt emits, in its order."""
    lin = acc - grav
    gmag = np.linalg.norm(grav, axis=1, keepdims=True) + 1e-6
    ghat = grav / gmag
    a_vert = np.sum(lin * ghat, axis=1)
    horiz = lin - a_vert[:, None] * ghat
    return np.stack([
        np.linalg.norm(horiz, axis=1),      # a_horiz
        a_vert,                             # a_vert
        np.linalg.norm(lin, axis=1),        # a_lin_mag
        gyro[:, 0], gyro[:, 1], gyro[:, 2], # gyr x/y/z
        np.linalg.norm(gyro, axis=1),       # gyro_mag
    ], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trip"); ap.add_argument("telemetry")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    lab = labels(args.telemetry)
    if not lab:
        raise SystemExit("no GNSS-live ticks — this ride carries no usable labels")
    print(f"GNSS-live labelled ticks: {len(lab)} ({len(lab) / RATE_HZ / 60:.1f} min)")

    d = imu_rows(args.trip)
    t = d[:, 0]
    print(f"trip trace: {len(d)} IMU rows, {(t[-1] - t[0]) / 1e9 / 60:.1f} min")

    lt = np.array([x[0] for x in lab], dtype=np.int64)
    lv = np.array([x[1] for x in lab], dtype=np.float32)
    if lt[-1] < t[0] or lt[0] > t[-1]:
        raise SystemExit("trip and telemetry do not overlap in time — different sessions?")

    # Decimate the raw trace onto the model's 10 Hz grid, mean-binned like Decimator does.
    grid = np.arange(t[0], t[-1], int(1e9 / RATE_HZ))
    idx = np.searchsorted(t, grid)
    binned = np.empty((len(grid) - 1, 9), np.float32)
    for i in range(len(grid) - 1):
        a, b = idx[i], idx[i + 1]
        binned[i] = d[a:b, 1:10].mean(axis=0) if b > a else d[min(a, len(d) - 1), 1:10]
    feat = features(binned[:, 0:3], binned[:, 3:6], binned[:, 6:9])
    gt = grid[:-1]

    # Label each grid point from the nearest GNSS-live tick, and refuse anything not covered.
    near = np.searchsorted(lt, gt).clip(0, len(lt) - 1)
    gap = np.abs(lt[near] - gt) / 1e9
    speed = lv[near]
    covered = gap <= 0.5

    X, y, dropped_spread, dropped_gap = [], [], 0, 0
    for i in range(len(feat) - WIN):
        sl = slice(i, i + WIN)
        if not covered[sl].all():
            dropped_gap += 1
            continue
        s = speed[sl]
        if s.max() - s.min() > MAX_LABEL_SPREAD_MPS:
            dropped_spread += 1
            continue
        X.append(feat[sl]); y.append(s[-1])

    if not X:
        raise SystemExit("no usable windows — was GNSS live for at least 5 s at a stretch?")
    X = np.stack(X); y = np.asarray(y, np.float32)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"\nwrote {len(X)} windows -> {args.out}")
    print(f"  dropped: {dropped_gap} (no GNSS cover), {dropped_spread} (label moved >{MAX_LABEL_SPREAD_MPS} m/s in-window)")
    print(f"  label speed: min {y.min():.2f}  p50 {np.median(y):.2f}  max {y.max():.2f} m/s")
    print(f"  stopped windows (<0.5 m/s): {(y < 0.5).sum()}  — these are the negatives v2 never had")


if __name__ == "__main__":
    main()
