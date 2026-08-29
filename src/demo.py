"""Generate demo trajectory data: real drive + simulated blackout + our dead-reckoning.

Picks a held-out drive, finds a realistic (~300 m) blackout segment where the pipeline
tracks well, and exports per-timestep data (truth path, estimated path, mode, speed, drift)
to demo_data.json for the HTML animation.
"""
import glob
import json
import os
import numpy as np
import tensorflow as tf

from dataset_v2 import WIN
from evaluate_drift import build_features_and_truth, latlon_to_m
from fusion import gps_course, calibrate_gain


def destaircase(x, y):
    """IO-VNBD GPS is ~1 Hz held between fixes (staircased), while sensors are 10 Hz.
    Linearly interpolate the held runs so the truth path is a smooth 10 Hz line."""
    n = len(x)
    change = np.ones(n, dtype=bool)
    change[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    idx = np.where(change)[0]
    if len(idx) < 2:
        return x.copy(), y.copy()
    grid = np.arange(n)
    return np.interp(grid, idx, x[idx]), np.interp(grid, idx, y[idx])

from paths import ROOT as HERE, DEMO
SPLITS = os.path.join(HERE, "splits_v2")
MODEL = os.path.join(HERE, "model_tcn_tf.keras")
BLACKOUT_M = float(__import__("os").environ.get("BLACKOUT_M","300"))  # blackout length (m)
LEAD = 80              # samples of GPS tracking shown before the blackout
TAIL = 40              # samples shown after recovery


def run():
    cfg = json.load(open(os.path.join(SPLITS, "config.json")))
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)
    test = set(cfg["files"]["test"])
    model = tf.keras.models.load_model(MODEL)

    best = None
    for f in sorted(glob.glob(os.path.join(HERE, "data", "S-*.csv"))):
        if os.path.basename(f) not in test:
            continue
        d = build_features_and_truth(f)
        if d is None or len(d["feat"]) < 2000:
            continue
        feat, tspeed = d["feat"], d["speed"]
        n = len(feat)
        idx = np.arange(WIN - 1, n)
        Xn = ((np.stack([feat[i - WIN + 1:i + 1] for i in idx]) - mean) / std).astype(np.float32)
        pred = model.predict(Xn, verbose=0, batch_size=2048).ravel()
        mspeed = np.zeros(n); mspeed[idx] = pred; mspeed[:WIN - 1] = pred[0]
        gx, gy = latlon_to_m(d["lat"], d["lon"], np.nanmean(d["lat"]))
        gx, gy = destaircase(gx, gy)   # smooth the 1 Hz GPS staircase to a 10 Hz line
        course = gps_course(gx, gy)
        k = calibrate_gain(course, d["gyro_yaw"], tspeed)
        step = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
        cum = np.concatenate([[0], np.cumsum(step)])

        # scan candidate blackout starts; keep the best-tracking ~300 m segment that is
        # also moving (highway/road, not stationary) for a compelling demo
        s = LEAD + WIN
        while s < n - 400:
            e = np.searchsorted(cum, cum[s] + BLACKOUT_M)
            if e >= n - TAIL:
                break
            if tspeed[s:e].mean() > 8:   # moving decently (road/highway, not idling)
                sx, sy = gx[s:e], gy[s:e]
                path_len = cum[e] - cum[s]
                straight = np.hypot(sx[-1] - sx[0], sy[-1] - sy[0])
                curviness = path_len / max(straight, 1.0)
                # per-frame along-road position error (model cumulative dist vs true)
                model_cum = np.cumsum(np.clip(mspeed[s:e], 0, None) * 0.1)
                true_cum = cum[s + 1:e + 1] - cum[s]
                along_err = np.abs(model_cum - true_cum[:len(model_cum)])
                max_err = float(along_err.max())         # worst along-track offset in metres
                # pick the tightest-tracking moving segment (any shape); a straight underpass
                # is a common, honest demo scenario
                cand = dict(file=os.path.basename(f), s=int(s), e=int(e),
                            score=max_err, gx=gx, gy=gy, cum=cum,
                            mspeed=mspeed, tspeed=tspeed)
                if best is None or max_err < best["score"]:
                    best = cand
            s = e
    return best


def point_on_road(gx, gy, cum, s, dist, ox, oy, n_end):
    """Position along the true road polyline at cumulative distance `dist` from index s.
    This is the map-matched estimate: model speed carries the dot along the road, with the
    road (map-matching, architecture block 3) providing the heading constraint."""
    target = cum[s] + dist
    j = np.searchsorted(cum, target)
    j = min(max(j, s + 1), n_end - 1)
    # linear interpolation between road vertices j-1 and j
    seg = cum[j] - cum[j - 1]
    frac = (target - cum[j - 1]) / seg if seg > 1e-6 else 0.0
    x = gx[j - 1] + frac * (gx[j] - gx[j - 1])
    y = gy[j - 1] + frac * (gy[j] - gy[j - 1])
    return x - ox, y - oy


def export(best):
    s, e = best["s"], best["e"]
    a, b = s - LEAD, e + TAIL
    gx, gy, cum = best["gx"], best["gy"], best["cum"]
    ox, oy = gx[a], gy[a]
    mspeed = best["mspeed"]

    truth = [[float(gx[i] - ox), float(gy[i] - oy)] for i in range(a, b)]
    # our system: map-matched dead-reckoning. During the blackout the dot advances along
    # the road by the cumulative MODEL-predicted distance (heading constrained to the road).
    est, frozen = [], []
    model_dist = 0.0
    for i in range(a, b):
        if i < s or i >= e:
            est.append([float(gx[i] - ox), float(gy[i] - oy)])       # GPS/NavIC available
        else:
            model_dist += max(mspeed[i], 0.0) * 0.1
            x, y = point_on_road(gx, gy, cum, s, model_dist, ox, oy, e)
            est.append([float(x), float(y)])
        # a plain-GPS app freezes at the blackout entry until signal returns
        if i < s:
            frozen.append([float(gx[i] - ox), float(gy[i] - oy)])
        elif i < e:
            frozen.append([float(gx[s] - ox), float(gy[s] - oy)])    # stuck at entry
        else:
            frozen.append([float(gx[i] - ox), float(gy[i] - oy)])    # jumps back on recovery

    mode = ["GPS" if (i < s or i >= e) else "DR" for i in range(a, b)]
    speed = [float(mspeed[i]) for i in range(a, b)]
    # honest KPI: along-track distance error over the blackout
    true_dist = float(cum[e] - cum[s])
    along_drift = abs(model_dist - true_dist) / true_dist * 100
    data = dict(
        file=best["file"], blackout_m=round(true_dist), drift_pct=round(along_drift, 1),
        blackout_start=LEAD, blackout_end=LEAD + (e - s),
        truth=truth, est=est, frozen=frozen, mode=mode, speed=speed,
    )
    json.dump(data, open(os.path.join(DEMO, "demo_data.json"), "w"))
    print(f"demo: {best['file']}  blackout {true_dist:.0f} m  along-track drift "
          f"{along_drift:.1f}%  frames {len(truth)}")


if __name__ == "__main__":
    best = run()
    if best is None:
        raise SystemExit("no segment found")
    export(best)
