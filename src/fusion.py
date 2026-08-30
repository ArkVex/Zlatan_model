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
  D. TRUE speed  + TRUE heading    -> the method's own floor (anchors C)
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

from paths import ROOT as HERE, RESULTS
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


def true_heading_from_path(gx, gy):
    """Per-sample true heading (rad, clockwise from north) from the travelled path.

    Two earlier attempts got this wrong, and variant D is what caught both -- true speed along a
    true heading must retrace the path, so a large D means the heading array is not an oracle:

    * gps_course() uses a fixed 10-SAMPLE baseline. That is fine for seeding one heading, but
      IO-VNBD's GPS position is heavily quantised -- 99% of samples carry zero position delta while
      71% of those are genuinely moving. Whenever the window spans no change, arctan2(0, 0) returns
      0, i.e. due north. Distance came out right and direction was ~90% wrong.
    * A distance baseline (walk back until N metres of travel) fails differently: on a cumulative
      path that is a step function, the search lands inconsistently across jumps. Measured worse.

    What works is the literal definition: the bearing of each actual position increment, carried
    forward through the flat runs where the receiver simply did not report a new fix. Yields a
    method floor of about 5% over a 1 km segment, which is GPS quantisation plus integration error.
    """
    de = np.zeros_like(gx)
    dn = np.zeros_like(gy)
    de[1:] = np.diff(gx)
    dn[1:] = np.diff(gy)
    heading = np.arctan2(de, dn)
    moved = np.hypot(de, dn) > 1e-6
    if not moved.any():
        return np.zeros_like(gx)
    # Carry the last real bearing through samples where the position did not change.
    idx = np.maximum.accumulate(np.where(moved, np.arange(len(heading)), 0))
    return heading[idx]


def calibrate_gain(course, gyro_yaw, speed):
    """Least-squares gain k so that d(course) ~= k * gyro_yaw * dt, on moving samples
    (no intercept -> assumes zero gyro bias)."""
    dcourse = np.diff(np.unwrap(course))
    g = gyro_yaw[1:] * DT
    move = speed[1:] > 3.0
    g, dcourse = g[move], dcourse[move]
    denom = float(np.sum(g * g))
    return float(np.sum(g * dcourse) / denom) if denom > 1e-9 else 0.0


def calibrate_gain_bias(course, gyro_yaw, speed):
    """Fit d(course)/step ~= k * gyro_yaw * DT + c0 on moving samples (WITH intercept).

    The intercept c0 is the constant per-step heading error from gyro bias. Propagating
    heading as h += k*gyro*DT + c0 removes that bias during a blackout. Returns (k, c0).
    """
    dcourse = np.diff(np.unwrap(course))
    g = gyro_yaw[1:] * DT
    move = speed[1:] > 3.0
    g, dcourse = g[move], dcourse[move]
    if len(g) < 10:
        return calibrate_gain(course, gyro_yaw, speed), 0.0
    A = np.column_stack([g, np.ones_like(g)])   # [gyro*dt, 1]
    (k, c0), *_ = np.linalg.lstsq(A, dcourse, rcond=None)
    return float(k), float(c0)


def dead_reckon(px, py, h, speeds, yaws, k, c0=0.0):
    """Integrate a segment with gyro-propagated heading: returns end (x, y).
    c0 is the per-step gyro-bias correction (0 = uncorrected)."""
    for v, w in zip(speeds, yaws):
        h += k * w * DT + c0
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
    # Separate oracle for variants C/D: gps_course is a seed, not a per-sample heading (see
    # true_heading_from_path). D is what proves it -- true speed along a true heading must retrace
    # the path, so a large D means the heading array is not an oracle.
    true_heading = true_heading_from_path(gx, gy)
    k = calibrate_gain(course, gyro_yaw, tspeed)          # device-frame gyro yaw axis

    # Correct turn rate = angular velocity projected onto the vertical (gravity) axis.
    # The vehicle turns about vertical; the phone's device-yaw axis only matches it when the
    # phone is flat. For a tilted mount, this projection is the physically correct yaw rate.
    up = d["grav"] / (np.linalg.norm(d["grav"], axis=1, keepdims=True) + 1e-6)
    gyro_vert = np.sum(d["gyro"] * up, axis=1)
    kv, c0v = calibrate_gain_bias(course, gyro_vert, tspeed)  # gain + bias on the vertical axis

    step = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
    cum = np.concatenate([[0], np.cumsum(step)])

    out = {"A": [], "B": [], "C": [], "D": [], "G": [], "H": [], "dist_true": [], "dist_model": []}
    start = WIN
    while start < n - 10:
        end = np.searchsorted(cum, cum[start] + TARGET_DIST)
        if end >= n:
            break
        true_dist = float(cum[end] - cum[start])
        if true_dist > 0:
            seg = slice(start, end)
            gyaw = gyro_yaw[seg]
            # Seed heading from the GPS course at blackout start. (A 3 s circular-mean seed
            # was tested and made drift much worse, so the single-sample seed is kept.)
            h0 = course[start]
            gx0, gy0 = gx[start], gy[start]
            # A: model speed + gyro heading -> the REAL system's 2D drift (gyro-only).
            ax, ay = dead_reckon(gx0, gy0, h0, mspeed[seg], gyaw, k)
            out["A"].append(np.hypot(ax - gx[end], ay - gy[end]) / true_dist * 100)
            # B: TRUE speed + the same gyro-heading pipeline. A - B isolates the speed
            #    contribution; B is the drift floor from gyro-heading error alone.
            bx, by = dead_reckon(gx0, gy0, h0, tspeed[seg], gyaw, k)
            out["B"].append(np.hypot(bx - gx[end], by - gy[end]) / true_dist * 100)
            # G / H: same as A / B but heading from the VERTICAL-AXIS (gravity-projected)
            #        gyro rate + bias correction. G = improved real system, H = new floor.
            gvseg = gyro_vert[seg]
            gxx, gyy = dead_reckon(gx0, gy0, h0, mspeed[seg], gvseg, kv, c0v)
            out["G"].append(np.hypot(gxx - gx[end], gyy - gy[end]) / true_dist * 100)
            hxx, hyy = dead_reckon(gx0, gy0, h0, tspeed[seg], gvseg, kv, c0v)
            out["H"].append(np.hypot(hxx - gx[end], hyy - gy[end]) / true_dist * 100)
            # C: model speed + TRUE heading. The mirror of B, and the number the whole
            #    heading work plan is sized against: what the CURRENT model would score if
            #    heading were solved outright. B says a perfect speed oracle does not help;
            #    C says how much a perfect heading oracle does. Near the along-track drift
            #    means heading is the entire gap; well above it means the speed model still
            #    needs substantial work and the schedule roughly doubles.
            cxx, cyy = dead_reckon_headings(gx0, gy0, mspeed[seg], true_heading[seg])
            out["C"].append(np.hypot(cxx - gx[end], cyy - gy[end]) / true_dist * 100)
            # D: TRUE speed + TRUE heading. The floor of the whole method -- everything left
            #    here is GPS-course noise and integration error, not anything we can fix.
            #    Without it, C is unanchored: a C of 8% means little if D is already 7%.
            dxx, dyy = dead_reckon_headings(gx0, gy0, tspeed[seg], true_heading[seg])
            out["D"].append(np.hypot(dxx - gx[end], dyy - gy[end]) / true_dist * 100)
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

    agg = {"A": [], "B": [], "C": [], "D": [], "G": [], "H": [], "dist_true": [], "dist_model": []}
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
        C_model_speed_true_heading=summarize(agg["C"]),
        D_true_speed_true_heading=summarize(agg["D"]),
        G_model_speed_biascorr_heading=summarize(agg["G"]),
        H_true_speed_biascorr_heading=summarize(agg["H"]),
        along_track_distance_drift=summarize(list(dist_drift)),
    )
    json.dump(result, open(os.path.join(RESULTS, "fusion_drift.json"), "w"), indent=1)
    print("2D POSITION DRIFT over 1 km blackouts (target < 10%):")
    print("  A  model speed + gyro heading (real, no bias fix): ", result["A_model_speed_gyro_heading"])
    print("  B  TRUE  speed + gyro heading (floor, no bias fix):", result["B_true_speed_gyro_heading"])
    print("  C  model speed + TRUE  heading (heading solved):    ", result["C_model_speed_true_heading"])
    print("  D  TRUE  speed + TRUE  heading (method floor):      ", result["D_true_speed_true_heading"])
    print("  G  model speed + VERTICAL-AXIS gyro (real):        ", result["G_model_speed_biascorr_heading"])
    print("  H  TRUE  speed + VERTICAL-AXIS gyro (new floor):   ", result["H_true_speed_biascorr_heading"])
    print("  along-track distance drift (speed only):           ", result["along_track_distance_drift"])
    print("\n  Interpretation: A->G shows the bias-correction gain on the real system;")
    print("  B->H shows how much bias correction lowers the heading floor.")
    print("  A vs B: perfect SPEED changes little -> speed is not the bottleneck.")
    print("  A vs C: perfect HEADING is the ceiling available from the heading work plan.")
    print("  C vs D: what is left after heading is solved, i.e. the speed model's real cost.")
    print("  D:      the method's own floor -- GPS-course noise and integration error.")


if __name__ == "__main__":
    main()
