"""V1 increment reformulation: predict SPEED CHANGE, anchor at outage-entry GNSS speed.

Motivation (results/screening/summary.json): the constant-velocity baseline beats the
absolute-speed model up to ~60 s because road speed is autocorrelated and the model's
per-window noise integrates. The model's signed bias is ~0, so the fix is to keep the CV
anchor and let the model supply only the *change*:

    v_hat(t) = v0_at_blackout_entry + sum of predicted 1-second deltas

Model: same TCN architecture, same 50x7 input window; target = speed(t) - speed(t-1s),
standardised. Trained on driving data only (delta of a stationary phone is ~0 anyway).

Evaluates in the same outage buckets {10,30,60,180 s} against the CV baseline and the
absolute-speed model numbers, writing results/v1_delta_eval.json.
"""
import glob
import json
import math
import os
import numpy as np
import tensorflow as tf

import paths
from dataset_v2 import load_file, WIN

DT = 0.1
DELTA_STEPS = 10          # predict speed change over 1 s (10 samples @10 Hz)
BUCKETS_S = [10, 30, 60, 180]
SEED = 42
EPOCHS = 18


def build_tcn(win, c):
    ch, blocks = 64, [1, 2, 4, 8, 16]
    inp = tf.keras.Input((win, c), name="imu_window")
    x = inp
    for d in blocks:
        prev = x
        x = tf.keras.layers.Conv1D(ch, 3, padding="causal", dilation_rate=d, activation="relu")(x)
        x = tf.keras.layers.Conv1D(ch, 3, padding="causal", dilation_rate=d, activation="relu")(x)
        if prev.shape[-1] != ch:
            prev = tf.keras.layers.Conv1D(ch, 1, padding="same")(prev)
        x = tf.keras.layers.Add()([prev, x])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, name="dv")(x)
    return tf.keras.Model(inp, out)


def drive_windows_with_delta(files):
    """Per drive: windows (stride 5) + delta target dv_1s = speed[end] - speed[end-10]."""
    Xs, ys = [], []
    for f in files:
        p = os.path.join(paths.DATA, f)
        r = load_file(p)
        if r[0] is None:
            continue
        X, y = r          # X: [N,50,7] windows at stride 5; y: speed at each window's end
        # window i ends at sample WIN-1 + 5i; 1 s earlier = 2 windows back (stride 5 = 0.5 s)
        if len(y) <= 2:
            continue
        dv = y[2:] - y[:-2]
        Xs.append(X[2:])
        ys.append(dv)
    return np.concatenate(Xs), np.concatenate(ys)


def load_eval_drive(f):
    """Full 10 Hz sequences for outage simulation on a test drive."""
    r = load_file(os.path.join(paths.DATA, f))
    if r[0] is None:
        return None
    X, y = r
    return X, y           # stride-5 windows + speeds at window ends (0.5 s grid)


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    cfg = json.load(open(os.path.join(paths.SPLITS_V2, "config.json")))
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)

    Xtr, dvtr = drive_windows_with_delta(cfg["files"]["train"])
    print(f"train windows {Xtr.shape}, dv std {dvtr.std():.3f} m/s (per 1 s)")
    dv_std = float(dvtr.std()) + 1e-6
    Xn = ((Xtr - mean) / std).astype(np.float32)
    tgt = (dvtr / dv_std).astype(np.float32)

    model = build_tcn(WIN, 7)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(Xn, tgt, batch_size=256, epochs=EPOCHS, shuffle=True, verbose=2)
    model.save(os.path.join(paths.MODELS, "model_tcn_v1delta.keras"))

    # ---- outage-bucket evaluation on held-out drives (0.5 s grid) ----
    results = {b: {"v1": [], "cv": []} for b in BUCKETS_S}
    for f in cfg["files"]["test"]:
        r = load_eval_drive(f)
        if r is None:
            continue
        X, spd = r                     # both on the 0.5 s window grid
        n = len(spd)
        Xn = ((X - mean) / std).astype(np.float32)
        dv_pred = model.predict(Xn, verbose=0, batch_size=1024).ravel() * dv_std
        # dv_pred[i] = predicted speed change over the 1 s ending at grid point i
        GRID_DT = 0.5                  # seconds between grid points (stride 5 @10 Hz)
        for b in BUCKETS_S:
            L = int(b / GRID_DT)
            s = 4                      # need i-2 history for nothing; just start clear of edge
            while s + L < n:
                seg_speed = spd[s:s + L]
                if seg_speed.mean() > 3.0:
                    true_dist = float(np.sum(seg_speed) * GRID_DT)
                    v0 = float(spd[s])
                    # constant velocity baseline
                    cv_dist = v0 * b
                    # V1: v(t) = v0 + cumulative deltas; dv is per-1s, grid is 0.5 s,
                    # so add dv/2 per grid step
                    v = v0
                    v1_dist = 0.0
                    for j in range(s, s + L):
                        v = min(max(v + dv_pred[j] * GRID_DT / 1.0, 0.0), 60.0)
                        v1_dist += v * GRID_DT
                    pct = lambda x: abs(x - true_dist) / true_dist * 100
                    results[b]["v1"].append(pct(v1_dist))
                    results[b]["cv"].append(pct(cv_dist))
                s += L

    summary = {}
    for b in BUCKETS_S:
        v1 = np.array(results[b]["v1"])
        cv = np.array(results[b]["cv"])
        if len(v1) == 0:
            continue
        summary[f"{b}s"] = dict(
            n=len(v1),
            v1_median=round(float(np.median(v1)), 2),
            v1_pass10=round(float(np.mean(v1 < 10) * 100), 1),
            cv_median=round(float(np.median(cv)), 2),
            cv_pass10=round(float(np.mean(cv < 10) * 100), 1),
        )
    json.dump(summary, open(os.path.join(paths.RESULTS, "v1_delta_eval.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("V1 DELTA EVAL DONE")


if __name__ == "__main__":
    main()
