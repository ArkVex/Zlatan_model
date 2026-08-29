"""Drift KPI with a realistic GPS-reset anchor.

At the moment a blackout starts, GPS was still valid an instant earlier, so we know the
true speed at t=start. We use that to remove the model's systematic bias at this operating
point: offset = true_speed[start] - pred_speed[start], then integrate (pred + offset).

This is exactly the "integrate with GPS reset" idea and is what a real fused system does.
Compares raw-model drift vs anchored drift so we can see how much the reset helps.

Usage: MODEL_FILE=model_tcn_tf.keras python evaluate_drift_anchored.py
"""
import glob
import json
import os
import numpy as np
import tensorflow as tf

from dataset_v2 import WIN, _resolve
from evaluate_drift import build_features_and_truth, latlon_to_m, TARGET_DIST, DT

from paths import ROOT as HERE, RESULTS
SPLITS = os.path.join(HERE, "splits_v2")
MODEL = os.path.join(HERE, os.environ.get("MODEL_FILE", "model_tcn_tf.keras"))


def eval_drive(path, model, mean, std):
    d = build_features_and_truth(path)
    if d is None or len(d["feat"]) < WIN + 300:
        return [], []
    feat, speed = d["feat"], d["speed"]
    lat, lon = d["lat"], d["lon"]
    n = len(feat)
    idx = np.arange(WIN - 1, n)
    X = np.stack([feat[i - WIN + 1:i + 1] for i in idx])
    Xn = ((X - mean) / std).astype(np.float32)
    pred = model.predict(Xn, verbose=0, batch_size=1024).ravel()
    pred_speed = np.zeros(n)
    pred_speed[idx] = pred
    pred_speed[:WIN - 1] = pred[0]

    lat0 = np.nanmean(lat)
    gx, gy = latlon_to_m(lat, lon, lat0)
    step = np.sqrt(np.diff(gx) ** 2 + np.diff(gy) ** 2)
    cum = np.concatenate([[0], np.cumsum(step)])

    raw, anch = [], []
    start = WIN
    while start < n - 10:
        end = np.searchsorted(cum, cum[start] + TARGET_DIST)
        if end >= n:
            break
        true_dist = float(cum[end] - cum[start])
        if true_dist > 0:
            v = np.clip(pred_speed[start:end], 0.0, None)
            raw.append(abs(float(np.sum(v) * DT) - true_dist) / true_dist * 100)
            # anchored: remove bias using the (known) true speed at blackout start
            offset = speed[start] - pred_speed[start]
            v2 = np.clip(pred_speed[start:end] + offset, 0.0, None)
            anch.append(abs(float(np.sum(v2) * DT) - true_dist) / true_dist * 100)
        start = end
    return raw, anch


def summarize(a):
    a = np.array(a)
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

    raw_all, anch_all = [], []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "S-*.csv"))):
        if os.path.basename(f) not in test:
            continue
        r, a = eval_drive(f, model, mean, std)
        raw_all += r
        anch_all += a

    out = dict(model=os.path.basename(MODEL), target_dist_m=TARGET_DIST,
               raw=summarize(raw_all), anchored=summarize(anch_all))
    json.dump(out, open(os.path.join(RESULTS, "drift_anchored.json"), "w"), indent=1)
    print("DRIFT (raw model):     ", out["raw"])
    print("DRIFT (GPS-anchored):  ", out["anchored"])


if __name__ == "__main__":
    main()
