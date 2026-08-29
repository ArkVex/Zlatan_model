"""Time-varying gain blend: trust the anchor early, the absolute model late.

    lam(t) = t / (t + tau)          t = seconds since blackout entry
    v(t) = (1 - lam(t)) * [ v(t-1) + dv_model(t) ]  +  lam(t) * v_abs(t)

Early in the outage lam ~ 0: the estimate rides the GNSS anchor + deltas (the CV
baseline's strong zone). As the outage ages, lam grows and the duration-stable absolute
model takes over before delta noise random-walks away. tau is swept on VALIDATION,
scored once on TEST. Writes results/blend_tv_eval.json.
"""
import json
import os
import numpy as np
import tensorflow as tf

import paths
from dataset_v2 import load_file

GRID_DT = 0.5
BUCKETS_S = [10, 30, 60, 180]
TAUS = [15.0, 30.0, 60.0, 120.0, 240.0]


def predictions_for(files, mean, std, abs_model, delta_model, dv_std):
    out = []
    for f in files:
        r = load_file(os.path.join(paths.DATA, f))
        if r[0] is None:
            continue
        X, spd = r
        Xn = ((X - mean) / std).astype(np.float32)
        v_abs = np.clip(abs_model.predict(Xn, verbose=0, batch_size=1024).ravel(), 0, 60)
        dv = delta_model.predict(Xn, verbose=0, batch_size=1024).ravel() * dv_std
        out.append((f, spd, v_abs, dv))
    return out


def eval_tv(drives, tau):
    res = {b: [] for b in BUCKETS_S}
    for _, spd, v_abs, dv in drives:
        n = len(spd)
        for b in BUCKETS_S:
            L = int(b / GRID_DT)
            s = 4
            while s + L < n:
                seg = spd[s:s + L]
                if seg.mean() > 3.0:
                    true_dist = float(np.sum(seg) * GRID_DT)
                    v = float(spd[s])
                    dist = 0.0
                    for j in range(s, s + L):
                        t = (j - s) * GRID_DT
                        lam = t / (t + tau)
                        v = (1 - lam) * (v + dv[j] * GRID_DT) + lam * v_abs[j]
                        v = min(max(v, 0.0), 60.0)
                        dist += v * GRID_DT
                    res[b].append(abs(dist - true_dist) / true_dist * 100)
                s += L
    return res


def summarize(res):
    return {f"{b}s": dict(n=len(v),
                          median=round(float(np.median(v)), 2),
                          pass10=round(float(np.mean(np.array(v) < 10) * 100), 1))
            for b, v in res.items() if v}


def main():
    cfg = json.load(open(os.path.join(paths.SPLITS_V2, "config.json")))
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)
    abs_model = tf.keras.models.load_model(os.path.join(paths.MODELS, "model_tcn_tf.keras"))
    delta_model = tf.keras.models.load_model(os.path.join(paths.MODELS, "model_tcn_v1delta.keras"))
    dv_std = 1.144

    print("validation predictions...")
    val = predictions_for(cfg["files"]["val"], mean, std, abs_model, delta_model, dv_std)
    best_tau, best = None, 1e9
    for tau in TAUS:
        r = eval_tv(val, tau)
        meds = [np.median(v) for v in r.values() if v]
        score = float(np.mean(meds))
        print(f"  tau={tau:<6} mean-of-medians {score:.2f}")
        if score < best:
            best_tau, best = tau, score
    print(f"chosen tau = {best_tau}")

    print("test predictions...")
    test = predictions_for(cfg["files"]["test"], mean, std, abs_model, delta_model, dv_std)
    tv = summarize(eval_tv(test, best_tau))

    # CV baseline on the same segments for the comparison table
    cv_res = {b: [] for b in BUCKETS_S}
    for _, spd, _, _ in test:
        n = len(spd)
        for b in BUCKETS_S:
            L = int(b / GRID_DT)
            s = 4
            while s + L < n:
                seg = spd[s:s + L]
                if seg.mean() > 3.0:
                    true_dist = float(np.sum(seg) * GRID_DT)
                    cv_res[b].append(abs(float(spd[s]) * b - true_dist) / true_dist * 100)
                s += L
    out = dict(tau=best_tau, time_varying_blend=tv, cv_baseline=summarize(cv_res))
    json.dump(out, open(os.path.join(paths.RESULTS, "blend_tv_eval.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("TV BLEND DONE")


if __name__ == "__main__":
    main()
