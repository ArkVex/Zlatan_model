"""Blend evaluation: complementary filter over the three speed estimators.

    v(t) = (1 - lam) * [ v(t-1) + dv_model(t) ]  +  lam * v_abs(t)

  - v(t-1) + dv: the delta model propagates speed (tracks braking/accel, but random-walks)
  - v_abs: the absolute model pulls the estimate back (noisy but duration-stable)
  - lam: blend gain — 0 = pure delta integration, 1 = pure absolute model
  - anchor: v(0) = GNSS speed at outage entry (the CV baseline's strength)

This is a fixed-gain 1-D Kalman filter on speed — the miniature of the doc's EKF design.
lam is swept on the VALIDATION drives; the chosen value is then scored once on TEST.
Writes results/blend_eval.json.
"""
import json
import os
import numpy as np
import tensorflow as tf

import paths
from dataset_v2 import load_file, WIN

GRID_DT = 0.5
BUCKETS_S = [10, 30, 60, 180]
LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]


def predictions_for(files, mean, std, abs_model, delta_model, dv_std):
    """Per drive: (speeds, v_abs, dv_pred) on the 0.5 s window grid."""
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


def eval_blend(drives, lam):
    """Bucketed drift for one lambda. Returns {bucket: [drift%...]}"""
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
                    v = float(spd[s])                      # GNSS anchor at entry
                    dist = 0.0
                    for j in range(s, s + L):
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
    dv_std = 1.144   # printed by train_v1_delta.py (target standardisation constant)

    print("predicting on validation drives...")
    val = predictions_for(cfg["files"]["val"], mean, std, abs_model, delta_model, dv_std)
    print("sweeping lambda on validation...")
    best_lam, best_score = None, 1e9
    for lam in LAMBDAS:
        r = eval_blend(val, lam)
        # score: mean of bucket medians (equal weight per duration)
        meds = [np.median(v) for v in r.values() if v]
        score = float(np.mean(meds))
        print(f"  lam={lam:<5} mean-of-medians {score:.2f}")
        if score < best_score:
            best_lam, best_score = lam, score

    print(f"chosen lambda = {best_lam}")
    print("scoring on test drives...")
    test = predictions_for(cfg["files"]["test"], mean, std, abs_model, delta_model, dv_std)
    blend = summarize(eval_blend(test, best_lam))
    cv = summarize(eval_blend(test, 0.0))      # lam=0 with dv forced 0 would be CV; compute real CV:
    # real CV baseline: v stays at anchor
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
    cv = summarize(cv_res)

    out = dict(lambda_=best_lam, blend=blend, cv_baseline=cv)
    json.dump(out, open(os.path.join(paths.RESULTS, "blend_eval.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("BLEND EVAL DONE")


if __name__ == "__main__":
    main()
