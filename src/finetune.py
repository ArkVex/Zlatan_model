"""Retrain the TCN so NON-VEHICLE motion (still / hand-held / walking) -> ~0 speed.

The base model was trained only on driving (IO-VNBD), so hand-held motion is out of
distribution and it invents 10-40 km/h. Here we add "negative" windows labelled 0 m/s:
  - synthetic stationary (near-zero IMU + noise)
  - the app's recorded hand-held trips (negatives/*.csv), decimated 453Hz -> 10Hz
and retrain. Goal: keep driving accuracy, but output ~0 when not in a vehicle.

Writes exports/engine_tcn_v2_*.tflite + matching norm.json / manifest.json.
"""
import glob
import hashlib
import json
import os
import numpy as np
import tensorflow as tf

import paths
from dataset_v2 import load_file, FEATURES, WIN

DATE = os.environ.get("DATE", "2026-08-30")
HASH = os.environ.get("HASH", "ft")
VERSION = f"tcn_v2_{DATE}_{HASH}"
SEED = 42
rng = np.random.default_rng(SEED)


def driving_windows(files):
    Xs, ys = [], []
    for f in files:
        p = os.path.join(paths.DATA, f)
        X, y = load_file(p)
        if X is None:
            continue
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys)


def handheld_negatives(path, n_aug=8):
    """Decimate a 453 Hz trip to 10 Hz, build feature windows, label 0. Augment with noise."""
    rows = [l.strip().split(",") for l in open(path) if l.startswith("IMU")]
    if len(rows) < 500:
        return np.zeros((0, WIN, 7), np.float32)
    d = np.array([[float(x) for x in r[1:]] for r in rows])
    t = d[:, 0] / 1e9; t -= t[0]
    acc, grav, gyro = d[:, 1:4], d[:, 4:7], d[:, 7:10]
    grid = np.arange(t[0], t[-1], 0.1)
    idx = np.searchsorted(t, np.concatenate([[grid[0]], (grid[:-1] + grid[1:]) / 2, [grid[-1] + 1]]))
    def binavg(x):
        out = np.empty((len(grid), x.shape[1]))
        for i in range(len(grid)):
            a, b = idx[i], idx[i + 1]
            out[i] = x[a:b].mean(0) if b > a else x[min(a, len(x) - 1)]
        return out
    accD, gravD, gyroD = binavg(acc), binavg(grav), binavg(gyro)
    a_lin = accD - gravD
    g_hat = gravD / (np.linalg.norm(gravD, axis=1, keepdims=True) + 1e-6)
    a_vert = np.sum(a_lin * g_hat, axis=1)
    a_horiz = np.linalg.norm(a_lin - a_vert[:, None] * g_hat, axis=1)
    feat = np.stack([a_horiz, a_vert, np.linalg.norm(a_lin, axis=1),
                     gyroD[:, 0], gyroD[:, 1], gyroD[:, 2], np.linalg.norm(gyroD, axis=1)], 1).astype(np.float32)
    base = np.stack([feat[i:i + WIN] for i in range(0, len(feat) - WIN)])
    out = [base]
    for _ in range(n_aug):
        out.append(base + rng.normal(0, 0.05, base.shape).astype(np.float32))  # jitter augmentation
    return np.concatenate(out)


def synthetic_stationary(n):
    """Near-zero-motion windows (phone at rest): small positive magnitudes + noise."""
    a_horiz = np.abs(rng.normal(0, 0.12, (n, WIN)))
    a_vert = rng.normal(0, 0.12, (n, WIN))
    a_lin = np.abs(rng.normal(0, 0.15, (n, WIN)))
    gyr = rng.normal(0, 0.01, (n, WIN, 3))
    gyro_mag = np.abs(rng.normal(0, 0.02, (n, WIN)))
    return np.stack([a_horiz, a_vert, a_lin, gyr[:, :, 0], gyr[:, :, 1], gyr[:, :, 2], gyro_mag], -1).astype(np.float32)


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
    out = tf.keras.layers.Dense(1, name="speed")(x)
    return tf.keras.Model(inp, out)


def main():
    tf.random.set_seed(SEED)
    cfg = json.load(open(os.path.join(paths.SPLITS_V2, "config.json")))

    Xtr_d, ytr_d = driving_windows(cfg["files"]["train"])
    Xte_d, yte_d = driving_windows(cfg["files"]["test"])
    print(f"driving: train {Xtr_d.shape} test {Xte_d.shape}")

    # negatives (~15% of driving train), split so we can measure them
    neg_files = sorted(glob.glob(os.path.join(paths.ROOT, "negatives", "*.csv")))
    hh = np.concatenate([handheld_negatives(f) for f in neg_files]) if neg_files else np.zeros((0, WIN, 7), np.float32)
    n_syn = max(int(0.12 * len(Xtr_d)) - len(hh), 2000)
    syn = synthetic_stationary(n_syn)
    Xneg = np.concatenate([hh, syn]); yneg = np.zeros(len(Xneg), np.float32)
    print(f"negatives: handheld {len(hh)} + synthetic {len(syn)} = {len(Xneg)} (labelled 0)")
    # hold out 15% of negatives to evaluate
    perm = rng.permutation(len(Xneg)); cut = int(0.85 * len(Xneg))
    neg_tr, neg_te = perm[:cut], perm[cut:]

    Xtr = np.concatenate([Xtr_d, Xneg[neg_tr]]); ytr = np.concatenate([ytr_d, yneg[neg_tr]])
    p = rng.permutation(len(Xtr)); Xtr, ytr = Xtr[p], ytr[p]

    mean = Xtr.reshape(-1, 7).mean(0); std = Xtr.reshape(-1, 7).std(0) + 1e-6
    norm = lambda X: ((X - mean) / std).astype(np.float32)

    model = build_tcn(WIN, 7)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.fit(norm(Xtr), ytr, batch_size=256, epochs=18, shuffle=True, verbose=2)

    # eval
    pd = model.predict(norm(Xte_d), verbose=0).ravel()
    mae = float(np.mean(np.abs(pd - yte_d)))
    r2 = 1 - np.sum((pd - yte_d) ** 2) / np.sum((yte_d - yte_d.mean()) ** 2)
    pn = model.predict(norm(Xneg[neg_te]), verbose=0).ravel()
    print(f"\nDRIVING test: MAE {mae:.2f} m/s  R2 {r2:.3f}")
    print(f"NON-VEHICLE (should be ~0): mean predicted {pn.mean():.2f} m/s  ({pn.mean()*3.6:.1f} km/h)  max {pn.max():.2f}")

    # export
    model.save(os.path.join(paths.MODELS, "model_tcn_v2.keras"))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = conv.convert()
    tfl_path = os.path.join(paths.EXPORTS, f"engine_{VERSION}.tflite")
    open(tfl_path, "wb").write(tfl)
    sha = hashlib.sha256(tfl).hexdigest()
    json.dump({"schema_version": 1, "model_version": VERSION, "channel_order": FEATURES,
               "input_norm": {"method": "standardize", "mean": [round(float(x), 6) for x in mean], "std": [round(float(x), 6) for x in std]},
               "output_norm": {"method": "none", "target": "forward_speed_mps"},
               "input_units": {"accel": "m/s^2", "gravity": "m/s^2", "gyro": "rad/s"},
               "computed_over": "train_split_plus_negatives"},
              open(os.path.join(paths.EXPORTS, "norm.json"), "w"), indent=1)
    # self-test vector
    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    sv = norm(Xte_d[:1])[0]
    it.set_tensor(di["index"], sv[None]); it.invoke(); svo = float(it.get_tensor(do["index"])[0][0])
    json.dump({"asset_id": "speed_model", "version": VERSION, "sha256": sha, "asset_type": "SPEED_MODEL",
               "tensor_shapes": {"imu_window": [1, 50, 7], "speed": [1, 1]},
               "tensor_dtypes": {"imu_window": "float32", "speed": "float32"},
               "channel_order": FEATURES, "gravity_handling": "accel_minus_gravity",
               "window_samples": 50, "stride_samples": 1, "expected_rate_hz": 10.0,
               "input_norm": {"mean": [round(float(x), 6) for x in mean], "std": [round(float(x), 6) for x in std]},
               "output_norm": {"method": "none"},
               "self_test": {"input": sv.round(5).tolist(), "expected_output": round(svo, 5), "tolerance_abs": 1e-3},
               "min_engine_version": "0.1.0-phase0"},
              open(os.path.join(paths.EXPORTS, "manifest.json"), "w"), indent=1)
    print(f"\nexported {os.path.basename(tfl_path)} ({len(tfl)//1024} KB)  sha {sha[:12]}")


if __name__ == "__main__":
    main()
