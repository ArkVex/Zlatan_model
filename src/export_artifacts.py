"""Produce all integration deliverables for the app team, in one pass:

  engine_<ver>_<date>_<hash>.tflite   real TCN model, builtins-only
  norm.json                           per-channel standardisation constants (real values)
  manifest.json                       asset id, sha256, tensor spec, self-test vector
  testset.npz                         ~100 parity windows incl. hard cases (raw + norm + outputs)
  verify_export.py output             G1 Python export parity (keras vs tflite)

Run:  DATE=2026-08-29 HASH=419ee90 python export_artifacts.py
"""
import glob
import hashlib
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset_v2 import WIN, _resolve, RAW_KEYS, FEATURES

from paths import ROOT as HERE, EXPORTS
SPLITS = os.path.join(HERE, "splits_v2")
DATE = os.environ.get("DATE", "2026-08-29")
HASH = os.environ.get("HASH", "dev")
VERSION = f"tcn_v1_{DATE}_{HASH}"
TFLITE = f"engine_{VERSION}.tflite"


def raw_and_features(path):
    """Return raw [T,9] (accel,gravity,gyro), features [T,7], speed [T] for one file, 10 Hz."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    c = _resolve(df.columns)
    need = ["speed", "ax", "ay", "az", "grx", "gry", "grz", "wy", "wp", "wr"]
    if any(c[k] is None for k in need):
        return None

    def num(k):
        col = df[c[k]]
        if getattr(col, "ndim", 1) > 1:      # duplicate header -> DataFrame; take first
            col = col.iloc[:, 0]
        return pd.to_numeric(col, errors="coerce").to_numpy(np.float64).ravel()

    cols = {k: num(k) for k in ["ax", "ay", "az", "grx", "gry", "grz", "wy", "wp", "wr", "speed"]}
    m = min(len(v) for v in cols.values())     # some files have ragged trailing columns
    cols = {k: v[:m] for k, v in cols.items()}
    acc = np.stack([cols["ax"], cols["ay"], cols["az"]], 1)
    grav = np.stack([cols["grx"], cols["gry"], cols["grz"]], 1)
    gyro = np.stack([cols["wy"], cols["wp"], cols["wr"]], 1)
    speed = cols["speed"]
    a_lin = acc - grav
    g_hat = grav / (np.linalg.norm(grav, axis=1, keepdims=True) + 1e-6)
    a_vert = np.sum(a_lin * g_hat, axis=1)
    a_horiz = np.linalg.norm(a_lin - a_vert[:, None] * g_hat, axis=1)
    a_lin_mag = np.linalg.norm(a_lin, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    feat = np.stack([a_horiz, a_vert, a_lin_mag, gyro[:, 0], gyro[:, 1], gyro[:, 2], gyro_mag], 1)
    raw9 = np.concatenate([acc, grav, gyro], 1)
    good = np.isfinite(feat).all(1) & np.isfinite(speed) & np.isfinite(raw9).all(1)
    return raw9[good], feat[good].astype(np.float32), speed[good].astype(np.float32)


def collect_hardcase_windows(cfg):
    """Windows from held-out test drives, biased to hard cases (turns, braking, stop, fast)."""
    mean = np.array(cfg["mean"]); std = np.array(cfg["std"])
    test = set(cfg["files"]["test"])
    raws, feats, sps = [], [], []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "S-*.csv"))):
        if os.path.basename(f) not in test:
            continue
        r = raw_and_features(f)
        if r is None:
            continue
        raw9, feat, speed = r
        for i in range(0, len(feat) - WIN, 7):
            raws.append(raw9[i:i + WIN]); feats.append(feat[i:i + WIN]); sps.append(speed[i + WIN - 1])
    raws = np.array(raws, np.float32); feats = np.array(feats, np.float32); sps = np.array(sps, np.float32)

    # scores for hard-case picking
    gyro_energy = np.abs(feats[:, :, 6]).mean(1)      # turning
    acc_energy = feats[:, :, 2].mean(1)               # linear accel (braking/accel)
    spd = sps
    def top(idx, n): return list(idx[:n])
    pick = set()
    pick |= set(top(np.argsort(-gyro_energy), 22))    # sharpest turns
    pick |= set(top(np.argsort(-acc_energy), 20))     # hard braking / acceleration
    pick |= set(top(np.argsort(spd), 16))             # stationary / stop-and-go (slowest)
    pick |= set(top(np.argsort(-spd), 16))            # highest speed
    rng = np.random.default_rng(42)
    pick |= set(rng.choice(len(feats), 30, replace=False).tolist())
    pick = sorted(pick)[:110]
    return raws[pick], feats[pick], mean, std


def main():
    cfg = json.load(open(os.path.join(SPLITS, "config.json")))
    model = tf.keras.models.load_model(os.path.join(HERE, "model_tcn_tf.keras"))

    # ---- convert to TFLite (builtins only) ----
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = conv.convert()
    open(os.path.join(EXPORTS, TFLITE), "wb").write(tfl)
    sha = hashlib.sha256(tfl).hexdigest()

    it = tf.lite.Interpreter(model_content=tfl); it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]

    # ---- norm.json ----
    norm = {
        "schema_version": 1, "model_version": VERSION,
        "channel_order": FEATURES,
        "input_norm": {"method": "standardize",
                       "mean": [round(x, 6) for x in cfg["mean"]],
                       "std": [round(x, 6) for x in cfg["std"]]},
        "output_norm": {"method": "none", "target": "forward_speed_mps"},
        "input_units": {"accel": "m/s^2", "gravity": "m/s^2", "gyro": "rad/s"},
        "computed_over": "train_split_only",
        "n_windows": int(np.load(os.path.join(SPLITS, "X_train.npy"), mmap_mode="r").shape[0]),
    }
    json.dump(norm, open(os.path.join(EXPORTS, "norm.json"), "w"), indent=1)

    # ---- testset.npz (hard cases) ----
    raws, feats, mean, std = collect_hardcase_windows(cfg)
    norm_in = ((feats - mean) / std).astype(np.float32)
    keras_out = model.predict(norm_in, verbose=0, batch_size=256).astype(np.float32)
    tfl_out = np.zeros_like(keras_out)
    for i in range(len(norm_in)):
        it.set_tensor(di["index"], norm_in[i:i + 1]); it.invoke()
        tfl_out[i] = it.get_tensor(do["index"])[0]
    np.savez(os.path.join(EXPORTS, "testset.npz"),
             raw_inputs=raws.astype(np.float32),      # [K,50,9] accel+gravity+gyro @10Hz
             raw_high_rate=raws.astype(np.float32),   # IO-VNBD is natively 10 Hz -> identical
             model_inputs=feats.astype(np.float32),   # [K,50,7] after compute_features
             model_inputs_norm=norm_in,               # [K,50,7] after normalize
             keras_outputs=keras_out, tflite_outputs=tfl_out,
             meta=json.dumps({"model_version": VERSION, "tf": tf.__version__, "date": DATE}))

    # ---- G1 parity ----
    max_abs = float(np.max(np.abs(keras_out - tfl_out)))
    g1 = "PASS" if max_abs <= 1e-4 else "CHECK"

    # ---- manifest.json (with self-test vector) ----
    sv_in = norm_in[0]
    it.set_tensor(di["index"], sv_in[None]); it.invoke()
    sv_out = float(it.get_tensor(do["index"])[0][0])
    manifest = {
        "asset_id": "engine", "version": VERSION, "sha256": sha,
        "tflite_file": TFLITE,
        "input": {"name": di["name"], "shape": [int(x) for x in di["shape"]], "dtype": "float32"},
        "output": {"name": do["name"], "shape": [int(x) for x in do["shape"]], "dtype": "float32"},
        "channel_order": FEATURES,
        "units": {"accel": "m/s^2", "gravity": "m/s^2", "gyro": "rad/s"},
        "gravity_handling": "accel - TYPE_GRAVITY",
        "window_samples": WIN, "stride_samples": 1, "expected_input_rate_hz": 10,
        "input_norm": {"mean": [round(x, 6) for x in cfg["mean"]],
                       "std": [round(x, 6) for x in cfg["std"]]},
        "output_norm": "none",
        "self_test_vector": {"input": sv_in.round(5).tolist(),
                             "expected_output_mps": round(sv_out, 5), "atol": 1e-3},
        "min_engine_version": "0.1.0",
    }
    json.dump(manifest, open(os.path.join(EXPORTS, "manifest.json"), "w"), indent=1)

    print(f"TFLITE  {TFLITE}  ({len(tfl)//1024} KB)  sha256 {sha[:12]}...")
    print(f"norm.json, manifest.json written")
    print(f"testset.npz: {len(feats)} windows (hard cases)")
    print(f"G1 export parity: max_abs_diff {max_abs:.2e} -> {g1}")
    print(f"self_test: input[0] -> {sv_out:.4f} m/s")


if __name__ == "__main__":
    main()
