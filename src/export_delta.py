"""1) Re-check the TV blend using the v2 absolute model (what the phone actually ships).
2) Export the delta model to TFLite with the output de-standardisation (x dv_std) baked
   into the graph, so the app receives delta-speed directly in m/s per second.
Writes exports/engine_delta_v1_<date>.tflite + exports/delta.manifest.json.
"""
import hashlib
import json
import os
import numpy as np
import tensorflow as tf

import paths
from dataset_v2 import load_file, FEATURES, WIN
from blend_tv_eval import predictions_for, eval_tv, summarize

DV_STD = 1.144
TAU = 240.0
VERSION = "delta_v1_2026-08-30"


def main():
    cfg = json.load(open(os.path.join(paths.SPLITS_V2, "config.json")))
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)
    delta_model = tf.keras.models.load_model(os.path.join(paths.MODELS, "model_tcn_v1delta.keras"))

    # ---- 1. blend check with the v2 absolute model (negatives-trained, its own norm) ----
    v2 = tf.keras.models.load_model(os.path.join(paths.MODELS, "model_tcn_v2.keras"))
    v2norm = json.load(open(os.path.join(paths.EXPORTS, "norm.json")))["input_norm"]
    v2m, v2s = np.array(v2norm["mean"], np.float32), np.array(v2norm["std"], np.float32)

    test = []
    for f in cfg["files"]["test"]:
        r = load_file(os.path.join(paths.DATA, f))
        if r[0] is None:
            continue
        X, spd = r
        v_abs = np.clip(v2.predict(((X - v2m) / v2s).astype(np.float32),
                                   verbose=0, batch_size=1024).ravel(), 0, 60)
        dv = delta_model.predict(((X - mean) / std).astype(np.float32),
                                 verbose=0, batch_size=1024).ravel() * DV_STD
        test.append((f, spd, v_abs, dv))
    print("TV blend with V2-ABS on test:")
    print(json.dumps(summarize(eval_tv(test, TAU)), indent=1))

    # ---- 2. export delta model with baked output scale ----
    inp = tf.keras.Input((WIN, 7), name="imu_window")
    scaled = delta_model(inp) * DV_STD
    wrapped = tf.keras.Model(inp, tf.keras.layers.Identity(name="dv_mps")(scaled))
    conv = tf.lite.TFLiteConverter.from_keras_model(wrapped)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = conv.convert()
    tfl_path = os.path.join(paths.EXPORTS, f"engine_{VERSION}.tflite")
    open(tfl_path, "wb").write(tfl)
    sha = hashlib.sha256(tfl).hexdigest()

    # parity + self-test vector
    it = tf.lite.Interpreter(model_content=tfl)
    it.allocate_tensors()
    di, do = it.get_input_details()[0], it.get_output_details()[0]
    # pick the longest test drive for a stable self-test vector
    best = None
    for f in cfg["files"]["test"]:
        r = load_file(os.path.join(paths.DATA, f))
        if r[0] is not None and (best is None or len(r[0]) > len(best)):
            best = r[0]
    sv_in = ((best[min(100, len(best) - 1)] - mean) / std).astype(np.float32)
    it.set_tensor(di["index"], sv_in[None])
    it.invoke()
    sv_out = float(it.get_tensor(do["index"])[0][0])
    keras_out = float(delta_model.predict(sv_in[None], verbose=0)[0][0]) * DV_STD
    print(f"parity: keras {keras_out:.5f} vs tflite {sv_out:.5f} diff {abs(keras_out-sv_out):.2e}")

    manifest = {
        "asset_id": "speed_delta_model", "version": VERSION, "sha256": sha,
        "asset_type": "SPEED_MODEL",
        "tensor_shapes": {"imu_window": [1, WIN, 7], "dv_mps": [1, 1]},
        "tensor_dtypes": {"imu_window": "float32", "dv_mps": "float32"},
        "channel_order": FEATURES, "gravity_handling": "accel_minus_gravity",
        "window_samples": WIN, "stride_samples": 1, "expected_rate_hz": 10.0,
        "input_norm": {"mean": [round(float(x), 6) for x in mean],
                       "std": [round(float(x), 6) for x in std]},
        "output_norm": {"method": "none",
                        "_comment": "dv_std baked into the graph; output is m/s per second"},
        "self_test": {"input": sv_in.round(5).tolist(),
                      "expected_output": round(sv_out, 5), "tolerance_abs": 1e-3},
        "min_engine_version": "0.1.0-phase0",
        "blend": {"tau_seconds": TAU,
                  "formula": "lam=t/(t+tau); v=(1-lam)*(v+dv*dt)+lam*v_abs; anchor v at last trusted GNSS speed"},
    }
    json.dump(manifest, open(os.path.join(paths.EXPORTS, "delta.manifest.json"), "w"), indent=1)
    print(f"exported {os.path.basename(tfl_path)} ({len(tfl)//1024} KB) sha {sha[:12]}")
    print("DELTA EXPORT DONE")


if __name__ == "__main__":
    main()
