"""Round 4: a Temporal Convolutional Network (TCN) speed model in TensorFlow.

Bigger capacity than the CNN+GRU baseline, built from dilated CAUSAL 1-D convolutions
with residual connections. Uses only TFLite builtin ops (Conv1D, ReLU, Add, Dense,
GlobalAveragePooling) -> no SELECT_TF_OPS, settling the A8 decision. Stateless.

Trains on the same cached splits (splits_v2 = all 72 drives). Writes results_tcn.json
and model_tcn_tf.keras.
"""
import json
import os
import time
import numpy as np
import tensorflow as tf

from common_eval import metrics

from paths import ROOT as HERE, RESULTS
SPLITS = os.path.join(HERE, "splits_v2")
SEED = 42
EPOCHS = 20
BATCH = 256
LR = 1e-3
CH = 64
BLOCKS = [1, 2, 4, 8, 16]   # dilation rates


def load(split):
    return (np.load(os.path.join(SPLITS, f"X_{split}.npy")),
            np.load(os.path.join(SPLITS, f"y_{split}.npy")))


def residual_block(x, dilation):
    """Causal dilated conv residual block."""
    prev = x
    x = tf.keras.layers.Conv1D(CH, 3, padding="causal", dilation_rate=dilation,
                               activation="relu")(x)
    x = tf.keras.layers.Conv1D(CH, 3, padding="causal", dilation_rate=dilation,
                               activation="relu")(x)
    if prev.shape[-1] != CH:
        prev = tf.keras.layers.Conv1D(CH, 1, padding="same")(prev)
    return tf.keras.layers.Add()([prev, x])


def build(win, c_in):
    inp = tf.keras.Input(shape=(win, c_in), name="imu_window")
    x = inp
    for d in BLOCKS:
        x = residual_block(x, d)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, name="speed")(x)
    return tf.keras.Model(inp, out)


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    win, c_in = Xtr.shape[1], Xtr.shape[2]

    model = build(win, c_in)
    n_params = model.count_params()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")

    class Log(tf.keras.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            vm = metrics(yva, self.model.predict(Xva, verbose=0))
            print(f"[tcn] epoch {ep+1:2d}/{EPOCHS}  loss {logs['loss']:.3f}  "
                  f"val_MAE {vm['mae_ms']:.3f}")

    t0 = time.time()
    model.fit(Xtr, ytr, batch_size=BATCH, epochs=EPOCHS, shuffle=True,
              verbose=0, callbacks=[Log()])
    train_time = time.time() - t0

    m = metrics(yte, model.predict(Xte, verbose=0))
    model.save(os.path.join(HERE, "model_tcn_tf.keras"))
    size_kb = os.path.getsize(os.path.join(HERE, "model_tcn_tf.keras")) / 1024
    result = dict(framework="tensorflow-tcn", test=m, params=int(n_params),
                  train_time_s=round(train_time, 1), model_size_kb=round(size_kb, 1))
    json.dump(result, open(os.path.join(RESULTS, "results_tcn.json"), "w"), indent=1)
    print("TCN TEST:", m, "| params", n_params, "| time", result["train_time_s"], "s")


if __name__ == "__main__":
    main()
