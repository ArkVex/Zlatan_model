"""Train the SAME model in TensorFlow/Keras. Writes results_tf.json.

Architecture, hyperparameters, seed, optimizer, and data are identical to the
PyTorch version so the comparison is fair.
"""
import json
import os
import time
import numpy as np
import tensorflow as tf

from common_eval import HP, metrics

from paths import ROOT as HERE, RESULTS
SPLITS = os.path.join(HERE, os.environ.get("SPLIT_DIR", "splits"))
TAG = os.environ.get("TAG", "")   # e.g. "_v2" for round-2 outputs


def load(split):
    X = np.load(os.path.join(SPLITS, f"X_{split}.npy"))
    y = np.load(os.path.join(SPLITS, f"y_{split}.npy"))
    return X, y


def build(c_in, win):
    k = HP["kernel"]
    m = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(win, c_in)),
        tf.keras.layers.Conv1D(HP["conv1"], k, padding="same", activation="relu"),
        tf.keras.layers.Conv1D(HP["conv2"], k, padding="same", activation="relu"),
        tf.keras.layers.GRU(HP["gru"]),
        tf.keras.layers.Dense(1),
    ])
    return m


def main():
    tf.random.set_seed(HP["seed"])
    np.random.seed(HP["seed"])

    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    win, c_in = Xtr.shape[1], Xtr.shape[2]

    model = build(c_in, win)
    n_params = model.count_params()
    model.compile(optimizer=tf.keras.optimizers.Adam(HP["lr"]), loss="mse")

    class Log(tf.keras.callbacks.Callback):
        def on_epoch_end(self, ep, logs=None):
            vp = self.model.predict(Xva, verbose=0)
            vm = metrics(yva, vp)
            print(f"[tf]    epoch {ep+1:2d}/{HP['epochs']}  "
                  f"train_mse {logs['loss']:.4f}  val_MAE {vm['mae_ms']:.3f} m/s")

    t0 = time.time()
    model.fit(Xtr, ytr, batch_size=HP["batch"], epochs=HP["epochs"],
              shuffle=True, verbose=0, callbacks=[Log()])
    train_time = time.time() - t0

    yp = model.predict(Xte, verbose=0)
    m = metrics(yte, yp)

    model.save(os.path.join(HERE, f"model_tf{TAG}.keras"))
    size_kb = os.path.getsize(os.path.join(HERE, f"model_tf{TAG}.keras")) / 1024

    result = dict(
        framework="tensorflow", test=m, params=int(n_params),
        train_time_s=round(train_time, 1), model_size_kb=round(size_kb, 1),
    )
    json.dump(result, open(os.path.join(RESULTS, f"results_tf{TAG}.json"), "w"), indent=1)
    print("TF TEST:", m, "| params", n_params,
          "| time", result["train_time_s"], "s")


if __name__ == "__main__":
    main()
