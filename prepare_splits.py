"""Build ONE fixed train/val/test split and cache it to .npy.

Both the PyTorch and TensorFlow trainers load these exact arrays, so any
difference in results comes from the framework/model, never the data.

Split is BY FILE (a whole drive goes to one split) to avoid leakage between
overlapping windows of the same drive. Normalization stats are computed on the
training split only and applied to all splits.
"""
import glob
import json
import os
import numpy as np

from dataset import load_file, FEATURES, WIN

OUT = os.path.join(os.path.dirname(__file__), "splits")
SEED = 42


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "S-*.csv")))

    # Deterministic file-level shuffle, then 70/15/15 split by file.
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(files))
    files = [files[i] for i in order]
    n = len(files)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    groups = {
        "train": files[:n_tr],
        "val":   files[n_tr:n_tr + n_va],
        "test":  files[n_tr + n_va:],
    }

    data = {}
    for split, flist in groups.items():
        Xs, ys = [], []
        for f in flist:
            X, y = load_file(f)
            if X is None:
                continue
            Xs.append(X)
            ys.append(y)
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.float32)
        data[split] = (X, y)
        print(f"{split:5s}: {len(flist)} files -> X{X.shape} y{y.shape}")

    # Per-channel standardization using TRAIN only.
    Xtr = data["train"][0]
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6

    for split, (X, y) in data.items():
        Xn = ((X - mean) / std).astype(np.float32)
        np.save(os.path.join(OUT, f"X_{split}.npy"), Xn)
        np.save(os.path.join(OUT, f"y_{split}.npy"), y)

    cfg = {
        "seed": SEED, "win": WIN, "features": FEATURES,
        "n_features": len(FEATURES),
        "mean": mean.tolist(), "std": std.tolist(),
        "files": {k: [os.path.basename(f) for f in v] for k, v in groups.items()},
    }
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    print("Saved splits + config to", OUT)


if __name__ == "__main__":
    main()
