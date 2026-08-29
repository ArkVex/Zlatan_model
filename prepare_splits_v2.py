"""Build the Round-2 (v2) split using dataset_v2 features -> splits_v2/."""
import glob
import json
import os
import numpy as np

from dataset_v2 import load_file, FEATURES, WIN

OUT = os.path.join(os.path.dirname(__file__), "splits_v2")
SEED = 42


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data", "S-*.csv")))
    rng = np.random.default_rng(SEED)
    files = [files[i] for i in rng.permutation(len(files))]
    n = len(files)
    groups = {
        "train": files[:int(0.70 * n)],
        "val":   files[int(0.70 * n):int(0.85 * n)],
        "test":  files[int(0.85 * n):],
    }
    data = {}
    for split, flist in groups.items():
        Xs, ys = [], []
        for f in flist:
            X, y = load_file(f)
            if X is None:
                continue
            Xs.append(X); ys.append(y)
        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.float32)
        data[split] = (X, y)
        print(f"{split:5s}: {len(flist)} files -> X{X.shape} y{y.shape}")

    Xtr = data["train"][0]
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    for split, (X, y) in data.items():
        np.save(os.path.join(OUT, f"X_{split}.npy"), ((X - mean) / std).astype(np.float32))
        np.save(os.path.join(OUT, f"y_{split}.npy"), y)
    json.dump(dict(seed=SEED, win=WIN, features=FEATURES, n_features=len(FEATURES),
                   mean=mean.tolist(), std=std.tolist(),
                   files={k: [os.path.basename(f) for f in v] for k, v in groups.items()}),
              open(os.path.join(OUT, "config.json"), "w"), indent=1)
    print("Saved v2 splits to", OUT)


if __name__ == "__main__":
    main()
