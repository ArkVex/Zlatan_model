"""Train the speed-estimation model in PyTorch. Writes results_pytorch.json."""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from common_eval import HP, metrics

HERE = os.path.dirname(__file__)
SPLITS = os.path.join(HERE, os.environ.get("SPLIT_DIR", "splits"))
TAG = os.environ.get("TAG", "")   # e.g. "_v2" for round-2 outputs


def load(split):
    X = np.load(os.path.join(SPLITS, f"X_{split}.npy"))
    y = np.load(os.path.join(SPLITS, f"y_{split}.npy"))
    return X, y


class SpeedNet(nn.Module):
    """Conv1D -> Conv1D -> GRU -> Dense(1). Input (B, WIN, C)."""
    def __init__(self, c_in, k=5, c1=32, c2=64, gru=64):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(c_in, c1, k, padding=pad)
        self.conv2 = nn.Conv1d(c1, c2, k, padding=pad)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(c2, gru, batch_first=True)
        self.fc = nn.Linear(gru, 1)

    def forward(self, x):
        x = x.transpose(1, 2)              # (B, C, WIN)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.transpose(1, 2)              # (B, WIN, C2)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def main():
    torch.manual_seed(HP["seed"])
    np.random.seed(HP["seed"])
    device = "cpu"

    Xtr, ytr = load("train")
    Xva, yva = load("val")
    Xte, yte = load("test")
    c_in = Xtr.shape[-1]

    tr = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=HP["batch"], shuffle=True,
        generator=torch.Generator().manual_seed(HP["seed"]),
    )

    model = SpeedNet(c_in, HP["kernel"], HP["conv1"], HP["conv2"], HP["gru"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"])
    lossf = nn.MSELoss()

    t0 = time.time()
    for ep in range(HP["epochs"]):
        model.train()
        tot = 0.0
        for xb, yb in tr:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        # quick val
        model.eval()
        with torch.no_grad():
            vp = model(torch.tensor(Xva)).numpy()
        vm = metrics(yva, vp)
        print(f"[torch] epoch {ep+1:2d}/{HP['epochs']}  "
              f"train_mse {tot/len(Xtr):.4f}  val_MAE {vm['mae_ms']:.3f} m/s")
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        yp = model(torch.tensor(Xte)).numpy()
    m = metrics(yte, yp)

    torch.save(model.state_dict(), os.path.join(HERE, f"model_pytorch{TAG}.pt"))
    size_kb = os.path.getsize(os.path.join(HERE, f"model_pytorch{TAG}.pt")) / 1024

    result = dict(
        framework="pytorch", test=m, params=n_params,
        train_time_s=round(train_time, 1), model_size_kb=round(size_kb, 1),
    )
    json.dump(result, open(os.path.join(HERE, f"results_pytorch{TAG}.json"), "w"), indent=1)
    print("PYTORCH TEST:", m, "| params", n_params,
          "| time", result["train_time_s"], "s")


if __name__ == "__main__":
    main()
