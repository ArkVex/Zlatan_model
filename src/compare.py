"""Print a side-by-side comparison of the PyTorch and TensorFlow results."""
import json
import os
import sys

from paths import RESULTS as HERE
TAG = sys.argv[1] if len(sys.argv) > 1 else ""   # e.g. "_v2"


def load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    pt = load(f"results_pytorch{TAG}.json")
    tf = load(f"results_tf{TAG}.json")
    if not pt or not tf:
        print(f"Missing results for TAG='{TAG}'. Run both trainers first.")
        return
    print(f"=== Comparison (TAG='{TAG or 'round1'}') ===")

    rows = [
        ("Test MAE (m/s)",   pt["test"]["mae_ms"],   tf["test"]["mae_ms"],   "lower"),
        ("Test RMSE (m/s)",  pt["test"]["rmse_ms"],  tf["test"]["rmse_ms"],  "lower"),
        ("Test R^2",         pt["test"]["r2"],       tf["test"]["r2"],       "higher"),
        ("Speed %error",     pt["test"]["pct_error"], tf["test"]["pct_error"], "lower"),
        ("Params",           pt["params"],           tf["params"],           "-"),
        ("Train time (s)",   pt["train_time_s"],     tf["train_time_s"],     "lower"),
        ("Model size (KB)",  pt["model_size_kb"],    tf["model_size_kb"],    "lower"),
    ]

    print(f"\n{'Metric':18s} {'PyTorch':>12s} {'TensorFlow':>12s}   Winner")
    print("-" * 58)
    for name, a, b, better in rows:
        win = ""
        if better == "lower":
            win = "PyTorch" if a < b else "TensorFlow" if b < a else "tie"
        elif better == "higher":
            win = "PyTorch" if a > b else "TensorFlow" if b > a else "tie"
        print(f"{name:18s} {a:>12} {b:>12}   {win}")
    print("-" * 58)
    print("Test set is byte-identical for both (cached .npy). Lower MAE/RMSE = better.")


if __name__ == "__main__":
    main()
