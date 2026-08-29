"""Reference decimator: phone's high-rate IMU -> the model's 10 Hz grid.

IO-VNBD smartphone data is natively 10 Hz (verified: 100 ms steps). A phone samples at
~50-200 Hz, so the app must downsample to exactly 10 Hz BEFORE compute_features(). Doing it
naively (pick-nearest) aliases vibration energy into the band the speed model relies on, so
the app must anti-alias first. This is the twin of the Kotlin `Decimator` (gate G2a).

Method: causal low-pass (moving average over one 10 Hz bin ~= a 5 Hz-ish anti-alias) then
sample onto the 10 Hz grid. Causal only — no `filtfilt` (the phone can't run it live).
"""
import numpy as np

TARGET_HZ = 10.0
DT = 1.0 / TARGET_HZ


def decimate(raw, timestamps_s, target_hz=TARGET_HZ):
    """raw: [T, C] high-rate samples. timestamps_s: [T] seconds (monotonic).
    Returns [M, C] on a uniform target_hz grid, anti-aliased by per-bin averaging.

    Each output sample = mean of the input samples whose timestamp falls in that bin. That
    box-average is a simple causal anti-alias filter matched to the output rate; it needs no
    filter state and is trivial to reproduce in Kotlin.
    """
    raw = np.asarray(raw, np.float64)
    t = np.asarray(timestamps_s, np.float64)
    t0, t1 = t[0], t[-1]
    grid = np.arange(t0, t1, 1.0 / target_hz)
    out = np.empty((len(grid), raw.shape[1]), np.float64)
    # bin edges centred on grid points
    edges = np.concatenate([[grid[0] - 0.5 / target_hz],
                            grid[:-1] + 0.5 / target_hz,
                            [grid[-1] + 0.5 / target_hz]])
    idx = np.searchsorted(t, edges)
    for m in range(len(grid)):
        a, b = idx[m], idx[m + 1]
        if b <= a:                       # no samples in the bin -> nearest sample
            j = min(np.searchsorted(t, grid[m]), len(raw) - 1)
            out[m] = raw[j]
        else:
            out[m] = raw[a:b].mean(0)
    return out.astype(np.float32)


if __name__ == "__main__":
    # 100 Hz input -> 10 Hz output smoke test
    t = np.arange(0, 5, 0.01)
    raw = np.stack([np.sin(2 * np.pi * 1.0 * t), np.cos(2 * np.pi * 0.5 * t)], 1)
    out = decimate(raw, t)
    print("in", raw.shape, "-> out", out.shape, "(expect ~50 rows @10Hz)")
