"""Extract "stopped, but engine running" negatives from a recorded ride.

Why this exists
---------------
`finetune.py` built v2's negatives from whole hand-held trips labelled 0 m/s: a phone on a desk, a
phone in a hand. All of them QUIET. A powered two-wheeler idling at a traffic light is not quiet —
the engine shakes the frame, and the phone feels vibration that looks exactly like motion.

Measured on session tel_20260901 (106 min, 50,649 ticks), v2 reads a stationary vehicle at roughly
8 km/h, and the two classes overlap so badly that no threshold can separate them:

    v_model, GNSS says STOPPED:  p50 2.16   p90 6.47 m/s
    v_model, GNSS says MOVING :  p50 6.13   p90 11.58 m/s

Stopped p90 sits above moving p50. This is not fixable downstream — three filtering rules were
replayed against the real data and all three failed, one making things worse — so the model has to
learn the signature. "Stationary and quiet" and "stationary with the engine running" are two
different negative classes, and v2 only ever saw the first.

What this does
--------------
Unlike `handheld_negatives`, which labels an entire file 0, this labels only the intervals where
GNSS independently confirms the vehicle was stopped. It needs two files from the same session:

  * the raw trip trace (`trip_*.csv`, IMU rows at ~250 Hz) — the model's input
  * the telemetry CSV (`tel_*.csv`, schema >= 3)                — the GNSS ground truth

Both stamp `t_nanos` from `SystemClock.elapsedRealtimeNanos`, so they share a clock.

Usage:
    python src/engine_running_negatives.py trip_*.csv tel_*.csv -o negatives/engine_idle_<name>.csv
"""
import argparse
import csv
import sys

import numpy as np

# GNSS speed below this counts as stopped. Deliberately tight: a negative that is actually
# creeping teaches the model that slow motion is zero, which is a worse failure than the one
# being fixed.
STOPPED_MPS = 0.3

# Only trust intervals where GNSS was actually being received. During dead reckoning `v_gnss` is a
# stale value from the last fix and proves nothing about what the vehicle is doing now.
AIDED_MODES = {"GNSS", "NAVIC"}

# Drop intervals shorter than this. A single tick under the threshold is as likely to be GNSS noise
# as a real halt, and a window has to sit entirely inside one interval to be usable.
MIN_INTERVAL_S = 3.0

# Guard band trimmed from each end of an interval, seconds. GNSS speed lags the vehicle, so the
# moments either side of a stop are ambiguous — the vehicle may still be rolling while the fix
# already reads zero. Trimming keeps the label honest at the cost of a little data.
EDGE_TRIM_S = 1.0


def stopped_intervals(tel_path):
    """[(t0_nanos, t1_nanos)] where GNSS was live AND reported the vehicle stopped."""
    with open(tel_path) as f:
        first = f.readline()
        if not first.startswith("#schema"):
            f.seek(0)
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{tel_path}: no rows")
    for col in ("t_nanos", "v_gnss_mps", "mode"):
        if col not in rows[0]:
            raise SystemExit(f"{tel_path}: missing column '{col}' — telemetry schema too old")

    intervals, start, prev_t = [], None, None
    for r in rows:
        try:
            t = int(float(r["t_nanos"]))
            v = float(r["v_gnss_mps"])
        except ValueError:
            continue
        stopped = r["mode"] in AIDED_MODES and v < STOPPED_MPS
        if stopped and start is None:
            start = t
        elif not stopped and start is not None:
            intervals.append((start, prev_t))
            start = None
        prev_t = t
    if start is not None:
        intervals.append((start, prev_t))

    trim = int(EDGE_TRIM_S * 1e9)
    out = []
    for a, b in intervals:
        a, b = a + trim, b - trim
        if (b - a) / 1e9 >= MIN_INTERVAL_S:
            out.append((a, b))
    return out


def imu_rows(trip_path):
    """(t_nanos, ax,ay,az, gx,gy,gz, wx,wy,wz) from a raw trip trace."""
    keep = []
    with open(trip_path) as f:
        for line in f:
            if not line.startswith("IMU"):
                continue
            p = line.strip().split(",")
            if len(p) < 11:
                continue
            try:
                keep.append([float(x) for x in p[1:11]])
            except ValueError:
                continue
    if not keep:
        raise SystemExit(f"{trip_path}: no IMU rows — is this a trip trace?")
    return np.asarray(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trip", help="raw trip_*.csv (IMU rows)")
    ap.add_argument("telemetry", help="tel_*.csv from the SAME session")
    ap.add_argument("-o", "--out", required=True, help="output negatives CSV")
    args = ap.parse_args()

    intervals = stopped_intervals(args.telemetry)
    total_s = sum(b - a for a, b in intervals) / 1e9
    print(f"GNSS-confirmed stopped: {len(intervals)} intervals, {total_s / 60:.1f} min "
          f"(after {EDGE_TRIM_S}s edge trim, min {MIN_INTERVAL_S}s)")
    if not intervals:
        raise SystemExit("no usable stopped intervals — was GNSS live during this session?")

    d = imu_rows(args.trip)
    t = d[:, 0].astype(np.int64)
    lo, hi = t.min(), t.max()
    print(f"trip trace: {len(d)} IMU rows spanning {(hi - lo) / 1e9 / 60:.1f} min")

    overlap = [(a, b) for a, b in intervals if b > lo and a < hi]
    if not overlap:
        raise SystemExit(
            "trip trace and telemetry do not overlap in time — different sessions?\n"
            f"  trip      {lo} .. {hi}\n"
            f"  telemetry {intervals[0][0]} .. {intervals[-1][1]}"
        )

    mask = np.zeros(len(d), bool)
    for a, b in overlap:
        mask |= (t >= a) & (t <= b)
    kept = d[mask]
    if not len(kept):
        raise SystemExit("no IMU samples fell inside a confirmed-stopped interval")

    with open(args.out, "w") as f:
        for row in kept:
            f.write("IMU," + ",".join(
                (f"{int(row[0])}" if i == 0 else f"{row[i]:.6f}") for i in range(10)
            ) + "\n")

    span = (kept[:, 0].max() - kept[:, 0].min()) / 1e9
    print(f"wrote {len(kept)} IMU rows -> {args.out}")

    # Usable yield, not raw row count. A window is 50 samples at 10 Hz, so every window needs 5
    # CONTIGUOUS seconds inside one confirmed-stopped interval. Short intervals contribute nothing,
    # and reporting rows rather than windows would make an unusable file look productive.
    WIN_S = 5.0
    windows = sum(max(0, int(((b - a) / 1e9 - WIN_S) * 10)) for a, b in overlap)
    stopped_min = sum(b - a for a, b in overlap) / 1e9 / 60
    print(f"  {stopped_min:.1f} min of stopped-with-engine-running -> ~{windows} training windows")
    if windows < 500:
        print()
        print(f"  NOT ENOUGH TO RETRAIN. Roughly 500+ windows (about 5 min of stopped time) is the")
        print(f"  minimum worth fine-tuning on, and more matters less than VARIETY — aim for ~10")
        print(f"  separate stops rather than one long one.")
        print()
        print(f"  The usual cause is blackout probes: while GNSS is muted there is no ground truth,")
        print(f"  so those minutes cannot label anything. Collecting negatives and running blackout")
        print(f"  probes are mutually exclusive uses of the same ride.")
        return
    print("\nNext: add it to finetune.py's negatives glob and retrain as v3.")
    print("Keep it SEPARATE from the hand-held negatives when reporting results — the whole point")
    print("is that these two classes are different, so their per-class error should be read apart.")


if __name__ == "__main__":
    main()
