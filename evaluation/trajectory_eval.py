"""
Trajectory / dynamic tracking-accuracy analysis
================================================
Reads a traj_*.csv logged by pressing 't' in core/aruco_detect.py while the
robot drives a KNOWN physical path (e.g. a tape-measured straight line between
two flat marks, kept in the central region to avoid edge distortion).

Reports, from the tracked positions (cm):
  • displacement  — straight-line start→end distance
  • path length   — summed frame-to-frame distance (≈ displacement if straight)
  • duration / average speed
  • cross-track RMS — RMS perpendicular deviation from the ideal start→end line,
    i.e. how straight/clean the tracked path is (tracking jitter under motion)

Pass --truth <cm> (the tape-measured path length) to get distance accuracy.

Usage:
  python evaluation/trajectory_eval.py evaluation/traj_150312.csv --truth 80.0
  python evaluation/trajectory_eval.py evaluation/traj_150312.csv --truth 80 --plot
"""

import argparse
import csv
import math
import os


def load(path):
    xs, ys, ts = [], [], []
    for r in csv.DictReader(open(path)):
        try:
            ts.append(float(r["t_s"])); xs.append(float(r["x_cm"])); ys.append(float(r["y_cm"]))
        except (KeyError, ValueError):
            continue
    return ts, xs, ys


def cross_track_rms(xs, ys):
    """RMS perpendicular distance of each point from the start→end line (cm).
    Only meaningful if the TRUE path is straight (else it just measures how much
    the robot curved).  For curved paths use jitter_rms instead."""
    ax, ay, bx, by = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return 0.0
    # signed perpendicular distance via 2D cross product / |AB|
    dev = [((px - ax) * dy - (py - ay) * dx) / L for px, py in zip(xs, ys)]
    return math.sqrt(sum(d * d for d in dev) / len(dev))


def jitter_rms(xs, ys, win=5):
    """
    RMS deviation of each tracked point from a moving-average-smoothed path.

    Real motion is smooth, so high-frequency wiggle around the smoothed path is
    tracking jitter — independent of whether the path is straight or curved.
    Compare slow vs. fast runs: the increase is the dynamic (motion) error.
    """
    n = len(xs)
    if n < win + 2:
        return float("nan")
    h = win // 2
    def smooth(v):
        return [sum(v[max(0, i - h):min(n, i + h + 1)]) /
                (min(n, i + h + 1) - max(0, i - h)) for i in range(n)]
    sx, sy = smooth(xs), smooth(ys)
    dev = [math.hypot(xs[i] - sx[i], ys[i] - sy[i]) for i in range(n)]
    return math.sqrt(sum(d * d for d in dev) / n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="a traj_*.csv from the 't' key")
    ap.add_argument("--truth", type=float, default=None,
                    help="tape-measured true path length (cm) for accuracy")
    ap.add_argument("--plot", action="store_true", help="save a 2D path figure")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No file: {args.csv}"); return
    ts, xs, ys = load(args.csv)
    if len(xs) < 2:
        print("Need ≥2 tracked points."); return

    disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    path = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))
    dur = ts[-1] - ts[0]
    spd = path / dur if dur > 0 else 0.0
    ctr = cross_track_rms(xs, ys)
    jit = jitter_rms(xs, ys)

    print("=" * 60)
    print(f"TRAJECTORY  {os.path.basename(args.csv)}   ({len(xs)} points)")
    print("=" * 60)
    print(f"  start → end displacement : {disp:8.2f} cm")
    print(f"  path length (summed)     : {path:8.2f} cm")
    print(f"  duration                 : {dur:8.2f} s")
    print(f"  average speed            : {spd:8.2f} cm/s")
    print(f"  jitter RMS (any path)         : {jit:6.3f} cm   ← dynamic error metric")
    print(f"  cross-track RMS (straight only): {ctr:6.3f} cm")
    if args.truth:
        for name, val in [("displacement", disp), ("path length", path)]:
            e = val - args.truth
            pe = 100 * e / args.truth if args.truth else float("nan")
            print(f"  {name:<12} vs truth {args.truth:.1f} cm: "
                  f"err {e:+.2f} cm ({pe:+.1f}%)")
        print("  (for a straight path use displacement; path length is inflated by jitter)")

    if not args.plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping figure"); return
    out = os.path.splitext(args.csv)[0] + ".png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(xs, ys, "-", color="tab:blue", alpha=0.6, label="tracked path")
    ax.scatter(xs, ys, s=10, color="tab:blue")
    ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], "k--", alpha=0.7, label="ideal start→end")
    ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], c=["green", "red"], s=60, zorder=5)
    ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Tracked path  (disp {disp:.1f} cm, cross-track RMS {ctr:.2f} cm)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"Plot → {out}")


if __name__ == "__main__":
    main()
