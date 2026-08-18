"""
Figure: dynamic tracking accuracy vs. robot speed (parked-endpoint runs).

Reads the traj_*.csv runs recorded with the target held still at both ends, so
each run's A->B distance is independent of when 't' was pressed, and plots:
  left  — measured displacement per run against the tape truth, showing the
          run-to-run spread (repeatability) and the systematic offset;
  right — cross-track RMS vs. speed: deviation of the tracked path from the
          ideal straight line.

Usage:  python evaluation/plot_dynamic_accuracy.py
"""

import csv
import glob
import math
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
from trajectory_eval import stationary_endpoints, cross_track_rms  # noqa: E402

OUT = ROOT / "evaluation" / "dynamic_accuracy.png"
PATTERN = str(ROOT / "evaluation" / "traj_20260818_*.csv")
TRUTH_CM = 94.0
# Runs started from the wrong mark (~101 cm) — excluded by the operator.
EXCLUDE = {"181450", "181526", "181554", "181625"}

# Speed bands (cm/s) for colouring, validated categorical palette slots 1-3
BANDS = [(0, 4.5, "Slow", "#2a78d6"),
         (4.5, 7.5, "Medium", "#eb6834"),
         (7.5, 99, "Fast", "#1baf7a")]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def load_runs():
    runs = []
    for f in sorted(glob.glob(PATTERN)):
        rid = os.path.basename(f).replace("traj_20260818_", "").replace(".csv", "")
        if rid in EXCLUDE:
            continue
        ts, xs, ys = [], [], []
        for r in csv.DictReader(open(f)):
            ts.append(float(r["t_s"])); xs.append(float(r["x_cm"])); ys.append(float(r["y_cm"]))
        A, B, _, _ = stationary_endpoints(ts, xs, ys)
        if A is None:                     # no parked frames — endpoints not captured
            continue
        path = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))
        runs.append((path / (ts[-1] - ts[0]),
                     math.hypot(B[0] - A[0], B[1] - A[1]),
                     cross_track_rms(xs, ys)))
    return runs


def band_of(speed):
    for lo, hi, name, colour in BANDS:
        if lo <= speed < hi:
            return name, colour
    return BANDS[-1][2], BANDS[-1][3]


def main():
    runs = load_runs()
    if len(runs) < 2:
        print("Not enough parked-endpoint runs found."); return
    disp = [r[1] for r in runs]
    mean, sd = statistics.mean(disp), statistics.stdev(disp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        for s in ax.spines.values():
            s.set_color("#d8d7d2")
        ax.tick_params(colors=INK_2, labelsize=9)

    # ── Left: measured displacement per run, vs truth and vs the mean ──
    ax1.axhline(TRUTH_CM, color=INK_2, linestyle="--", linewidth=1.4, zorder=1,
                label=f"True distance ({TRUTH_CM:.0f} cm)")
    ax1.axhspan(mean - sd, mean + sd, color="#9a9a94", alpha=0.15, zorder=0)
    ax1.axhline(mean, color=INK_2, linestyle=":", linewidth=1.2, zorder=1,
                label=f"Mean {mean:.2f} ± {sd:.2f} cm (1σ)")
    seen = set()
    for spd, d, _ in runs:
        name, colour = band_of(spd)
        ax1.scatter(spd, d, s=70, color=colour, zorder=3, edgecolors=SURFACE,
                    linewidths=2, label=None if name in seen else f"{name} runs")
        seen.add(name)
    ax1.set_xlabel("Robot speed (cm/s)", color=INK_2, fontsize=10)
    ax1.set_ylabel("Measured displacement (cm)", color=INK_2, fontsize=10)
    ax1.set_title(f"Measured distance vs. truth  (n={len(runs)})",
                  color=INK, fontsize=11, pad=12)
    ax1.set_ylim(TRUTH_CM - 1.5, max(disp) + 2.2)
    leg = ax1.legend(fontsize=8, framealpha=0.9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_2)

    # ── Right: cross-track RMS vs speed ──
    for spd, _, ct in runs:
        _, colour = band_of(spd)
        ax2.scatter(spd, ct, s=70, color=colour, zorder=3,
                    edgecolors=SURFACE, linewidths=2)
    ax2.set_xlabel("Robot speed (cm/s)", color=INK_2, fontsize=10)
    ax2.set_ylabel("Cross-track RMS (cm)", color=INK_2, fontsize=10)
    ax2.set_title("Path deviation vs. speed", color=INK, fontsize=11, pad=12)
    ax2.set_ylim(0, max(r[2] for r in runs) * 1.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    print(f"Plot → {OUT}")
    print(f"  n={len(runs)}  mean={mean:.2f} ± {sd:.2f} cm  "
          f"CV={100 * sd / mean:.2f}%  vs truth {mean - TRUTH_CM:+.2f} cm")


if __name__ == "__main__":
    main()
