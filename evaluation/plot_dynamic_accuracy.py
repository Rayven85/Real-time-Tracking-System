"""
Figure: dynamic tracking accuracy vs. robot speed.

Reads the traj_*.csv runs grouped by speed (slow/medium/fast) and plots:
  left  — cross-track RMS vs. speed: how much the tracked path deviates from the
          ideal straight line, i.e. the dynamic (motion) error;
  right — measured displacement per run against the tape truth, showing both
          accuracy and run-to-run spread.

Usage:  python evaluation/plot_dynamic_accuracy.py
"""

import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evaluation" / "dynamic_accuracy.png"
TRUTH_CM = 100.0

# Speed groups: (label, [run ids]) — ids are the traj_<id>.csv timestamps.
GROUPS = [
    ("Slow",   ["155048", "155125", "155204", "155226"]),
    ("Medium", ["190209", "190233", "190300", "190327", "190356"]),
    ("Fast",   ["190545", "190603", "190621", "190640", "190659"]),
]

# Validated categorical palette (slots 1-3, light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def load(path):
    ts, xs, ys = [], [], []
    for r in csv.DictReader(open(path)):
        try:
            ts.append(float(r["t_s"])); xs.append(float(r["x_cm"])); ys.append(float(r["y_cm"]))
        except (KeyError, ValueError):
            continue
    return ts, xs, ys


def cross_track_rms(xs, ys):
    ax, ay, bx, by = xs[0], ys[0], xs[-1], ys[-1]
    L = math.hypot(bx - ax, by - ay)
    if L < 1:
        return 0.0
    dev = [((px - ax) * (by - ay) - (py - ay) * (bx - ax)) / L for px, py in zip(xs, ys)]
    return math.sqrt(sum(d * d for d in dev) / len(dev))


def main():
    stats = []          # (label, mean_speed, mean_ctr, [speeds], [disps])
    for label, ids in GROUPS:
        speeds, disps, ctrs = [], [], []
        for rid in ids:
            p = ROOT / "evaluation" / f"traj_{rid}.csv"
            if not p.exists():
                continue
            ts, xs, ys = load(p)
            if len(xs) < 8:
                continue
            path = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) for i in range(1, len(xs)))
            speeds.append(path / (ts[-1] - ts[0]))
            disps.append(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
            ctrs.append(cross_track_rms(xs, ys))
        if not speeds:
            continue
        stats.append((label, sum(speeds) / len(speeds),
                      sum(ctrs) / len(ctrs), speeds, disps))

    if not stats:
        print("No trajectory runs found."); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, alpha=0.25, linewidth=0.8)
        for s in ax.spines.values():
            s.set_color("#d8d7d2")
        ax.tick_params(colors=INK_2, labelsize=9)

    # ── Left: cross-track RMS vs speed (single series → no legend) ──
    sp = [s[1] for s in stats]
    ct = [s[2] for s in stats]
    ax1.plot(sp, ct, "-", color=COLORS[0], linewidth=2, zorder=2)
    ax1.scatter(sp, ct, s=90, color=COLORS[0], zorder=3,
                edgecolors=SURFACE, linewidths=2)
    for (label, x, y, _, _) in stats:      # direct labels, not a number per point
        ax1.annotate(f"{label}\n{y:.2f} cm", (x, y), textcoords="offset points",
                     xytext=(0, 13), ha="center", fontsize=9, color=INK_2)
    ax1.set_xlabel("Robot speed (cm/s)", color=INK_2, fontsize=10)
    ax1.set_ylabel("Cross-track RMS (cm)", color=INK_2, fontsize=10)
    ax1.set_title("Dynamic error vs. speed", color=INK, fontsize=11, pad=12)
    ax1.set_ylim(0, max(ct) * 1.6)
    ax1.set_xlim(min(sp) - 2.5, max(sp) + 2.5)   # room for the end labels

    # ── Right: displacement per run vs the 100 cm truth (3 series → legend) ──
    ax2.axhline(TRUTH_CM, color=INK_2, linestyle="--", linewidth=1.4,
                zorder=1, label=f"True distance ({TRUTH_CM:.0f} cm)")
    for i, (label, _, _, speeds, disps) in enumerate(stats):
        ax2.scatter(speeds, disps, s=70, color=COLORS[i], zorder=3,
                    edgecolors=SURFACE, linewidths=2, label=f"{label} runs")
    ax2.set_xlabel("Robot speed (cm/s)", color=INK_2, fontsize=10)
    ax2.set_ylabel("Measured displacement (cm)", color=INK_2, fontsize=10)
    ax2.set_title("Measured distance vs. truth", color=INK, fontsize=11, pad=12)
    # Headroom above the data so the legend never sits on top of a run.
    all_d = [d for s in stats for d in s[4]]
    ax2.set_ylim(min(all_d) - 1.5, max(all_d) + 5.5)
    leg = ax2.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    print(f"Plot → {OUT}")
    for label, s, c, _, disps in stats:
        m = sum(disps) / len(disps)
        sd = math.sqrt(sum((d - m) ** 2 for d in disps) / (len(disps) - 1))
        print(f"  {label:7s} {s:5.1f} cm/s   cross-track {c:.3f} cm   "
              f"disp {m:6.2f} ± {sd:.2f} cm  ({m - TRUTH_CM:+.1f} vs truth)")


if __name__ == "__main__":
    main()
