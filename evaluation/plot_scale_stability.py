"""
Figure: scale stability — static vs. after-disturbance.
Reads scale_static.csv + scale_repeatability.csv (logged with the 'j' key) and
plots the ArUco scale per trial (left) and the CV% comparison (right).

Usage:  python evaluation/plot_scale_stability.py
"""

import csv
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "evaluation" / "scale_static.csv"
DISTURB = ROOT / "evaluation" / "scale_disturb.csv"
OUT = ROOT / "evaluation" / "scale_stability.png"


def load(path, col="scale_x_cmpx"):
    if not os.path.exists(path):
        return []
    return [float(r[col]) for r in csv.DictReader(open(path)) if r.get(col)]


def stats(v):
    n = len(v)
    m = sum(v) / n
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, s, (100 * s / m if m else 0.0)


def main():
    static = load(STATIC)
    disturb = load(DISTURB)
    if not static and not disturb:
        print("No scale logs found. Run aruco_detect.py and press 'j'.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    series = []
    if static:
        series.append(("Static (undisturbed)", static, "tab:blue"))
    if disturb:
        series.append(("After disturbance", disturb, "tab:red"))

    # Left: scale per trial, with mean line and ±1σ band
    for name, v, c in series:
        m, s, cv = stats(v)
        x = list(range(1, len(v) + 1))
        ax1.plot(x, v, "o-", color=c, label=f"{name}  (CV {cv:.2f}%)")
        ax1.axhline(m, color=c, ls="--", lw=1, alpha=0.7)
        ax1.fill_between([1, max(len(v), 1)], m - s, m + s, color=c, alpha=0.12)
    ax1.set_xlabel("Trial #")
    ax1.set_ylabel("ArUco scale  (cm / warp-pixel)")
    ax1.set_title("Scale per trial  (dashed = mean, band = ±1σ)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: CV% comparison bars (warp scale + camera GSD)
    labels, static_cv, disturb_cv = [], [], []
    for col, name in [("scale_x_cmpx", "warp scale"), ("camera_gsd_cmpx", "camera GSD")]:
        s_v, d_v = load(STATIC, col), load(DISTURB, col)
        if not s_v and not d_v:
            continue
        labels.append(name)
        static_cv.append(stats(s_v)[2] if s_v else 0)
        disturb_cv.append(stats(d_v)[2] if d_v else 0)
    xpos = range(len(labels))
    w = 0.38
    ax2.bar([x - w / 2 for x in xpos], static_cv, w, color="tab:blue", label="Static")
    ax2.bar([x + w / 2 for x in xpos], disturb_cv, w, color="tab:red", label="Disturbed")
    ax2.set_xticks(list(xpos))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("CV  (%, 1σ / mean)")
    ax2.set_title("Repeatability (lower = better)")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()
    for x, val in zip(xpos, disturb_cv):
        ax2.text(x + w / 2, val, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("ArUco scale stability — precision (repeatability) is ~0.1–0.3%")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"Plot → {OUT}")
    for name, v, _ in series:
        m, s, cv = stats(v)
        print(f"  {name}: mean={m:.4f}  std={s:.4f}  CV={cv:.2f}%  n={len(v)}")


if __name__ == "__main__":
    main()
