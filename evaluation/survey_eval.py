"""
Marker-survey analysis — radial distortion and parallax
========================================================
Consumes evaluation/marker_survey.csv, logged with the 'n' key while moving a
marker to positions at different radii from the camera nadir (the crosshair,
'x' key). See docs/next_phase_plan.md, steps 1.3 and 1.4.

Two questions, one dataset:

  FLAT marker (--size <true edge cm>)
    Does apparent size grow with radius? Residual radial distortion stretches
    the periphery, which would inflate the corner markers, shrink the derived
    cm/px and make every distance read short — the suspected source of the
    measured -5.11 % scale offset.

  RAISED marker (--size <true edge cm> --height <h cm> --camera-height <H cm>)
    A marker h above the table is pushed outward from the nadir by H/(H-h).
    Apparent size is inflated by the same factor, so h can be recovered from
    size alone: h = H (1 - S_true / S_apparent).

Usage:
  python evaluation/survey_eval.py --size 10.5
  python evaluation/survey_eval.py --size 10.5 --camera-height 146
  python evaluation/survey_eval.py --size 10.5 --id 7 --plot
"""

import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "evaluation" / "marker_survey.csv"


def load(path, only_id=None):
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            mid = int(r["marker_id"])
            if only_id is not None and mid != only_id:
                continue
            rows.append((mid, float(r["radius_cm"]), float(r["apparent_size_cm"]),
                         float(r["dx_cm"]), float(r["dy_cm"])))
        except (KeyError, ValueError):
            continue
    return rows


def linfit(xs, ys):
    """Least-squares slope/intercept of y = a + b x."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--size", type=float, required=True,
                    help="the marker's TRUE printed edge length (cm)")
    ap.add_argument("--camera-height", type=float, default=None,
                    help="H (cm) — enables solving for the marker height h")
    ap.add_argument("--id", type=int, default=None, help="analyse only this marker id")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No survey log at {args.csv}.")
        print("Run aruco_detect.py, press 'x' for the crosshair, 'n' at each position.")
        return
    rows = load(args.csv, args.id)
    if len(rows) < 3:
        print(f"Need >=3 survey points; found {len(rows)}.")
        return

    radii = [r[1] for r in rows]
    sizes = [r[2] for r in rows]

    print("=" * 66)
    print(f"MARKER SURVEY  ({len(rows)} points, true edge {args.size} cm)")
    print("=" * 66)
    print(f"{'id':>3}{'radius':>9}{'dx':>8}{'dy':>8}{'apparent':>10}{'vs true':>9}")
    print("-" * 66)
    for mid, r, s, dx, dy in sorted(rows, key=lambda x: x[1]):
        print(f"{mid:>3}{r:>9.1f}{dx:>8.1f}{dy:>8.1f}{s:>10.3f}"
              f"{100 * (s / args.size - 1):>8.1f}%")
    print("-" * 66)

    a, b = linfit(radii, sizes)
    # Predicted apparent size at the nadir, and its growth per 10 cm of radius
    print(f"Fit: apparent = {a:.3f} + {b:.5f} x radius   (cm)")
    print(f"  at nadir (r=0): {a:.3f} cm   vs true {args.size} cm "
          f"→ {100 * (a / args.size - 1):+.2f}%")
    print(f"  growth: {100 * b * 10 / args.size:+.2f}% per 10 cm of radius")

    if abs(b) * max(radii) < 0.01 * args.size:
        print("\n  → Apparent size is essentially FLAT with radius:")
        print("    radial distortion is NOT the cause of the scale offset.")
    else:
        print("\n  → Apparent size varies with radius: residual radial distortion.")
        print("    This inflates the peripheral corner markers and biases cm/px.")

    if args.camera_height:
        H = args.camera_height
        mean_s = sum(sizes) / len(sizes)
        h = H * (1 - args.size / mean_s)
        print(f"\nHeight from apparent size (mean {mean_s:.3f} cm):")
        print(f"  h = H(1 - S_true/S_apparent) = {H}(1 - {args.size}/{mean_s:.3f}) "
              f"= {h:.2f} cm")
        print(f"  parallax magnification H/(H-h) = {H / (H - h):.4f} "
              f"({100 * (H / (H - h) - 1):+.2f}%)")
        print(f"  correction factor to apply = {(H - h) / H:.4f}")

    if not args.plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping figure"); return
    out = ROOT / "evaluation" / "marker_survey.png"
    SURFACE, INK_2 = "#fcfcfb", "#52514e"
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(args.size, color=INK_2, ls="--", lw=1.4, label=f"True size ({args.size} cm)")
    ax.scatter(radii, sizes, s=70, color="#2a78d6", zorder=3,
               edgecolors=SURFACE, linewidths=2, label="Measured")
    xs = [0, max(radii) * 1.05]
    ax.plot(xs, [a + b * x for x in xs], color="#eb6834", lw=2, zorder=2, label="Linear fit")
    ax.set_xlabel("Radius from camera nadir (cm)", color=INK_2)
    ax.set_ylabel("Apparent marker size (cm)", color=INK_2)
    ax.set_title("Apparent marker size vs. distance from nadir")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"\nPlot → {out}")


if __name__ == "__main__":
    main()
