"""
Analytical precision of the overhead tracking system (Task 3)
=============================================================
Kevin's ask: "calculate precision using analytical means - i.e. using pixels in
the frame vs distance seen by the camera for a few different heights."

The fundamental precision limit of any pixel-based measurement is the
GROUND SAMPLING DISTANCE (GSD) — how many cm of real table one pixel covers at
the table plane.  For an overhead (nadir) pinhole camera at height H:

        GSD(H) = H / f_live          [cm per pixel]

where f_live is the focal length in PIXELS at the live streaming resolution.

  • f_live from calibration:  f_calib (px) is measured at the calibration
    resolution; it scales linearly with image width, so
        f_live = f_calib * (W_live / W_calib).
  • f_live anchored empirically:  the ArUco scale already gives the REALISED
    cm/px at the real mounting height (measured in core/aruco_detect.py).  Pass
    it with --measured-gsd / --measured-height and the script back-solves
        f_eff = H_measured / GSD_measured
    which is the ground-truth focal length for the live stream (accounts for the
    GoPro webcam FOV/crop differing from the photo-mode calibration).

Reported per height:
  • GSD            theoretical 1-px precision (cm/px) — worse as you go higher
  • effective      GSD * sub-pixel factor (ArUco corner refinement ≈ 0.1-0.3 px)
  • coverage       GSD * frame_width (cm) — the precision-vs-coverage trade-off

Usage:
  python evaluation/precision_analysis.py
  python evaluation/precision_analysis.py --width 1920 --heights 60 80 100 120 150
  python evaluation/precision_analysis.py --measured-gsd 0.105 --measured-height 95
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIB = ROOT / "calib_images" / "gopro_calib.npz"


def load_focal_px(calib_path):
    """Return (f_calib_px, calib_width_px) from the saved GoPro intrinsics."""
    data = np.load(calib_path)
    K = data["K"]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    f_calib = 0.5 * (fx + fy)                 # ≈ isotropic after the aspect fix
    calib_w = int(data["img_size"][0])
    return f_calib, calib_w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", default=str(DEFAULT_CALIB),
                    help="camera intrinsics .npz (K, img_size)")
    ap.add_argument("--width", type=int, default=1920,
                    help="live stream width in px (GoPro webcam mode)")
    ap.add_argument("--heights", type=float, nargs="+",
                    default=[60, 80, 100, 120, 150],
                    help="camera heights above the table (cm)")
    ap.add_argument("--subpx", type=float, default=0.2,
                    help="ArUco sub-pixel corner uncertainty (px), default 0.2")
    ap.add_argument("--measured-gsd", type=float, default=None,
                    help="empirical cm/px from the ArUco scale (optional anchor)")
    ap.add_argument("--measured-height", type=float, default=None,
                    help="height (cm) at which --measured-gsd was observed")
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "precision_vs_height.png"))
    ap.add_argument("--csv", default=str(ROOT / "evaluation" / "precision_vs_height.csv"))
    ap.add_argument("--measured-csv", default=None,
                    help="overlay measured GSD per height (a scale log with height_cm + camera_gsd_cmpx)")
    args = ap.parse_args()

    f_calib, calib_w = load_focal_px(args.calib)
    f_live = f_calib * (args.width / calib_w)

    print("=" * 68)
    print("ANALYTICAL PRECISION  (overhead pinhole, GSD = H / f_live)")
    print("=" * 68)
    print(f"Calibration focal length : {f_calib:8.1f} px  @ {calib_w}px wide")
    print(f"Live stream width        : {args.width} px")
    print(f"Theoretical f_live       : {f_live:8.1f} px")

    # Optional empirical anchor from the ArUco-measured scale.
    f_used, source = f_live, "calibration (theoretical)"
    if args.measured_gsd and args.measured_height:
        f_eff = args.measured_height / args.measured_gsd
        err = 100.0 * (f_eff - f_live) / f_live
        print(f"Empirical f_eff          : {f_eff:8.1f} px   "
              f"(from {args.measured_gsd} cm/px @ {args.measured_height} cm)")
        print(f"  → theory vs empirical differs by {err:+.1f}%  "
              f"(FOV/crop of webcam mode vs photo calibration)")
        f_used, source = f_eff, "empirical (ArUco-anchored)"
    print(f"Focal length used        : {f_used:8.1f} px   [{source}]")
    print(f"Sub-pixel corner factor  : {args.subpx} px")
    print("-" * 68)

    header = f"{'Height':>8} {'GSD':>10} {'eff.prec':>10} {'coverage':>12}"
    units  = f"{'(cm)':>8} {'(cm/px)':>10} {'(mm)':>10} {'(cm wide)':>12}"
    print(header); print(units); print("-" * 68)

    rows = []
    for H in args.heights:
        gsd = H / f_used                       # cm per pixel at the table plane
        eff_mm = gsd * args.subpx * 10.0       # effective precision in mm
        coverage_cm = gsd * args.width         # table span the frame can cover
        print(f"{H:8.0f} {gsd:10.4f} {eff_mm:10.3f} {coverage_cm:12.1f}")
        rows.append([H, round(gsd, 4), round(gsd * 10, 3),
                     round(eff_mm, 3), round(coverage_cm, 1)])
    print("-" * 68)
    print("eff.prec = best-case 1-σ with sub-pixel ArUco refinement.")
    print("coverage = widest table the frame spans at that height (precision↔FOV).")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["height_cm", "gsd_cm_per_px", "gsd_mm_per_px",
                    "eff_precision_mm", "coverage_cm"])
        w.writerows(rows)
    print(f"\nCSV  → {args.csv}")

    # Plot (optional — skipped cleanly if matplotlib is absent).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping figure")
        return

    H = np.array(args.heights, dtype=float)
    gsd = H / f_used
    coverage = gsd * args.width

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(H, gsd * 10, "o-", color="tab:blue", label="GSD (mm/px)")
    ax1.set_xlabel("Camera height above table (cm)")
    ax1.set_ylabel("Precision  —  GSD (mm/px)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(H, coverage, "s--", color="tab:red", label="Coverage (cm)")
    ax2.set_ylabel("Coverage — table width in frame (cm)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Overlay empirically-measured GSD per height, if provided.
    if args.measured_csv and os.path.exists(args.measured_csv):
        from collections import defaultdict
        byh = defaultdict(list)
        for r in csv.DictReader(open(args.measured_csv)):
            try:
                h, g = float(r.get("height_cm") or "nan"), float(r.get("camera_gsd_cmpx") or "nan")
            except ValueError:
                continue
            if h == h and g == g:
                byh[h].append(g)
        if byh:
            mh = sorted(byh)
            mg = [sum(byh[h]) / len(byh[h]) * 10 for h in mh]   # mm/px
            ax1.plot(mh, mg, "D", color="black", markersize=8,
                     label="measured GSD", zorder=6)
            print("Measured GSD overlaid:  " +
                  "  ".join(f"{h:.0f}cm={g:.3f}mm/px" for h, g in zip(mh, mg)))

    ax1.legend(loc="upper left")
    plt.title(f"Precision vs. height  (f_live={f_used:.0f}px @ {args.width}px, "
              f"{source.split()[0]})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"Plot → {args.out}")


if __name__ == "__main__":
    main()
