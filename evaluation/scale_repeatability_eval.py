"""
Scale-stability / repeatability analysis
========================================
Consumes evaluation/scale_repeatability.csv, logged by pressing 'j' in
core/aruco_detect.py once per trial (read scale → disturb the setup → read → …).

Validation showed the ArUco-derived scale — not pixel quantisation or lens
distortion — is the dominant accuracy limit: it drifts a few % between sessions
/ after disturbing the rig.  This script quantifies that drift.

For each scale column it reports mean, std (1σ), CV% and range.  The CV% of the
warp scale ≈ the multiplicative measurement error you should expect on any
distance after a disturbance (e.g. CV 3% → a true 100 cm reads 100 ± 3 cm).

Usage:
  python evaluation/scale_repeatability_eval.py
"""

import argparse
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "evaluation" / "scale_repeatability.csv"

COLUMNS = [
    ("scale_x_cmpx",   "warp scale x"),
    ("scale_y_cmpx",   "warp scale y"),
    ("camera_gsd_cmpx", "camera GSD"),
    ("table_w_cm",     "table width cm"),
    ("table_h_cm",     "table height cm"),
]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No scale log at {args.csv}.")
        print("Run core/aruco_detect.py, press 'j' each trial (read → disturb → read).")
        return
    rows = list(csv.DictReader(open(args.csv)))
    if len(rows) < 2:
        print(f"Need ≥2 trials in {args.csv}; found {len(rows)}.")
        return

    print(f"Loaded {len(rows)} trial(s) from {args.csv}\n")
    print("=" * 66)
    print("SCALE REPEATABILITY  (spread of the auto-scale across disturbances)")
    print("=" * 66)
    print(f"{'quantity':<16}{'n':>3}{'mean':>11}{'std(1σ)':>11}{'CV%':>8}{'range':>10}")
    print("-" * 66)
    for col, name in COLUMNS:
        vals = [_f(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) < 2:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        cv = 100.0 * std / mean if mean else float("nan")
        rng = max(vals) - min(vals)
        print(f"{name:<16}{n:>3}{mean:>11.4f}{std:>11.4f}{cv:>7.2f}%{rng:>10.4f}")
    print("-" * 66)
    print("CV% of the warp scale ≈ the % measurement error to expect on any")
    print("distance after a disturbance (this is the system's scale repeatability).")


if __name__ == "__main__":
    main()
