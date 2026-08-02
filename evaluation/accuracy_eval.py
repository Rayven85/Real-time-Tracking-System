"""
Accuracy + repeatability analysis (Tasks 2 & 4)
================================================
Consumes evaluation/measurements.csv, logged live by pressing 'l' in
core/aruco_detect.py (each row = one A–B distance the software measured, with an
optional tape-measured ground truth and a label).

Two analyses, auto-selected from the data:

  ACCURACY (Task 2) — every row that has a ground_truth_cm:
      error = measured − actual.  Reports bias (systematic offset), MAE, RMSE,
      max error and mean %error, plus a measured-vs-actual scatter against the
      ideal y = x line.  Mark fixed points across the table (centre AND edges —
      perspective error is worst at the periphery).

  REPEATABILITY (Task 4) — any label repeated ≥3 times (disturb the setup, then
      re-measure the SAME real distance):
      reports mean, std (1-σ = repeatability), CV% and range of the measured
      values.  Compare std against the analytical GSD from precision_analysis.py
      to see whether the scatter approaches the pixel-quantisation floor.

Usage:
  python evaluation/accuracy_eval.py
  python evaluation/accuracy_eval.py --csv evaluation/measurements.csv
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "evaluation" / "measurements.csv"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_measured"] = _f(r.get("measured_cm"))
        r["_gt"] = _f(r.get("ground_truth_cm"))
    return [r for r in rows if r["_measured"] is not None]


def accuracy(rows):
    pts = [r for r in rows if r["_gt"] is not None]
    if not pts:
        print("\n[ACCURACY] no rows with ground_truth_cm — fill that column to enable.")
        return None
    print("\n" + "=" * 64)
    print("ACCURACY  (measured vs. tape-measured actual)")
    print("=" * 64)
    print(f"{'label':<16}{'measured':>10}{'actual':>9}{'err':>8}{'%err':>8}")
    print("-" * 64)
    errs, pcts = [], []
    for r in sorted(pts, key=lambda x: x["_gt"]):
        e = r["_measured"] - r["_gt"]
        p = 100.0 * e / r["_gt"] if r["_gt"] else float("nan")
        errs.append(e); pcts.append(p)
        print(f"{r.get('label',''):<16}{r['_measured']:>10.2f}{r['_gt']:>9.2f}"
              f"{e:>+8.2f}{p:>+7.1f}%")
    n = len(errs)
    bias = sum(errs) / n
    mae = sum(abs(e) for e in errs) / n
    rmse = math.sqrt(sum(e * e for e in errs) / n)
    max_e = max(errs, key=abs)
    mean_pct = sum(abs(p) for p in pcts) / n
    print("-" * 64)
    print(f"n={n}   bias={bias:+.2f} cm   MAE={mae:.2f} cm   RMSE={rmse:.2f} cm")
    print(f"max|err|={abs(max_e):.2f} cm   mean|%err|={mean_pct:.1f}%")
    print("bias = systematic offset (scale error); RMSE = overall accuracy.")
    return pts


def repeatability(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("label", "")].append(r["_measured"])
    repeated = {k: v for k, v in groups.items() if len(v) >= 3}
    if not repeated:
        print("\n[REPEATABILITY] no label repeated ≥3× — reuse one label across "
              "disturbed runs to enable.")
        return
    print("\n" + "=" * 64)
    print("REPEATABILITY  (same distance, setup disturbed between runs)")
    print("=" * 64)
    print(f"{'label':<18}{'n':>3}{'mean':>9}{'std(1σ)':>10}{'CV%':>7}{'range':>9}")
    print("-" * 64)
    for label, vals in sorted(repeated.items()):
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        cv = 100.0 * std / mean if mean else float("nan")
        rng = max(vals) - min(vals)
        print(f"{label:<18}{n:>3}{mean:>9.2f}{std:>10.3f}{cv:>6.2f}%{rng:>9.3f}")
    print("-" * 64)
    print("std (1σ) is the repeatability; compare it to the analytical GSD")
    print("from precision_analysis.py (the pixel-quantisation floor).")


def plot(pts, out):
    if not pts:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping figure")
        return
    actual = [r["_gt"] for r in pts]
    measured = [r["_measured"] for r in pts]
    lo, hi = min(actual + measured), max(actual + measured)
    pad = 0.05 * (hi - lo + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", alpha=0.6, label="ideal y=x")
    a1.scatter(actual, measured, c="tab:blue", zorder=3)
    a1.set_xlabel("Actual distance (cm)"); a1.set_ylabel("Measured distance (cm)")
    a1.set_title("Measured vs. actual"); a1.grid(True, alpha=0.3); a1.legend()
    resid = [m - g for m, g in zip(measured, actual)]
    a2.axhline(0, color="k", ls="--", alpha=0.6)
    a2.scatter(actual, resid, c="tab:red", zorder=3)
    a2.set_xlabel("Actual distance (cm)"); a2.set_ylabel("Error (cm)")
    a2.set_title("Residuals"); a2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nPlot → {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(ROOT / "evaluation" / "accuracy_scatter.png"))
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"No measurement log at {args.csv}.")
        print("Run core/aruco_detect.py, press 'p' to measure, 'l' to log points.")
        return
    rows = load_rows(args.csv)
    if not rows:
        print(f"{args.csv} has no usable rows yet.")
        return
    print(f"Loaded {len(rows)} measurement(s) from {args.csv}")
    pts = accuracy(rows)
    repeatability(rows)
    plot(pts, args.out)


if __name__ == "__main__":
    main()
