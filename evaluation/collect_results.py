"""
从各模型的 results.csv 中提取最佳指标，更新 runs/comparison_results.csv
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "runs/detect/runs/train"

MODELS = {
    "yolov5nu":  BASE / "yolov5nu",
    "yolov8n":   BASE / "yolov8n",
    "yolov9t":   BASE / "yolov9t",
    "yolov10n":  BASE / "yolov10n",
    "yolov11n":  BASE / "yolov11n",
}

# 别名映射（用于 comparison_results.csv 中的展示名）
DISPLAY_NAMES = {
    "yolov5nu": "yolov5nu",
    "yolov8n":  "yolov8n",
    "yolov9t":  "yolov9t",
    "yolov10n": "yolov10n",
    "yolov11n": "yolov11n",
}

all_results = []

for name, run_dir in MODELS.items():
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"[SKIP] {name} — results.csv not found at {csv_path}")
        continue

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    # 去除列名中的空格
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]

    # 找最高 mAP50 的行
    best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0)))

    weights_path = run_dir / "weights" / "best.pt"
    result = {
        "model":      DISPLAY_NAMES[name],
        "mAP50":      float(best_row["metrics/mAP50(B)"]),
        "mAP50-95":   float(best_row["metrics/mAP50-95(B)"]),
        "precision":  float(best_row["metrics/precision(B)"]),
        "recall":     float(best_row["metrics/recall(B)"]),
        "weights":    str(weights_path),
    }
    all_results.append(result)
    print(f"[OK] {name:10s}  mAP50={result['mAP50']:.4f}  mAP50-95={result['mAP50-95']:.4f}  "
          f"P={result['precision']:.4f}  R={result['recall']:.4f}")

if not all_results:
    print("No results found.")
else:
    out = ROOT / "runs/comparison_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model","mAP50","mAP50-95","precision","recall","weights"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved {len(all_results)} models to {out}")

    print(f"\n{'='*70}")
    print(f"  {'Model':<12} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 54)
    for r in sorted(all_results, key=lambda x: x["mAP50"], reverse=True):
        print(f"  {r['model']:<12} {r['mAP50']:>8.4f} {r['mAP50-95']:>10.4f} "
              f"{r['precision']:>10.4f} {r['recall']:>8.4f}")
    print(f"{'='*70}")
