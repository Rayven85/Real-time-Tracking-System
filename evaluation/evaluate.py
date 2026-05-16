"""
Evaluate trained models on the test set and generate a comparison report.
Usage:
    python evaluate.py                         # evaluate all trained models
    python evaluate.py --models yolov8n yolov9t
    python evaluate.py --plot                  # also generate metric plots
"""

import argparse
import csv
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parent.parent

def auto_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested
    print("[INFO] No CUDA GPU detected — falling back to CPU.")
    return "cpu"


TRAINED_MODELS = {
    "yolov5nu": str(ROOT / "runs/train/yolov5nu/weights/best.pt"),
    "yolov8n":  str(ROOT / "runs/train/yolov8n/weights/best.pt"),
    "yolov9t":  str(ROOT / "runs/train/yolov9t/weights/best.pt"),
    "yolov10n": str(ROOT / "runs/train/yolov10n/weights/best.pt"),
    "yolov11n": str(ROOT / "runs/train/yolov11n/weights/best.pt"),
}


def evaluate_model(name: str, weights: str, data: str, imgsz: int, device: str) -> dict:
    """Run val on a single model and return metrics dict."""
    print(f"\n  Evaluating {name} ...")
    model   = YOLO(weights)
    metrics = model.val(
        data    = data,
        imgsz   = imgsz,
        device  = device,
        split   = "test",   # use test split; falls back to val if no test key in yaml
        verbose = False,
    )
    d = metrics.results_dict
    return {
        "model":    name,
        "mAP50":    d.get("metrics/mAP50(B)"),
        "mAP50-95": d.get("metrics/mAP50-95(B)"),
        "precision":d.get("metrics/precision(B)"),
        "recall":   d.get("metrics/recall(B)"),
        "weights":  weights,
    }


def plot_comparison(results: list[dict], out_dir: Path) -> None:
    """Bar-chart comparison of mAP50 across models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in results]

    metrics_to_plot = ["mAP50", "mAP50-95", "precision", "recall"]
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(16, 4))
    fig.suptitle("YOLO Model Comparison — Test Set", fontsize=14)

    for ax, metric in zip(axes, metrics_to_plot):
        values = [r.get(metric) or 0 for r in results]
        bars = ax.bar(models, values, color="steelblue", edgecolor="black")
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {out_path}")


def print_table(results: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("  EVALUATION RESULTS  (test set)")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 52)
    for r in results:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "  N/A  "
        print(f"{r['model']:<12} {fmt(r['mAP50']):>8} {fmt(r['mAP50-95']):>10} "
              f"{fmt(r['precision']):>10} {fmt(r['recall']):>8}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", default=list(TRAINED_MODELS.keys()))
    parser.add_argument("--data",    default=str(ROOT / "data.yaml"))
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--device",  default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--plot",    action="store_true", help="Generate comparison plots")
    opt = parser.parse_args()
    device = auto_device(opt.device)

    results = []
    for name in opt.models:
        weights = TRAINED_MODELS.get(name)
        if not weights or not Path(weights).exists():
            print(f"[SKIP] {name} — weights not found at {weights}")
            continue
        r = evaluate_model(name, weights, opt.data, opt.imgsz, device)
        results.append(r)

    if not results:
        print("No models evaluated. Run train.py first.")
        return

    print_table(results)

    out_dir = ROOT / "runs/evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "test_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model","mAP50","mAP50-95","precision","recall","weights"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    if opt.plot:
        plot_comparison(results, out_dir)


if __name__ == "__main__":
    main()
