"""
Multi-model YOLO comparison training script.
Supports: YOLOv5nu, YOLOv8n, YOLOv9t, YOLOv10n, YOLOv11n
All models are loaded via the ultralytics package — no separate yolov5 install needed.
"""

import argparse
import csv
from pathlib import Path
from ultralytics import YOLO
import torch


def auto_device(requested: str) -> str:
    """Return the requested device if CUDA available, else fall back to 'cpu'."""
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return requested
    print("[INFO] No CUDA GPU detected — falling back to CPU.")
    return "cpu"


# ── Model registry ────────────────────────────────────────────────────────────
# "yolov5nu" is the ultralytics-native port of YOLOv5n (same architecture,
#  fully compatible with Python 3.14 + PyTorch 2.6, no separate package needed)
MODELS = {
    "yolov5nu":  "yolov5nu.pt",
    "yolov8n":   "yolov8n.pt",
    "yolov9t":   "yolov9t.pt",
    "yolov10n":  "yolov10n.pt",
    "yolov11n":  "yolo11n.pt",   # ultralytics drops the 'v': yolo11n, not yolov11n
}

TRAIN_ARGS = dict(
    data     = "data.yaml",
    epochs   = 100,    # 50 for quick test, 100 for comparison, 150+ for full training
    imgsz    = 640,
    batch    = 16,     # reduce to 8 if memory runs out on CPU
    workers  = 4,
    device   = "0",   # overridden at runtime by auto_device()
    patience = 30,    # early-stop: halt if no improvement for 30 epochs
    exist_ok = True,
    verbose  = False,
)


def train_model(name: str, weights: str, args: dict) -> dict:
    """Train one model and return its final metrics."""
    print(f"\n{'='*60}")
    print(f"  Training {name}  ({weights})")
    print(f"{'='*60}")

    model   = YOLO(weights)
    results = model.train(project="runs/train", name=name, **args)

    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    return {
        "model":     name,
        "mAP50":     metrics.get("metrics/mAP50(B)"),
        "mAP50-95":  metrics.get("metrics/mAP50-95(B)"),
        "precision": metrics.get("metrics/precision(B)"),
        "recall":    metrics.get("metrics/recall(B)"),
        "weights":   str(Path("runs/train") / name / "weights" / "best.pt"),
    }


def print_summary(all_results: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("  COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<12} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 52)
    for r in all_results:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "  N/A  "
        print(f"{r['model']:<12} {fmt(r['mAP50']):>8} {fmt(r['mAP50-95']):>10} "
              f"{fmt(r['precision']):>10} {fmt(r['recall']):>8}")
    print(f"{'='*70}")
    print("Best weights: runs/train/<model>/weights/best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train multiple YOLO models for comparison")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help=f"Models to train. Choices: {list(MODELS.keys())}")
    parser.add_argument("--epochs",       type=int, default=TRAIN_ARGS["epochs"])
    parser.add_argument("--imgsz",        type=int, default=640)
    parser.add_argument("--batch",        type=int, default=16)
    parser.add_argument("--device",       default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--skip-trained", action="store_true",
                        help="Skip models whose best.pt already exists")
    opt = parser.parse_args()

    device = auto_device(opt.device)
    args   = {**TRAIN_ARGS, "epochs": opt.epochs, "imgsz": opt.imgsz,
              "batch": opt.batch, "device": device}

    all_results = []
    for model_name in opt.models:
        if model_name not in MODELS:
            print(f"[WARN] Unknown model '{model_name}'. Available: {list(MODELS.keys())}")
            continue

        best_pt = Path("runs/train") / model_name / "weights" / "best.pt"
        if opt.skip_trained and best_pt.exists():
            print(f"[SKIP] {model_name} — best.pt already exists")
            continue

        r = train_model(model_name, MODELS[model_name], args)
        all_results.append(r)

    if all_results:
        print_summary(all_results)
        out_csv = Path("runs/comparison_results.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["model", "mAP50", "mAP50-95", "precision", "recall", "weights"])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to  {out_csv}")


if __name__ == "__main__":
    main()
