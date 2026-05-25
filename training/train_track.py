"""
Train a track-specific YOLOv11n sign detector.
Fine-tunes from the generic yolo11n.pt pretrained weights.
Results saved to runs/train/track_signs/ — does NOT overwrite existing models.

Usage:
    python train_track.py
    python train_track.py --epochs 50 --device cpu
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    opt = parser.parse_args()

    # Use CUDA if available, MPS for Apple Silicon, otherwise CPU
    if opt.device == "mps":
        device = "mps"
    elif opt.device != "cpu" and torch.cuda.is_available():
        device = opt.device
    else:
        device = "cpu"
        if opt.device not in ("cpu", "mps"):
            print("[INFO] No CUDA GPU — using CPU")

    last_pt = ROOT / "runs/train/track_signs/weights/last.pt"

    if opt.resume:
        if not last_pt.exists():
            print(f"[ERROR] No checkpoint to resume from: {last_pt}")
            return
        print(f"Resuming from {last_pt}")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
    else:
        data_yaml = ROOT / "track_data.yaml"
        if not data_yaml.exists():
            print("[ERROR] track_data.yaml not found")
            return

        for split in ("train", "val"):
            d = ROOT / "track_dataset/images" / split
            if not d.exists() or not any(d.iterdir()):
                print(f"[ERROR] No images in {d}")
                return

        label_count = sum(1 for _ in (ROOT / "track_dataset/labels/train").glob("*.txt"))
        if label_count == 0:
            print("[ERROR] No label files found in track_dataset/labels/train/")
            return

        print(f"Dataset: {label_count} labelled training images")
        print(f"Device : {device}")
        print(f"Epochs : {opt.epochs}\n")

        model = YOLO(str(ROOT / "weights/yolo11n.pt"))
        results = model.train(
            data     = str(data_yaml),
            epochs   = opt.epochs,
            imgsz    = opt.imgsz,
            batch    = opt.batch,
            device   = device,
            project  = str(ROOT / "runs/train"),
            name     = "track_signs",
            patience = 15,
            exist_ok = True,
            verbose  = True,
        )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        print(f"\n{'='*50}")
        print(f"  Training complete!")
        print(f"  Best weights: {best}")
        print(f"\n  To use the new model, update aruco_detect.py:")
        print(f'  SIGN_MODEL_PATH = "{best}"')
        print(f"{'='*50}")
    else:
        print(f"\n[WARN] best.pt not found at {best}")


if __name__ == "__main__":
    main()
