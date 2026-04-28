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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default="cpu")
    opt = parser.parse_args()

    # Use CUDA if available, otherwise CPU
    if opt.device != "cpu" and torch.cuda.is_available():
        device = opt.device
    else:
        device = "cpu"
        if opt.device != "cpu":
            print("[INFO] No CUDA GPU — using CPU")

    data_yaml = Path("track_data.yaml")
    if not data_yaml.exists():
        print("[ERROR] track_data.yaml not found")
        return

    # Check dataset exists
    for split in ("train", "val"):
        d = Path("track_dataset/images") / split
        if not d.exists() or not any(d.iterdir()):
            print(f"[ERROR] No images in {d}")
            print("  Step 1: Put screenshots in track_dataset/images/train/ and val/")
            print("  Step 2: Run  python auto_label.py")
            print("  Step 3: Run  python auto_label.py --split val")
            return

    label_count = sum(1 for _ in Path("track_dataset/labels/train").glob("*.txt"))
    if label_count == 0:
        print("[ERROR] No label files found in track_dataset/labels/train/")
        print("  Run: python auto_label.py")
        return

    print(f"Dataset: {label_count} labelled training images")
    print(f"Device : {device}")
    print(f"Epochs : {opt.epochs}\n")

    model = YOLO("yolo11n.pt")   # start from ImageNet pretrained, NOT existing fine-tuned
    results = model.train(
        data     = str(data_yaml),
        epochs   = opt.epochs,
        imgsz    = opt.imgsz,
        batch    = opt.batch,
        device   = device,
        project  = "runs/train",
        name     = "track_signs",   # saves to runs/train/track_signs/ — separate from yolov11n
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
