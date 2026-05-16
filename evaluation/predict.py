"""
Run inference with a trained model on images, video, or webcam.
Usage:
    python predict.py --model yolov8n --source datasets/test/images
    python predict.py --model yolov11n --source 0              # webcam
    python predict.py --model yolov9t  --source video.mp4
    python predict.py --compare --source datasets/test/images  # side-by-side all models
"""

import argparse
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


def predict_single(name: str, weights: str, source: str, conf: float, device: str) -> None:
    model = YOLO(weights)
    model.predict(
        source  = source,
        conf    = conf,
        device  = device,
        save    = True,
        project = str(ROOT / "runs/predict"),
        name    = name,
        exist_ok= True,
    )
    print(f"  Results saved to runs/predict/{name}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolov8n", choices=list(TRAINED_MODELS.keys()),
                        help="Model to use for prediction")
    parser.add_argument("--source",  default=str(ROOT / "datasets/test/images"),
                        help="Image/video path, directory, or webcam index (0)")
    parser.add_argument("--conf",    type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device",  default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--compare", action="store_true",
                        help="Run all trained models on the same source")
    opt = parser.parse_args()
    device = auto_device(opt.device)

    if opt.compare:
        for name, weights in TRAINED_MODELS.items():
            if not Path(weights).exists():
                print(f"[SKIP] {name} — {weights} not found")
                continue
            print(f"\n  Running {name} ...")
            predict_single(name, weights, opt.source, opt.conf, device)
    else:
        weights = TRAINED_MODELS[opt.model]
        if not Path(weights).exists():
            print(f"[ERROR] {weights} not found. Train {opt.model} first.")
            return
        predict_single(opt.model, weights, opt.source, opt.conf, device)


if __name__ == "__main__":
    main()
