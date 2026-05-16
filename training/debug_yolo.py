"""
Quick YOLO sign detection debugger.
Loads one training image, crops the 3 ROIs, runs the model at conf=0.05,
and prints every detection. Use this to see if the model is actually producing
any output at all, and at what confidence.

Usage:
    python debug_yolo.py
    python debug_yolo.py --img track_dataset/images/train/aruco_001.jpg
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(ROOT / "runs/detect/runs/train/track_signs/weights/best.pt")

ROI_LIGHT = (10,  252, 72,  315)
ROI_STOP  = (305, 472, 402, 550)
ROI_55    = (468, 283, 568, 362)

ROIS = {
    'light': ROI_LIGHT,
    'stop':  ROI_STOP,
    'speed': ROI_55,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", default=None, help="Path to a 600x600 warped screenshot")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold")
    args = parser.parse_args()

    # Pick image
    if args.img:
        img_path = Path(args.img)
    else:
        candidates = list((ROOT / "track_dataset/images/train").glob("*.jpg"))
        if not candidates:
            print("No images found in track_dataset/images/train/")
            return
        img_path = candidates[0]

    print(f"Image: {img_path}")
    img = cv2.imread(str(img_path))
    if img is None:
        print("Cannot read image")
        return
    print(f"Size: {img.shape[1]}x{img.shape[0]}")

    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Classes: {model.names}\n")

    print("Running YOLO on full 600x600 image...")
    results = model(img, verbose=False, conf=args.conf)
    all_detections = []
    for r in results:
        for box, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            name = model.names[int(c)]
            bx = float((box[0] + box[2]) / 2)
            by = float((box[1] + box[3]) / 2)
            all_detections.append((name, float(conf), bx, by))
            print(f"  DETECTED: {name}  conf={float(conf):.3f}  centre=({bx:.0f},{by:.0f})")

    if not all_detections:
        print(f"  No detections at conf>={args.conf}")

    print()
    for roi_name, (x1, y1, x2, y2) in ROIS.items():
        in_roi = [(n, c) for n, c, bx, by in all_detections
                  if x1 <= bx <= x2 and y1 <= by <= y2]
        print(f"── ROI [{roi_name}] ({x1},{y1})-({x2},{y2}):")
        if in_roi:
            for n, c in in_roi:
                print(f"   {n}  conf={c:.3f}  ✓ inside ROI")
        else:
            print(f"   Nothing inside ROI")

        # Save crop for visual inspection
        crop = img[y1:y2, x1:x2]
        crop_path = f"debug_{roi_name}_crop.jpg"
        cv2.imwrite(crop_path, crop)
        print(f"   Crop saved → {crop_path}")

    print("\nDone. Open the debug_*_crop.jpg files to check what the model sees.")


if __name__ == "__main__":
    main()
