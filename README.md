# Software Architecture — Real-time Vehicle Tracking System

## System Architecture

The system is a single-process Python application running on Mac (development) and targeting Nvidia Jetson Orin (deployment). All computation happens locally — no network required. The main loop in `core/aruco_detect.py` drives everything; a daemon thread handles the slow track-mask computation without blocking the video stream.

```
┌─────────────────────────────────────────────────────────────────┐
│                   GoPro USB Webcam  (Wide FOV)                  │
│                   cv2.VideoCapture + CAP_AVFOUNDATION           │
└────────────────────────────┬────────────────────────────────────┘
                             │  raw frame  (~1080p)
                             ▼
                  ┌─────────────────────┐
                  │   ArUco Detection   │  IDs 0–3 → corner points
                  │   DICT_4X4_50       │  ID  4   → car position
                  └──────────┬──────────┘
                             │  homography M  (recomputed each frame)
                             ▼
                  ┌─────────────────────┐
                  │  Perspective Warp   │  raw → 600×600 px
                  │  warpPerspective    │  orthographic top-down view
                  └────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌──────────────────┐      ┌───────────────────────┐
   │  Vehicle Track   │      │   YOLO Sign Detect    │
   │  ArUco ID 4      │      │   every 8th frame     │
   │                  │      │   (frame-skip cache)  │
   │  query mask at   │      │   full 600×600 image  │
   │  car position    │      │   → ROI centre filter │
   └────────┬─────────┘      │   → STOP / 55 / LIGHT │
            │                └───────────────────────┘
   ON / OFF TRACK
   8-frame debounce
   ROI short-circuit
            │
            └──────────────────────────────────────┐
                                                   │
                  ┌────────────────────────────────▼────┐
                  │  Background Daemon Thread            │
                  │  auto_detect_track_mask()            │
                  │  • Dual-polarity Otsu threshold      │
                  │  • Morphological close + open        │
                  │  • _bridge_sign_roi() sticker fix    │
                  │  • Re-triggers on perspective shift  │
                  └──────────────────────────────────────┘
```

### Layer Map / 层级模块对应

| Layer | Technology | File |
|---|---|---|
| Camera capture | `cv2.VideoCapture` + `CAP_AVFOUNDATION` | `core/aruco_detect.py` |
| Marker detection | `cv2.aruco` DICT_4X4_50 | `core/aruco_detect.py` |
| Perspective warp | `cv2.getPerspectiveTransform` | `core/aruco_detect.py` |
| Vehicle localisation | ArUco ID 4 + track mask query | `core/aruco_detect.py` |
| Sign detection | Custom YOLOv11n (5 classes) | `core/aruco_detect.py` |
| Track mask (background) | Dual-polarity Otsu + morphology | `core/aruco_detect.py` |
| Model training — multi-model | Ultralytics YOLO | `training/train.py` |
| Model training — track signs | Ultralytics YOLO fine-tune | `training/train_track.py` |
| Data labelling | HSV bootstrap + mouse editor | `training/auto_label.py` |
| Accuracy evaluation | mAP50, mAP50-95, precision, recall | `evaluation/evaluate.py` |
| Efficiency benchmarking | params, GFLOPs, latency, FPS | `evaluation/benchmark.py` |
| Visualisation | GT vs prediction grids | `evaluation/visualize.py` |
| Camera calibration | Checkerboard + `calibrateCamera` | `camera/calibrate_gopro.py` |
| Distortion tuning | Manual slider tool | `camera/undistort.py` |
| Marker generation | `cv2.aruco.generateImageMarker` | `core/generate_markers.py` |

---

## Project Structure

All scripts use `ROOT = Path(__file__).resolve().parent.parent` to anchor paths to the project root, so they work correctly regardless of the calling directory.

```
TrackingSystem/
│
├── data.yaml            YOLO config: GTSDB 4-class traffic sign dataset
├── track_data.yaml      YOLO config: custom 5-class track sign dataset
├── track_mask.png       Auto-saved track mask (written at runtime by aruco_detect.py)
│
├── core/
├── training/
├── evaluation/
├── camera/
├── tracking/
│
├── weights/
├── calib_images/
├── datasets/
├── track_dataset/
├── gopro_samples/
├── RPi_samples/
├── markers/
├── screenshots/
├── runs/
└── docs/
```

---

### `core/` — Main System

The entry point of the entire project. Contains two scripts.

**`aruco_detect.py`** (768 lines) — The real-time pipeline. All major subsystems live here:

| Section | Lines | Responsibility |
|---|---|---|
| Constants & config | 1–75 | Camera params, ROI coordinates, model path, warp size |
| Camera management | `open_camera()` | Retry loop, CAP_AVFOUNDATION, warm-up frames |
| ArUco detection | `detect_markers()` | Find IDs 0–4, return corners and car position |
| Perspective warp | `get_perspective_transform()`, `warp_point()` | Homography M, point projection |
| Undistortion | `build_undistort_maps()` | Pre-compute remap tables (currently disabled) |
| Track mask (auto) | `auto_detect_track_mask()` | Otsu + morphology + bridge repair |
| Track mask (thread) | `_trigger_mask()`, `_run_mask_detection()` | Non-blocking daemon thread |
| Perspective change | `_perspective_changed()` | 15 px corner-shift threshold |
| On/off track | `is_on_track()` | Mask pixel query + ROI short-circuit |
| Sign detection | `detect_signs()` | YOLO full-image + ROI centre filter |
| Drawing — raw | `draw_raw()` | Overlay markers on original frame |
| Drawing — warped | `draw_warped()` | HUD: car position, ON/OFF, sign boxes |
| Main loop | `main()` | Camera → ArUco → warp → display → keyboard |

**`generate_markers.py`** (52 lines) — Generates 5 printable ArUco PNGs into `markers/`:

| Marker ID | File | Physical placement |
|---|---|---|
| 0 | `corner_0.png` | Table top-left |
| 1 | `corner_1.png` | Table top-right |
| 2 | `corner_2.png` | Table bottom-right |
| 3 | `corner_3.png` | Table bottom-left |
| 4 | `car.png` | Car roof |

---

### `training/` — Model Training & Data Labelling 

**`train.py`** (127 lines) — Trains all five YOLO variants on the GTSDB dataset in one run, then writes a comparison CSV to `runs/comparison_results.csv`.

```
Usage: python training/train.py --models yolov8n yolov11n --epochs 100
```

Base weights are loaded from `weights/`. Trained weights are saved to `runs/train/<model>/weights/best.pt`.

**`train_track.py`** (86 lines) — Fine-tunes `weights/yolo11n.pt` on the track-specific sign dataset (`track_dataset/`). Outputs to `runs/train/track_signs/`.

```
Usage: python training/train_track.py --epochs 50
```

Key parameters: 5 classes (stop / speed_55 / light_off / light_green / light_red), batch=8 (CPU-friendly), patience=15 (early stopping).

**`auto_label.py`** (307 lines) — Semi-automatic YOLO label generator for the 600×600 warped screenshots.

Pipeline: Load screenshot → HSV auto-detection in 3 ROIs → display bounding boxes → keyboard adjustments → save YOLO `.txt` label.

| Key | Action |
|---|---|
| `Enter` | Save & advance to next image |
| `1` / `2` | Toggle STOP / speed_55 box |
| `3` | Cycle light state: off → green → red |
| `e` | Mouse edit mode (`cv2.selectROI`) |
| `d` | Discard image |

Reads from `track_dataset/images/<split>/`, writes labels to `track_dataset/labels/<split>/`.

**`debug_yolo.py`** (94 lines) — Diagnostic tool. Loads one training image, runs the model at conf=0.05, prints every detection with ROI membership, and saves crops to `runs/`. Used to diagnose the domain mismatch issue (cropped ROI input vs full-image input).

```
Usage: python training/debug_yolo.py [--img path/to/image.jpg]
```

---

### `evaluation/` — Model Evaluation & Visualisation

**`evaluate.py`** (142 lines) — Runs `model.val()` on the test split for each trained model, outputs a CSV to `runs/evaluation/test_results.csv`. Optionally generates bar charts (`--plot`).

```
Usage: python evaluation/evaluate.py [--plot]
```

**`benchmark.py`** (214 lines) — Measures model efficiency (parameter count, GFLOPs, model size, CPU latency, FPS) separately from accuracy. Merges lightweight results with accuracy results from `collect_results.py` into `runs/comparison_results_full.csv`.

```
Usage: python evaluation/benchmark.py [--models yolov8n yolov11n] [--runs 100]
```

**`collect_results.py`** (74 lines) — Reads the per-epoch `results.csv` from each model's training directory, extracts the best-epoch row, and writes a unified `runs/comparison_results.csv`. Run automatically at the end of `train.py`, or manually after training.

**`visualize.py`** (265 lines) — Generates side-by-side GT vs prediction images. Three output modes:

| Mode | Output | Use |
|---|---|---|
| `per_model` | One image per model | Quick single-model check |
| `comparison` | All models on one image (different colours) | Visual comparison |
| `grid` | 2×3 grid (GT + 5 models) | Report figures |

```
Usage: python evaluation/visualize.py --mode grid --n 10 --seed 42
```

**`predict.py`** (77 lines) — Runs inference on a single model or all models in parallel on a given source (image dir, video, webcam). Saves annotated results to `runs/predict/<model>/`.

```
Usage: python evaluation/predict.py --compare --source datasets/test/images
```

---

### `camera/` — Camera Tools & Calibration

**`calibrate_gopro.py`** (161 lines) — Two-phase checkerboard calibration tool.

- Phase 1 (first run): generates `calib_images/calibration_checkerboard.png` for printing
- Phase 2 (after photos collected): calls `cv2.calibrateCamera()` on images in `calib_images/`, saves result to `calib_images/gopro_calib.npz`

Target reprojection error: < 1.0 px.

**`undistort.py`** (214 lines) — Interactive distortion correction tuner. Displays original vs corrected side by side with a green grid overlay. Three sliders: K1 (primary barrel), K2 (higher-order), focal scale.

Supports three modes: `--mode wide` (GoPro Wide), `--mode superview` (GoPro SuperView), `--mode rpi` (Raspberry Pi fisheye via `cv2.fisheye`). Manual estimate: k1 = −0.462, k2 = −0.054 — currently disabled in `aruco_detect.py` as overhead distortion is negligible.

**`gopro_distortion.py`** (162 lines) — Compares distortion across GoPro FOV modes. Overlays a green grid on sample images from `gopro_samples/{linear,wide,superview}/`, uses Hough line detection to score straightness, outputs `fov_comparison.jpg`.

**`gopro_latency.py`** (160 lines) — Measures GoPro USB Webcam streaming performance: actual FPS vs declared FPS, frame interval jitter, dropped frame detection. Renders a scrolling bar chart of frame intervals in real time.

**`test.py`** (220 lines) — Multi-mode camera and model testing suite:

| Mode | What it does |
|---|---|
| `camera_test` | Measure actual camera FPS for 5 seconds |
| `detect` | Real-time detection with FPS counter |
| `fps_test` | Stress test: measure per-frame latencies over N frames |
| `conf_sweep` | Interactive confidence threshold tuning (`+` / `-` keys) |

```
Usage: python camera/test.py --mode fps_test --model weights/yolov8n.pt
```

---

### `tracking/` — Object Tracking

**`track.py`** (141 lines) — Real-time multi-object tracking using YOLO + ByteTrack (built into Ultralytics). Maintains trajectory history (last 30 frames per ID) and draws alpha-blended trail lines (older = dimmer). Supports video file input or webcam.

```
Usage: python tracking/track.py [--source 0] [--conf 0.4] [--save]
```

Output video saved to `runs/track_output.mp4` when `--save` is passed.

---

### `weights/` — Pretrained YOLO Weights

ImageNet-pretrained base weights. Used as starting points for training — never used for inference directly.

| File | Used by |
|---|---|
| `yolo11n.pt` | `training/train_track.py` (track sign fine-tune) |
| `yolov5nu.pt` | `training/train.py` (multi-model comparison) |
| `yolov8n.pt` | `training/train.py` |
| `yolov9t.pt` | `training/train.py` |
| `yolov10n.pt` | `training/train.py` |

Trained weights (after running `train.py` / `train_track.py`) are stored separately in `runs/train/<model>/weights/best.pt`.

---

### `calib_images/` — Camera Calibration Data

| File | Description |
|---|---|
| `GOPR*.JPG` | 39 GoPro Wide mode photos of checkerboard from various angles |
| `calibration_checkerboard.png` | 10×7 checkerboard pattern (generated by `camera/calibrate_gopro.py`) |
| `gopro_calib.npz` | Calibration result: camera matrix K + distortion coefficients |

The `.npz` output from calibration is available for loading by `aruco_detect.py` (currently unused — distortion correction disabled).

---

### `datasets/` — GTSDB Evaluation Data

German Traffic Sign Detection Benchmark subset, YOLO format, 4 classes: `danger / mandatory / other / prohibitory`.

Used by `evaluation/evaluate.py`, `evaluation/benchmark.py`, `evaluation/visualize.py`, and `evaluation/predict.py` for test-set evaluation.

---

### `track_dataset/` — Custom Track Sign Dataset

600×600 px warped-view screenshots of the physical race track, labelled with YOLO format annotations.

```
track_dataset/
├── images/
│   ├── train/   (44 images)
│   └── val/     (12 images)
└── labels/
    ├── train/   (44 .txt files)
    └── val/     (12 .txt files)
```

5 classes: `stop / speed_55 / light_off / light_green / light_red`. Created by: run `core/aruco_detect.py`, press `s` to capture screenshots, then run `training/auto_label.py`.

---

### `gopro_samples/` — FOV Comparison Photos

```
gopro_samples/
├── linear/      GoPro Linear mode photos (minimal distortion)
├── wide/        GoPro Wide mode photos (current operating mode)
└── superview/   GoPro SuperView mode photos (severe barrel distortion)
```

Used by `camera/gopro_distortion.py` and `camera/undistort.py`.

---

### `markers/` — Generated ArUco PNGs

Output from `core/generate_markers.py`. 5 files: `corner_0.png` through `corner_3.png` and `car.png`. Print at 4–6 cm for corner markers, 3–4 cm for car marker.

---

### `screenshots/` — Runtime Detection Screenshots

Saved by pressing `s` in `core/aruco_detect.py`. Named sequentially `aruco_001.jpg`, `aruco_002.jpg`, … These are the source images for `training/auto_label.py`.

---

### `runs/` — Training & Evaluation Outputs

All output from training, evaluation, and inference scripts is written here. Never committed manually — all generated at runtime.

```
runs/
├── train/                         Output of training/train.py
│   ├── yolov5nu/weights/best.pt
│   ├── yolov8n/weights/best.pt
│   ├── yolov9t/weights/best.pt
│   ├── yolov10n/weights/best.pt
│   └── yolov11n/weights/best.pt
├── detect/runs/train/             Output when YOLO saves to detect/ subdirectory
│   └── track_signs/weights/best.pt    ← loaded by aruco_detect.py
├── evaluation/
│   └── test_results.csv           Output of evaluation/evaluate.py
├── visualization/                 Output of evaluation/visualize.py
├── predict/                       Output of evaluation/predict.py
├── comparison_results.csv         Unified accuracy table
├── comparison_results_full.csv    Accuracy + efficiency merged
└── lightweight_results.csv        Efficiency metrics only
```

---

### `docs/` — Project Documentation

| File / Folder | Content |
|---|---|
| `project_log.md` | Phase-by-phase technical decisions, issues encountered, resolutions |
| `ReadingNote.md` | Background reading notes (YOLO, ByteTrack, TensorRT, ROS/Gazebo) |
| `ReadingNoteImage/` | Images referenced in reading notes |
| `workingrecord/` | Notion export of weekly progress notes (Week 3 through Week 11) |

---

## Development Timeline

### Week 3–5: YOLO Model Baseline Comparison

**Dataset:** GTSDB (German Traffic Sign Detection Benchmark), YOLO format, 4 classes.

Five nano-scale YOLO variants trained for 100 epochs each on identical hardware:

| Model | mAP50 | mAP50-95 | Precision | Recall | Params (M) | GFLOPs | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|---|
| YOLOv5nu | 0.854 | 0.420 | **0.931** | 0.755 | 2.51 | 7.2 | 65.9 | 15.2 |
| YOLOv8n | 0.873 | **0.443** | 0.860 | **0.838** | 3.01 | 8.2 | 61.1 | 16.4 |
| YOLOv9t | 0.862 | 0.431 | 0.940 | 0.768 | **2.01** | 7.9 | 88.1 | 11.4 |
| YOLOv10n | 0.820 | 0.402 | 0.823 | 0.786 | 2.71 | 8.4 | 85.8 | 11.7 |
| **YOLOv11n ✓** | **0.881** | 0.440 | 0.883 | 0.807 | 2.59 | **6.4** | **59.1** | **16.9** |

**Decision:** YOLOv11n — highest mAP50, lowest latency (59.1 ms), fewest GFLOPs.

---

### Week 5–7: Camera, Calibration & Marker System

#### Camera Selection

Initial Logitech USB webcam (1280×720 @ 24 FPS) couldn't cover the full table even at maximum mount height — fixed focal length, insufficient FOV. Replaced with GoPro in USB Webcam mode.

GoPro USB Webcam mode supports three FOV modes:

| FOV Mode | Distortion | Available via USB? | Decision |
|---|---|---|---|
| Linear | Minimal | ✗ | Offline analysis only |
| **Wide** | Slight barrel | ✓ | **Selected** |
| SuperView | Severe barrel | ✓ | Rejected |

#### Distortion Correction

`camera/undistort.py` — Interactive tool: side-by-side original vs corrected with green grid overlay, three sliders (K1, K2, focal scale). Manual estimate: **k1 = −0.462, k2 = −0.054**.

`camera/calibrate_gopro.py` — Formal checkerboard pipeline: generates print pattern, then computes precise K matrix and distortion coefficients from 39 calibration photos. Output: `calib_images/gopro_calib.npz`.

At the current overhead mount height, Wide mode barrel distortion is negligible — manually estimated correction worsened the image. **Distortion correction is currently disabled** in `aruco_detect.py`.

#### ArUco vs AprilTag

| | AprilTag | **ArUco (selected)** |
|---|---|---|
| Library | `pupil-apriltags` (extra install) | Built into OpenCV |
| Speed | Slower | Faster |
| Docs | Academic papers | Full OpenCV docs |
| Close-range | Higher robustness | Sufficient for controlled setup |

Dictionary: `DICT_4X4_50` — simplest pattern (16 data bits), easiest to detect at close overhead range.

#### Perspective Warp

The oblique camera angle produces a trapezoidal view. Once the four corner markers (IDs 0–3) are detected, `cv2.getPerspectiveTransform` computes homography matrix `M`. Each frame is warped into a **600×600 px orthographic top-down view**. `M` is recomputed every frame — minor camera movement is automatically compensated.

#### ON / OFF TRACK Detection

**Problem:** The ArUco marker on the car physically covers the track beneath it. Real-time pixel sampling at the car's position reads the marker pattern, not the track — always reports OFF TRACK.

**Solution:** Before placing the car, press `c` to trigger `auto_detect_track_mask()` on the clean (car-free) warped view. The result is saved to `track_mask.png`. All subsequent queries check this pre-saved mask, not the live image.


**Debounce:** 8-frame rolling window, >60% majority required to update ON/OFF state. Prevents flickering when the car drives near the track edge.

---

### Week 7–9: Custom YOLO Sign Detection

Three stages of development:

**Stage 1 — HSV colour thresholds (abandoned):** Fixed ROI regions + red/green pixel ratio thresholds. Brittle to lighting changes; LED colour boundaries ambiguous; ROI coordinate calibration tedious. Abandoned.

**Stage 2 — Pre-trained YOLO model (failed):** Applied the GTSDB-trained YOLOv11n directly to track signs. Domain mismatch: training data = large real street signs; target = A4-printed miniatures (~3×3 cm, ~100×80 px in warped view). Zero detections even at conf=0.05.

**Stage 3 — Bootstrap custom training (successful):**

Bootstrap strategy: use imprecise HSV detections as an annotation seed → refine manually with `auto_label.py` → train YOLO from scratch on track-domain images.

Dataset: 44 training + 12 validation images (600×600 px warped screenshots). 5 classes:

| Class ID | Name | Description |
|---|---|---|
| 0 | `stop` | STOP sign visible |
| 1 | `speed_55` | Speed limit 55 sign visible |
| 2 | `light_off` | Traffic light off |
| 3 | `light_green` | Green light on |
| 4 | `light_red` | Red light on |

Training results: mAP50 = **0.995**, mAP50-95 = 0.873, Precision ~0.97, Recall ~0.96.

**Integration bug — cropped ROI vs full image:**

First attempt sent each ROI crop individually to YOLO. The model was trained on full 600×600 frames — in a crop, the sign fills the entire input. Distribution mismatch → zero detections even at conf=0.05. Diagnosed with `training/debug_yolo.py`.

**Fix:** Send the **complete 600×600 warped image** to YOLO, then filter detections by ROI centre position:

```python
for box, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
    bx = float((box[0] + box[2]) / 2)
    by = float((box[1] + box[3]) / 2)
    if roi[0] <= bx <= roi[2] and roi[1] <= by <= roi[3]:
        best[name] = (conf_f, box)
```

Immediate result: `stop=0.907, light_off=0.825, speed_55=0.475`.

**Frame-skip caching:** YOLO CPU inference ≈ 100–300 ms/call. Running every frame → ~5 FPS. Solution: run every 8th frame, cache the result for intermediate frames → **~25 FPS**.

---

### Week 9–11: Automatic Track Mask Detection

Previously required manual `c` keypress. Supervisor required full automation with adaptive re-detection.

**Pipeline:**

```
raw frame → perspective warp (600×600) → downsample (300×300)
→ Gaussian blur → Otsu dual-polarity threshold → morphological clean
→ upsample (600×600) → _bridge_sign_roi() → save track_mask.png
```

**Dual-polarity Otsu:** Standard Otsu only tries one binary direction. Track polarity (dark-on-light vs light-on-dark) can vary across setups. Both directions are tried; the one with a white-pixel ratio closest to 30% (typical oval track) is selected. No manual threshold needed.

**Half-resolution processing:** Detection runs at 300×300 (75% fewer pixels), upscaled to 600×600. Reduces CPU time enough to run in a background daemon thread without blocking the main loop.

**`_bridge_sign_roi()` — sticker occlusion fix:**

Black adhesive stickers under the three road signs match the track colour. Otsu classifies them as track; morphological close expands them into thick rectangular blobs. Four approaches were tried:

| Version | Strategy | Outcome |
|---|---|---|
| v1 | Paint ROI as table colour before Otsu → fill → draw line | Right-side track deleted; Y/T artefacts |
| v2 | Paint only car ArUco, not sign ROIs | Sticker blobs remain |
| v3 | Scan 18 px ROI perimeter → find entry points → bridge with straight line | Stable — clean thin lines |
| v4 | Skeletonisation + branch pruning | Complete failure — edge truncation erodes inward |

Current solution (v3): clear ROI interior, find the two most-distant track entry points on the ROI perimeter, connect with one straight line (avoids Y/T junctions).

**Non-blocking background thread:**

`auto_detect_track_mask()` takes ~200–400 ms on CPU. Daemon thread + single-element shared list pattern:

```python
_mask_running = False
_mask_result  = [None]   # thread-safe write via single-element list

# main loop: non-blocking check
if _mask_result[0] is not None:
    saved_mask = _mask_result[0]
    _mask_result[0] = None
```

**Perspective change re-detection:** The homography matrix `_mask_M` from the last detection is stored. Each frame, all four image corners are projected through both `_mask_M` and the current `M`. If any corner shifts more than **15 px**, re-detection is triggered.

**ROI short-circuit in `is_on_track()`:** When the car is within a sign ROI zone (where stickers can cause mask gaps), the function returns `True` immediately without querying the mask — these zones are always on-track by construction.

---

## Key Technical Decisions

| Decision | Rationale | Alternative Rejected |
|---|---|---|
| GoPro over Logitech | Wide FOV covers full table from single overhead mount | Logitech: fixed focal length, insufficient FOV even at max height |
| Wide FOV (not Linear) | Linear unavailable in USB Webcam mode | Linear: offline-only, unavailable for real-time streaming |
| ArUco over AprilTag | Native OpenCV, no extra install, faster at close range | AprilTag: third-party library, slower detection |
| `DICT_4X4_50` | Simplest 16-bit pattern, easiest to detect at close overhead range | Larger dicts: slower, unnecessary for this use case |
| Pre-capture track mask | Car ArUco marker physically covers track beneath it | Real-time pixel sampling: marker occludes track → always OFF TRACK |
| Custom YOLO training | Real-world sign models cannot generalise to miniature A4-printed signs | Direct transfer: domain gap too large, zero detections |
| Full-image inference + ROI filter | Model trained on full 600×600 frames — crops create distribution mismatch | Crop-then-detect: domain mismatch → zero detections |
| Frame-skip caching (every 8 frames) | CPU inference ~200 ms; skipping recovers ~25 FPS | Per-frame inference: ~5 FPS, unusable |
| Dual-polarity Otsu | Track polarity unknown across setups; one direction would fail for some | Single-polarity: requires manual threshold tuning per setup |
| Background daemon thread | Mask detection ~300 ms on CPU — synchronous call freezes video stream | Synchronous: ~3 FPS, unacceptable |
| Disable distortion correction | Wide mode distortion negligible at overhead height; correction degraded image | Apply manual k1/k2: estimated values too aggressive at this mount height |
| `_bridge_sign_roi` v3 | Skeletonisation (v4) failed due to edge erosion; simple bridging is robust | Skeletonisation + branch pruning: unstable when track has edge gaps |

---

## Results Summary

### YOLO Model Comparison — GTSDB

| Model | mAP50 ↑ | mAP50-95 ↑ | Latency (ms) ↓ | FPS ↑ | Params (M) ↓ |
|---|---|---|---|---|---|
| YOLOv5nu | 0.854 | 0.420 | 65.9 | 15.2 | 2.51 |
| YOLOv8n | 0.873 | 0.443 | 61.1 | 16.4 | 3.01 |
| YOLOv9t | 0.862 | 0.431 | 88.1 | 11.4 | 2.01 |
| YOLOv10n | 0.820 | 0.402 | 85.8 | 11.7 | 2.71 |
| **YOLOv11n ✓** | **0.881** | **0.440** | **59.1** | **16.9** | 2.59 |

### Custom Track Sign Model

| Metric | Value |
|---|---|
| mAP50 | **0.995** |
| mAP50-95 | 0.873 |
| Precision | ~0.97 |
| Recall | ~0.96 |
| Training set | 44 images |
| Validation set | 12 images |

### Runtime Performance

| Component | FPS / Latency |
|---|---|
| YOLO sign detection — per frame (no cache) | ~5 FPS |
| YOLO sign detection — every 8th frame (cached) | **~25 FPS** |
| Track mask detection (background thread) | ~200–400 ms, non-blocking |
| Perspective change detection | per-frame, negligible cost |
