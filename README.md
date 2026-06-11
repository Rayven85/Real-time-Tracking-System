# Software Architecture — Real-time Vehicle Tracking System

## System Architecture

The system is a single-process Python application running on Mac (development) and targeting Nvidia Jetson Orin (deployment). All computation happens locally — no network required. The main loop in `core/aruco_detect.py` drives everything; two daemon threads handle heavy computation: one for YOLO sign detection, one for track-mask computation.

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
   │                  │      │   (background thread) │
   │  query mask at   │      │   full 600×600 image  │
   │  car position    │      │   imgsz=1280 inference │
   └────────┬─────────┘      │   → STOP / 55 / LIGHT │
            │                └───────────────────────┘
   ON / OFF TRACK
   8-frame debounce
   Sign ROI short-circuit
   (_dynamic_sign_rois)
            │
   ┌────────┴─────────┐
   │ Distance Measure │
   │ press p → A→B    │
   │ press k → calib  │
   │ (ArUco corners)  │
   └──────────────────┘
            │
            └──────────────────────────────────────┐
                                                   │
                  ┌────────────────────────────────▼────┐
                  │  Background Daemon Thread            │
                  │  auto_detect_track_mask()            │
                  │  • YOLO (fresh) → sign positions     │
                  │  • Dual-polarity Otsu + adaptive     │
                  │  • Blank sign+tape areas (TAPE_PAD)  │
                  │  • Distance-transform ridge skeleton │
                  │  • Fixed-width dilation (TRACK_HALF) │
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
| Sign detection | Custom YOLOv11n (5 classes, imgsz=1280) | `core/aruco_detect.py` |
| Sign detection (thread) | daemon thread, single-element list handoff | `core/aruco_detect.py` |
| Distance measurement | ArUco-corner calibration + mouse click A→B | `core/aruco_detect.py` |
| Track mask (background) | Dual-polarity Otsu + distance-transform ridge | `core/aruco_detect.py` |
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
├── debug_otsu.png       Debug: binary segmentation before ridge (written at runtime)
├── debug_ridge.png      Debug: distance-transform ridge skeleton (written at runtime)
├── distance_calib.json  Scale calibration: cm/px for x and y axes (written at runtime)
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

**`aruco_detect.py`** — The real-time pipeline. All major subsystems live here:

| Section | Responsibility |
|---|---|
| Constants & config | Camera params, ROI coordinates, model path, warp size, `TRACK_HALF_WIDTH`, `SIGN_CONF` |
| Camera management | `open_camera()` — retry loop, CAP_AVFOUNDATION, warm-up frames |
| ArUco detection | `detect_markers()` — find IDs 0–4, return corners and car position |
| Perspective warp | `get_perspective_transform()`, `warp_point()` — homography M, point projection |
| Undistortion | `build_undistort_maps()` — pre-compute remap tables (currently disabled) |
| Track mask (auto) | `auto_detect_track_mask()` — Otsu + adaptive + ridge skeleton + fixed dilation |
| Track mask (thread) | `_trigger_mask()`, `_run_mask_detection()` — non-blocking daemon thread |
| Perspective change | `_perspective_changed()` — 15 px corner-shift threshold |
| On/off track | `is_on_track()` — mask pixel query + `_dynamic_sign_rois` short-circuit |
| Sign detection | `detect_signs()` — YOLO full-image at imgsz=1280, border filter, positional reclassification |
| Sign detection (thread) | `_trigger_sign_detection()`, `_sign_running`, `_sign_result` — non-blocking YOLO daemon |
| Distance measurement | `_on_mouse()`, `_draw_dist_overlay()`, `_load/save_dist_calib()` — click A→B, display real-world distance |
| Drawing — raw | `draw_raw()` — overlay markers on original frame |
| Drawing — warped | `draw_warped()` — HUD: car position, ON/OFF, sign boxes |
| Main loop | `main()` — camera → ArUco → warp → display → keyboard |

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

**`train_track.py`** (96 lines) — Fine-tunes `weights/yolo11n.pt` on the track-specific sign dataset (`track_dataset/`). Outputs to `runs/train/track_signs/`.

```
Usage: python training/train_track.py --epochs 50
       python training/train_track.py --epochs 50 --device mps   # Apple Silicon
       python training/train_track.py --resume                    # resume from last.pt
```

Key parameters: 5 classes (stop / speed_55 / light_off / light_green / light_red), batch=8 (CPU-friendly), patience=15 (early stopping). Training uses the default `imgsz=640`; inference runs at `imgsz=1280` to upscale small signs at detection time.

Supports `--device mps` for Apple Silicon GPU acceleration and `--resume` to continue from `runs/train/track_signs/weights/last.pt` without creating a new run directory.

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
├── train/                         Output of training scripts
│   ├── yolov5nu/weights/best.pt
│   ├── yolov8n/weights/best.pt
│   ├── yolov9t/weights/best.pt
│   ├── yolov10n/weights/best.pt
│   ├── yolov11n/weights/best.pt
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

**Integration bug — cropped ROI vs full image:**

First attempt sent each ROI crop individually to YOLO. The model was trained on full 600×600 frames — in a crop, the sign fills the entire input. Distribution mismatch → zero detections even at conf=0.05. Diagnosed with `training/debug_yolo.py`.

**Fix:** Send the **complete 600×600 warped image** to YOLO. No spatial ROI filter is applied — the model's confidence threshold is the only gate. This also allows the system to generalise across different table setups where sign positions may vary.

**imgsz=1280 inference:** The signs appear small relative to the 600×600 frame (~100×80 px). Running inference at `imgsz=1280` causes YOLO to internally upscale the image before the backbone, giving the network more pixels per sign. Returned bounding boxes are in the original 600×600 coordinate space. This increased real-detection confidence dramatically:

| Sign | Confidence (old model, imgsz=640) | Confidence (current model, imgsz=1280) |
|---|---|---|
| stop | up to 0.30 | 0.96–0.98 |
| speed_55 | up to 0.30 | 0.72–0.99 |
| light_off | up to 0.30 | 0.63–0.76 |

Noise/ghost detections remain consistently at 0.01–0.06, creating a clear gap. `SIGN_CONF = 0.40` safely sits in this gap.

**Border filter:** Any detection whose bounding box touches within 10 px of the frame edge is discarded — these are always artefacts of the black fill region produced by `warpPerspective` outside the table boundary.

**Positional reclassification:** The STOP sign and speed_55 sign look similar to the model. If a `stop` detection is centred in the bottom half of the frame (y > 55% of height), it is reclassified as `speed_55` — the bottom position is unambiguously the speed sign location in the physical layout.

**YOLO background thread:** YOLO CPU inference ≈ 100–300 ms/call. Running every frame → ~5 FPS. The original solution was to run every 8th frame and cache the result, but this still caused a periodic stutter on the 8th-frame blocking call. YOLO now runs in a background daemon thread (`_trigger_sign_detection()`), firing every 8th frame trigger but executing asynchronously. The main loop reads `_sign_result[0]` without blocking, eliminating the periodic frame stutter entirely → **~25 FPS**.

---

### Week 9–13: Automatic Track Mask Detection

Previously required manual `c` keypress. Supervisor required full automation with adaptive re-detection.

**Segmentation pipeline:**

```
raw frame → perspective warp (600×600) → downsample (300×300)
→ blank ArUco corner markers → CLAHE + Gaussian blur
→ Otsu(Otsu's method) BINARY_INV  ─┐
                     ├─ OR-combine → morphological close + open
→ adaptive threshold ─┘
→ connected-component filter (keep blobs ≥ 3000 px)
→ clear border + corner regions → upsample (600×600)
→ blank sign + tape areas (YOLO bbox + 20 px pad)
→ distance-transform ridge skeleton
→ MORPH_CLOSE (9×9) on ridge → dilate to TRACK_HALF_WIDTH=14
→ final MORPH_CLOSE (15×15) → save track_mask.png
```

**Dual-polarity Otsu + adaptive combine:** Standard Otsu only tries one binary direction. Track polarity (dark-on-light vs light-on-dark) can vary across setups. `BINARY_INV` Otsu handles the typical dark-on-light case; adaptive threshold (blockSize=51, C=8) catches track sections on the right side where glare makes them globally bright but still locally darker than the table surface. The two binary images are OR-combined.

**Half-resolution processing:** Segmentation runs at 300×300 (75% fewer pixels), upscaled to 600×600. Reduces CPU time enough to run in a background daemon thread without blocking the main loop.

**Tape misdetection fix — blank sign areas before ridge:**

Black adhesive tape holding signs to the track creates T/+ shaped blobs in the Otsu binary mask. These blobs survive into the ridge and expand into thick rectangular protrusions during dilation, causing non-uniform track width at sign positions.

Fix: After upsampling, blank a padded rectangle (YOLO bounding box + `TAPE_PAD=20` px on each side) for every `stop` and `speed_55` detection. The tape and sign area becomes zero in the binary mask before the ridge is computed. The gap is left empty — no bridging line is drawn.

**Distance-transform ridge skeleton:**

To achieve uniform track width regardless of the original track line's varying pixel thickness, the mask is reduced to a centerline (1–4 px wide ridge) then re-expanded to a fixed width.

```python
dist         = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
dilated_dist = cv2.dilate(dist, ellipse_kernel(9))
ridge        = (dist >= dilated_dist * 0.85) & (clean > 0)   # local maxima
ridge        = cv2.morphologyEx(ridge, MORPH_CLOSE, ellipse_kernel(9))
clean        = cv2.dilate(ridge, ellipse_kernel(TRACK_HALF_WIDTH*2+1))
clean        = cv2.morphologyEx(clean, MORPH_CLOSE, ellipse_kernel(15))
```

The ridge is the set of pixels where the distance-to-background equals the local maximum within a 9×9 neighbourhood — i.e., the skeleton centerline of the track band. Dilating by `TRACK_HALF_WIDTH=14` gives a uniform 28 px wide track band everywhere.

**`_dynamic_sign_rois` — on-track short-circuit for sign areas:**

Because the sign+tape area is blanked in the mask, the car will show as OFF TRACK when passing through a sign position. To prevent false off-track events, `auto_detect_track_mask()` stores the original YOLO bounding boxes (no padding) of all sign stickers in `_dynamic_sign_rois`. In `is_on_track()`, if the car's warped position falls inside any of these boxes, the function returns `True` immediately without consulting the mask — the sign position is on-track by construction.

Only the sign's actual bounding box counts (not the padded tape area) — this means the car must be squarely on the sign to trigger the short-circuit, matching the physical reality that the track passes directly under each sign.

**Non-blocking background threads:**

There are now two background daemon threads:

1. **YOLO sign detection thread** — `_trigger_sign_detection()` fires a daemon thread on every 8th frame trigger. The thread writes its result to `_sign_result[0]`; the main loop reads and clears this without blocking. `_sign_running` prevents duplicate concurrent threads.

2. **Track mask thread** — `auto_detect_track_mask()` takes ~200–400 ms on CPU. When mask re-detection is triggered (e.g. by a perspective shift), YOLO runs first *synchronously inside the mask thread* to obtain fresh sign positions, then the mask segmentation runs with those results. This guarantees that sign+tape areas are blanked correctly using up-to-date bounding boxes.

Both threads use the daemon thread + single-element shared list pattern:

```python
_mask_running = False
_mask_result  = [None]   # thread-safe write via single-element list

# main loop: non-blocking check
if _mask_result[0] is not None:
    saved_mask = _mask_result[0]
    _mask_result[0] = None
```

**Perspective change re-detection:** The homography matrix `_mask_M` from the last detection is stored. Each frame, all four image corners are projected through both `_mask_M` and the current `M`. If any corner shifts more than **15 px**, re-detection is triggered.

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
| Full-image inference + confidence gate | Model trained on full 600×600 frames; spatial ROI filter rejected in favour of model confidence | Crop-then-detect: domain mismatch; fixed ROI: sign positions differ between tables |
| `imgsz=1280` at inference | Upscales 600×600 image internally; signs ~100×80 px get more backbone pixels | `imgsz=640`: real detections plateau at conf≤0.30, noisy separation from background |
| `SIGN_CONF=0.40` | Clear gap between real detections (≥0.63) and noise (≤0.06); 0.40 sits safely in the middle | Lower threshold: ghost detections pass; higher threshold: may drop light_off (0.63) |
| Frame-skip caching → background thread (every 8 frames trigger) | CPU inference ~200 ms; moved to background thread to eliminate periodic frame stutter; YOLO runs inside mask thread on re-detection to guarantee fresh sign positions for tape blanking | Per-frame inference: ~5 FPS, unusable; synchronous every-8th-frame: recovers FPS but still causes periodic stutter |
| Blank sign+tape area before ridge | Tape creates T/+ blobs in Otsu that widen the ridge; blanking removes them before skeleton | Contour smoothing: Gaussian sigma=70 widened corners; sigma=10 created 4-pointed star artefacts |
| Distance-transform ridge + fixed dilation | Ridge = local maxima of distance transform = uniform centerline; dilating by `TRACK_HALF_WIDTH` gives constant width everywhere | Original Otsu binary: variable width; morphological erosion: killed thin ridges |
| Leave gap at sign positions | Tape area fully blanked; no bridging line needed | Bridge with straight line: required complex perimeter scan; any misdetection left permanent artefact |
| `_dynamic_sign_rois` short-circuit | Gap in mask at sign positions would falsely report OFF TRACK; short-circuit returns ON TRACK if car is inside sign bbox | Fixed static ROIs: sign positions differ between the two physical tables |
| Dual-polarity Otsu + adaptive OR | Track polarity and brightness varies across the table; neither method alone captures all sides | Single Otsu: misses glare-affected right side; adaptive alone: noisier on uniform sections |
| Background daemon thread | Mask detection ~300 ms on CPU — synchronous call freezes video stream | Synchronous: ~3 FPS, unacceptable |
| ArUco corners as calibration reference | Corner markers are mapped to exact pixel positions (0,0)→(600,0)→(600,600)→(0,600) by the perspective transform; scale_x = real_width/600, scale_y = real_height/600 — no manual point-clicking needed | Click-to-calibrate: underdetermined for non-square tables (one measurement → two unknowns) |
| Disable distortion correction | Wide mode distortion negligible at overhead height; correction degraded image | Apply manual k1/k2: estimated values too aggressive at this mount height |

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
| Inference imgsz | 1280 (upscaled at inference time) |
| Confidence threshold | 0.40 |

**Detection confidence with current model (imgsz=1280):**

| Sign | Typical conf range | Noise floor |
|---|---|---|
| stop | 0.96–0.98 | < 0.06 |
| speed_55 | 0.72–0.99 | < 0.06 |
| light_off | 0.63–0.76 | < 0.06 |

### Runtime Performance

| Component | FPS / Latency |
|---|---|
| YOLO sign detection (background thread, every 8th frame trigger) | ~200 ms per run, non-blocking |
| Main loop FPS (YOLO fully offloaded) | **~25 FPS, no periodic stutter** |
| Track mask detection (background thread) | ~200–400 ms, non-blocking |
| Perspective change detection | per-frame, negligible cost |
