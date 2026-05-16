# Software Architecture — Real-time Vehicle Tracking System
# 软件架构文档 — 实时车辆追踪系统

> **Student / 学生:** Rayven85 &nbsp;|&nbsp; **Supervisor / 导师:** Akshat &nbsp;|&nbsp; **Period / 周期:** March – May 2026

---

## System Architecture / 系统架构

The system is a single-process Python application running on Mac (development) and targeting Nvidia Jetson Orin (deployment). All computation happens locally — no network required. The main loop in `core/aruco_detect.py` drives everything; a daemon thread handles the slow track-mask computation without blocking the video stream.

系统是一个单进程 Python 应用，开发环境为 Mac，目标部署平台为 Nvidia Jetson Orin，所有计算均在本地完成。主循环位于 `core/aruco_detect.py`，一个守护线程负责耗时的轨道掩膜计算，不阻塞视频流。

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

| Layer / 层级 | Technology / 技术 | File / 文件 |
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

## Project Structure / 项目结构

All scripts use `ROOT = Path(__file__).resolve().parent.parent` to anchor paths to the project root, so they work correctly regardless of the calling directory.

所有脚本均通过 `ROOT = Path(__file__).resolve().parent.parent` 将路径锚定到项目根目录，从任意目录调用均可正确运行。

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

### `core/` — Main System / 核心系统

The entry point of the entire project. Contains two scripts.

整个项目的入口，包含两个脚本。

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

### `training/` — Model Training & Data Labelling / 模型训练与数据标注

**`train.py`** (127 lines) — Trains all five YOLO variants on the GTSDB dataset in one run, then writes a comparison CSV to `runs/comparison_results.csv`.

对五种 YOLO 变体统一训练，输出对比 CSV。

```
Usage: python training/train.py --models yolov8n yolov11n --epochs 100
```

Base weights are loaded from `weights/`. Trained weights are saved to `runs/train/<model>/weights/best.pt`.

**`train_track.py`** (86 lines) — Fine-tunes `weights/yolo11n.pt` on the track-specific sign dataset (`track_dataset/`). Outputs to `runs/train/track_signs/`.

基于 `weights/yolo11n.pt` 对赛道专用标志数据集进行微调。

```
Usage: python training/train_track.py --epochs 50
```

Key parameters: 5 classes (stop / speed_55 / light_off / light_green / light_red), batch=8 (CPU-friendly), patience=15 (early stopping).

**`auto_label.py`** (307 lines) — Semi-automatic YOLO label generator for the 600×600 warped screenshots.

半自动 YOLO 标注工具，用于 600×600 俯视截图。

Pipeline: Load screenshot → HSV auto-detection in 3 ROIs → display bounding boxes → keyboard adjustments → save YOLO `.txt` label.

流程：读入截图 → 三个 ROI 内 HSV 自动检测 → 显示边界框 → 键盘调整 → 保存 YOLO `.txt` 标注。

| Key | Action |
|---|---|
| `Enter` | Save & advance to next image |
| `1` / `2` | Toggle STOP / speed_55 box |
| `3` | Cycle light state: off → green → red |
| `e` | Mouse edit mode (`cv2.selectROI`) |
| `d` | Discard image |

Reads from `track_dataset/images/<split>/`, writes labels to `track_dataset/labels/<split>/`.

**`debug_yolo.py`** (94 lines) — Diagnostic tool. Loads one training image, runs the model at conf=0.05, prints every detection with ROI membership, and saves crops to `runs/`. Used to diagnose the domain mismatch issue (cropped ROI input vs full-image input).

诊断工具，用于定位域不匹配问题（ROI 裁剪输入 vs 全图推理）。

```
Usage: python training/debug_yolo.py [--img path/to/image.jpg]
```

---

### `evaluation/` — Model Evaluation & Visualisation / 模型评估与可视化

**`evaluate.py`** (142 lines) — Runs `model.val()` on the test split for each trained model, outputs a CSV to `runs/evaluation/test_results.csv`. Optionally generates bar charts (`--plot`).

对每个训练好的模型执行测试集评估，输出 CSV，可选生成柱状图。

```
Usage: python evaluation/evaluate.py [--plot]
```

**`benchmark.py`** (214 lines) — Measures model efficiency (parameter count, GFLOPs, model size, CPU latency, FPS) separately from accuracy. Merges lightweight results with accuracy results from `collect_results.py` into `runs/comparison_results_full.csv`.

独立测量模型效率（参数量、GFLOPs、体积、CPU延迟、FPS），与精度结果合并输出。

```
Usage: python evaluation/benchmark.py [--models yolov8n yolov11n] [--runs 100]
```

**`collect_results.py`** (74 lines) — Reads the per-epoch `results.csv` from each model's training directory, extracts the best-epoch row, and writes a unified `runs/comparison_results.csv`. Run automatically at the end of `train.py`, or manually after training.

从各模型训练目录中提取最优 epoch 指标，合并为统一 CSV。

**`visualize.py`** (265 lines) — Generates side-by-side GT vs prediction images. Three output modes:

生成真实框与预测框的并排对比图，三种输出模式：

| Mode | Output | Use |
|---|---|---|
| `per_model` | One image per model | Quick single-model check |
| `comparison` | All models on one image (different colours) | Visual comparison |
| `grid` | 2×3 grid (GT + 5 models) | Report figures |

```
Usage: python evaluation/visualize.py --mode grid --n 10 --seed 42
```

**`predict.py`** (77 lines) — Runs inference on a single model or all models in parallel on a given source (image dir, video, webcam). Saves annotated results to `runs/predict/<model>/`.

对单个或所有模型执行推理，结果保存至 `runs/predict/<model>/`。

```
Usage: python evaluation/predict.py --compare --source datasets/test/images
```

---

### `camera/` — Camera Tools & Calibration / 相机工具与标定

**`calibrate_gopro.py`** (161 lines) — Two-phase checkerboard calibration tool.

两阶段棋盘格标定工具。

- Phase 1 (first run): generates `calib_images/calibration_checkerboard.png` for printing
- Phase 2 (after photos collected): calls `cv2.calibrateCamera()` on images in `calib_images/`, saves result to `calib_images/gopro_calib.npz`

Target reprojection error: < 1.0 px.

**`undistort.py`** (214 lines) — Interactive distortion correction tuner. Displays original vs corrected side by side with a green grid overlay. Three sliders: K1 (primary barrel), K2 (higher-order), focal scale.

交互式畸变校正调参工具，实时左右对比，绿色网格叠加。

Supports three modes: `--mode wide` (GoPro Wide), `--mode superview` (GoPro SuperView), `--mode rpi` (Raspberry Pi fisheye via `cv2.fisheye`). Manual estimate: k1 = −0.462, k2 = −0.054 — currently disabled in `aruco_detect.py` as overhead distortion is negligible.

目前 `aruco_detect.py` 中畸变校正已关闭，因为当前俯拍高度下 Wide 模式畸变可忽略。

**`gopro_distortion.py`** (162 lines) — Compares distortion across GoPro FOV modes. Overlays a green grid on sample images from `gopro_samples/{linear,wide,superview}/`, uses Hough line detection to score straightness, outputs `fov_comparison.jpg`.

对比 GoPro 三种 FOV 模式的畸变程度，输出网格叠加对比图。

**`gopro_latency.py`** (160 lines) — Measures GoPro USB Webcam streaming performance: actual FPS vs declared FPS, frame interval jitter, dropped frame detection. Renders a scrolling bar chart of frame intervals in real time.

测量 GoPro USB Webcam 的实际帧率、帧间隔抖动和丢帧情况，实时渲染帧间隔柱状图。

**`test.py`** (220 lines) — Multi-mode camera and model testing suite:

多模式摄像头与模型测试工具：

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

### `tracking/` — Object Tracking / 目标追踪

**`track.py`** (141 lines) — Real-time multi-object tracking using YOLO + ByteTrack (built into Ultralytics). Maintains trajectory history (last 30 frames per ID) and draws alpha-blended trail lines (older = dimmer). Supports video file input or webcam.

使用 YOLO + ByteTrack 实现实时多目标追踪，维护每个 ID 最近 30 帧的轨迹，绘制透明度渐变轨迹线（越旧越淡）。

```
Usage: python tracking/track.py [--source 0] [--conf 0.4] [--save]
```

Output video saved to `runs/track_output.mp4` when `--save` is passed.

---

### `weights/` — Pretrained YOLO Weights / 预训练权重

ImageNet-pretrained base weights. Used as starting points for training — never used for inference directly.

ImageNet 预训练基础权重，仅作为训练起点，不直接用于推理。

| File | Used by |
|---|---|
| `yolo11n.pt` | `training/train_track.py` (track sign fine-tune) |
| `yolov5nu.pt` | `training/train.py` (multi-model comparison) |
| `yolov8n.pt` | `training/train.py` |
| `yolov9t.pt` | `training/train.py` |
| `yolov10n.pt` | `training/train.py` |

Trained weights (after running `train.py` / `train_track.py`) are stored separately in `runs/train/<model>/weights/best.pt`.

训练完成的权重单独存放于 `runs/train/<model>/weights/best.pt`，不覆盖此处的基础权重。

---

### `calib_images/` — Camera Calibration Data / 相机标定数据

| File | Description |
|---|---|
| `GOPR*.JPG` | 39 GoPro Wide mode photos of checkerboard from various angles |
| `calibration_checkerboard.png` | 10×7 checkerboard pattern (generated by `camera/calibrate_gopro.py`) |
| `gopro_calib.npz` | Calibration result: camera matrix K + distortion coefficients |

The `.npz` output from calibration is available for loading by `aruco_detect.py` (currently unused — distortion correction disabled).

---

### `datasets/` — GTSDB Evaluation Data / GTSDB评估数据

German Traffic Sign Detection Benchmark subset, YOLO format, 4 classes: `danger / mandatory / other / prohibitory`.

德国交通标志检测基准数据集子集，YOLO 格式，4 类。

Used by `evaluation/evaluate.py`, `evaluation/benchmark.py`, `evaluation/visualize.py`, and `evaluation/predict.py` for test-set evaluation.

---

### `track_dataset/` — Custom Track Sign Dataset / 赛道标志专用数据集

600×600 px warped-view screenshots of the physical race track, labelled with YOLO format annotations.

600×600 px 赛道俯视截图，YOLO 格式标注。

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

5 个类别：`stop / speed_55 / light_off / light_green / light_red`。

---

### `gopro_samples/` — FOV Comparison Photos / FOV 对比照片

```
gopro_samples/
├── linear/      GoPro Linear mode photos (minimal distortion)
├── wide/        GoPro Wide mode photos (current operating mode)
└── superview/   GoPro SuperView mode photos (severe barrel distortion)
```

Used by `camera/gopro_distortion.py` and `camera/undistort.py`.

---

### `markers/` — Generated ArUco PNGs / 生成的ArUco标记图

Output from `core/generate_markers.py`. 5 files: `corner_0.png` through `corner_3.png` and `car.png`. Print at 4–6 cm for corner markers, 3–4 cm for car marker.

`core/generate_markers.py` 的输出。桌角标记建议打印 4–6 cm，车顶标记建议 3–4 cm。

---

### `screenshots/` — Runtime Detection Screenshots / 运行时截图

Saved by pressing `s` in `core/aruco_detect.py`. Named sequentially `aruco_001.jpg`, `aruco_002.jpg`, … These are the source images for `training/auto_label.py`.

在 `core/aruco_detect.py` 运行时按 `s` 保存，顺序命名。这些图片是 `training/auto_label.py` 的输入来源。

---

### `runs/` — Training & Evaluation Outputs / 训练与评估输出

All output from training, evaluation, and inference scripts is written here. Never committed manually — all generated at runtime.

所有训练、评估、推理脚本的输出均写入此目录，运行时自动生成。

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

### `docs/` — Project Documentation / 项目文档

| File / Folder | Content |
|---|---|
| `project_log.md` | Phase-by-phase technical decisions, issues encountered, resolutions |
| `ReadingNote.md` | Background reading notes (YOLO, ByteTrack, TensorRT, ROS/Gazebo) |
| `ReadingNoteImage/` | Images referenced in reading notes |
| `workingrecord/` | Notion export of weekly progress notes (Week 3 through Week 11) |

---

## Development Timeline / 开发历程

### Week 3–5: YOLO Model Baseline Comparison / YOLO 模型基准对比

**Dataset:** GTSDB (German Traffic Sign Detection Benchmark), YOLO format, 4 classes.

Five nano-scale YOLO variants trained for 100 epochs each on identical hardware:

五种轻量 YOLO 变体在相同硬件上各训练 100 epoch：

| Model | mAP50 | mAP50-95 | Precision | Recall | Params (M) | GFLOPs | Latency (ms) | FPS |
|---|---|---|---|---|---|---|---|---|
| YOLOv5nu | 0.854 | 0.420 | **0.931** | 0.755 | 2.51 | 7.2 | 65.9 | 15.2 |
| YOLOv8n | 0.873 | **0.443** | 0.860 | **0.838** | 3.01 | 8.2 | 61.1 | 16.4 |
| YOLOv9t | 0.862 | 0.431 | 0.940 | 0.768 | **2.01** | 7.9 | 88.1 | 11.4 |
| YOLOv10n | 0.820 | 0.402 | 0.823 | 0.786 | 2.71 | 8.4 | 85.8 | 11.7 |
| **YOLOv11n ✓** | **0.881** | 0.440 | 0.883 | 0.807 | 2.59 | **6.4** | **59.1** | **16.9** |

**Decision:** YOLOv11n — highest mAP50, lowest latency (59.1 ms), fewest GFLOPs.

**结论：** YOLOv11n 综合最优，mAP50 最高、延迟最低、GFLOPs 最少。

---

### Week 5–7: Camera, Calibration & Marker System / 相机、标定与标记系统

#### Camera Selection / 相机选型

Initial Logitech USB webcam (1280×720 @ 24 FPS) couldn't cover the full table even at maximum mount height — fixed focal length, insufficient FOV. Replaced with GoPro in USB Webcam mode.

GoPro USB Webcam mode supports three FOV modes:

| FOV Mode | Distortion | Available via USB? | Decision |
|---|---|---|---|
| Linear | Minimal | ✗ | Offline analysis only |
| **Wide** | Slight barrel | ✓ | **Selected** |
| SuperView | Severe barrel | ✓ | Rejected |

#### Distortion Correction / 畸变校正

`camera/undistort.py` — Interactive tool: side-by-side original vs corrected with green grid overlay, three sliders (K1, K2, focal scale). Manual estimate: **k1 = −0.462, k2 = −0.054**.

`camera/calibrate_gopro.py` — Formal checkerboard pipeline: generates print pattern, then computes precise K matrix and distortion coefficients from 39 calibration photos. Output: `calib_images/gopro_calib.npz`.

At the current overhead mount height, Wide mode barrel distortion is negligible — manually estimated correction worsened the image. **Distortion correction is currently disabled** in `aruco_detect.py`.

当前俯拍高度下 Wide 模式畸变极小，手动估算的校正参数反而使图像变差，因此 `aruco_detect.py` 中畸变校正目前**关闭**。

#### ArUco vs AprilTag / 标记系统选型

| | AprilTag | **ArUco (selected)** |
|---|---|---|
| Library | `pupil-apriltags` (extra install) | Built into OpenCV |
| Speed | Slower | Faster |
| Docs | Academic papers | Full OpenCV docs |
| Close-range | Higher robustness | Sufficient for controlled setup |

Dictionary: `DICT_4X4_50` — simplest pattern (16 data bits), easiest to detect at close overhead range.

#### Perspective Warp / 透视变换

The oblique camera angle produces a trapezoidal view. Once the four corner markers (IDs 0–3) are detected, `cv2.getPerspectiveTransform` computes homography matrix `M`. Each frame is warped into a **600×600 px orthographic top-down view**. `M` is recomputed every frame — minor camera movement is automatically compensated.

每帧重新计算单应性矩阵 `M`，摄像机轻微晃动自动补偿，输出为 **600×600 px 正射俯视图**。

#### ON / OFF TRACK Detection / 在轨判断

**Problem:** The ArUco marker on the car physically covers the track beneath it. Real-time pixel sampling at the car's position reads the marker pattern, not the track — always reports OFF TRACK.

**车顶 ArUco 标记遮住车下赛道，实时像素采样始终读到标记图案而非赛道，导致永远判断为"出轨"。**

**Solution:** Before placing the car, press `c` to trigger `auto_detect_track_mask()` on the clean (car-free) warped view. The result is saved to `track_mask.png`. All subsequent queries check this pre-saved mask, not the live image.

**方案：** 放车前触发一次自动掩膜检测，将干净的掩膜保存为 `track_mask.png`，后续判断均查询此预存掩膜。

**Debounce:** 8-frame rolling window, >60% majority required to update ON/OFF state. Prevents flickering when the car drives near the track edge.

**防抖：** 8 帧滚动窗口，超过 60% 同意才更新状态，消除边缘抖动。

---

### Week 7–9: Custom YOLO Sign Detection / 赛道标志检测

Three stages of development:

开发经历三个阶段：

**Stage 1 — HSV colour thresholds (abandoned):** Fixed ROI regions + red/green pixel ratio thresholds. Brittle to lighting changes; LED colour boundaries ambiguous; ROI coordinate calibration tedious. Abandoned.

**Stage 2 — Pre-trained YOLO model (failed):** Applied the GTSDB-trained YOLOv11n directly to track signs. Domain mismatch: training data = large real street signs; target = A4-printed miniatures (~3×3 cm, ~100×80 px in warped view). Zero detections even at conf=0.05.

**Stage 3 — Bootstrap custom training (successful):**

Bootstrap strategy: use imprecise HSV detections as an annotation seed → refine manually with `auto_label.py` → train YOLO from scratch on track-domain images.

**自举策略：** 用 HSV 检测作为标注种子 → `auto_label.py` 人工微调 → 在赛道域图像上从头训练 YOLO。

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

**修复：** 发送完整 600×600 俯视图给 YOLO，再按检测框中心是否落在 ROI 内过滤：

```python
for box, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
    bx = float((box[0] + box[2]) / 2)
    by = float((box[1] + box[3]) / 2)
    if roi[0] <= bx <= roi[2] and roi[1] <= by <= roi[3]:
        best[name] = (conf_f, box)
```

Immediate result: `stop=0.907, light_off=0.825, speed_55=0.475`.

**Frame-skip caching:** YOLO CPU inference ≈ 100–300 ms/call. Running every frame → ~5 FPS. Solution: run every 8th frame, cache the result for intermediate frames → **~25 FPS**.

**帧跳缓存：** 每 8 帧推理一次，中间帧复用缓存，帧率从 ~5 FPS 恢复至 **~25 FPS**。

---

### Week 9–11: Automatic Track Mask Detection / 自动轨道识别

Previously required manual `c` keypress. Supervisor required full automation with adaptive re-detection.

原需手动按键，导师要求全自动并在视角改变时自适应重检测。

**Pipeline:**

```
raw frame → perspective warp (600×600) → downsample (300×300)
→ Gaussian blur → Otsu dual-polarity threshold → morphological clean
→ upsample (600×600) → _bridge_sign_roi() → save track_mask.png
```

**Dual-polarity Otsu:** Standard Otsu only tries one binary direction. Track polarity (dark-on-light vs light-on-dark) can vary across setups. Both directions are tried; the one with a white-pixel ratio closest to 30% (typical oval track) is selected. No manual threshold needed.

**双极性 Otsu：** 两种二值化方向都尝试，选择白色像素比例最接近 30%（典型椭圆轨道）的方向，无需手动设阈值。

**Half-resolution processing:** Detection runs at 300×300 (75% fewer pixels), upscaled to 600×600. Reduces CPU time enough to run in a background daemon thread without blocking the main loop.

**半分辨率处理：** 在 300×300 下检测（像素减少 75%），上采样回 600×600，计算量足够低以在后台线程中运行。

**`_bridge_sign_roi()` — sticker occlusion fix:**

Black adhesive stickers under the three road signs match the track colour. Otsu classifies them as track; morphological close expands them into thick rectangular blobs. Four approaches were tried:

三处路标底部黑色贴纸与轨道同色，Otsu 误判为轨道像素。尝试了四种方案：

| Version | Strategy | Outcome |
|---|---|---|
| v1 | Paint ROI as table colour before Otsu → fill → draw line | Right-side track deleted; Y/T artefacts |
| v2 | Paint only car ArUco, not sign ROIs | Sticker blobs remain |
| v3 | Scan 18 px ROI perimeter → find entry points → bridge with straight line | Stable — clean thin lines |
| v4 | Skeletonisation + branch pruning | Complete failure — edge truncation erodes inward |

Current solution (v3): clear ROI interior, find the two most-distant track entry points on the ROI perimeter, connect with one straight line (avoids Y/T junctions).

当前方案（v3）：清除 ROI 内所有像素，在 ROI 外缘找到最远两个入射点，连一条直线（避免 Y/T 形伪影）。

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

**视角变化重检测：** 对比上次检测时的透视矩阵 `_mask_M` 与当前矩阵，四角点位移超过 **15 px** 即触发重检测。

**ROI short-circuit in `is_on_track()`:** When the car is within a sign ROI zone (where stickers can cause mask gaps), the function returns `True` immediately without querying the mask — these zones are always on-track by construction.

**ROI 短路：** 当小车位于标志 ROI 区域内时（贴纸可能导致掩膜断裂），`is_on_track()` 直接返回 `True`，不查询掩膜。

---

## Key Technical Decisions / 关键技术决策

| Decision / 决策 | Rationale / 原因 | Alternative Rejected / 被否定方案 |
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

## Results Summary / 实验结果汇总

### YOLO Model Comparison — GTSDB / YOLO 模型对比

| Model | mAP50 ↑ | mAP50-95 ↑ | Latency (ms) ↓ | FPS ↑ | Params (M) ↓ |
|---|---|---|---|---|---|
| YOLOv5nu | 0.854 | 0.420 | 65.9 | 15.2 | 2.51 |
| YOLOv8n | 0.873 | 0.443 | 61.1 | 16.4 | 3.01 |
| YOLOv9t | 0.862 | 0.431 | 88.1 | 11.4 | 2.01 |
| YOLOv10n | 0.820 | 0.402 | 85.8 | 11.7 | 2.71 |
| **YOLOv11n ✓** | **0.881** | **0.440** | **59.1** | **16.9** | 2.59 |

### Custom Track Sign Model / 赛道专用模型

| Metric | Value |
|---|---|
| mAP50 | **0.995** |
| mAP50-95 | 0.873 |
| Precision | ~0.97 |
| Recall | ~0.96 |
| Training set | 44 images |
| Validation set | 12 images |

### Runtime Performance / 实时性能

| Component | FPS / Latency |
|---|---|
| YOLO sign detection — per frame (no cache) | ~5 FPS |
| YOLO sign detection — every 8th frame (cached) | **~25 FPS** |
| Track mask detection (background thread) | ~200–400 ms, non-blocking |
| Perspective change detection | per-frame, negligible cost |
