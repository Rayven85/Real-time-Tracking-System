"""
ArUco Detection + Perspective Warp (GoPro Wide)
=================================================
功能：
  1. 自动找到 GoPro Webcam
  2. 每帧先做 Wide 鱼眼矫正（k1/k2/fx 已标定）
  3. 检测桌角 ArUco 标记(ID 0-3)，做透视矫正得到正射俯视图
  4. 在矫正后画面中追踪小车标记(ID 4)位置

用法:  python aruco_detect.py
操作:  'w'=切换透视矫正  'd'=切换畸变矫正对比  's'=保存截图  'q'=退出
"""

import cv2
import numpy as np
import time
import os
import csv
import signal
import threading
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 畸变矫正开关 ──────────────────────────────────────────────────
# GoPro Wide 在当前俯拍高度畸变极小，关闭矫正效果更好
UNDISTORT_ENABLED = False
UNDISTORT_K1 = -0.462
UNDISTORT_K2 = -0.054
UNDISTORT_FX_SCALE = 23   # %

SCREENSHOT_DIR = str(ROOT / "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ArUco 字典（与生成时一致）
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# Subpixel refinement gives more accurate corner positions, reducing per-frame jitter
ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# Larger adaptive threshold window helps in uneven / patchy lighting
ARUCO_PARAMS.adaptiveThreshWinSizeMax = 53
# More tolerant of perspective-distorted marker shapes (oblique camera angle)
ARUCO_PARAMS.polygonalApproxAccuracyRate = 0.08
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# ID 定义
CORNER_IDS = {0: '左上', 1: '右上', 2: '右下', 3: '左下'}
CAR_ID = 4

# ── Corner stabilisation ──────────────────────────────────────────────
# EMA weight for each new detection (lower = smoother but slower to follow real movement)
CORNER_SMOOTH_ALPHA   = 0.15
# Frames to coast on last known position when a corner marker drops out
# 30 frames ≈ 1 s at 30 fps — handles brief occlusion without M going None
CORNER_PERSIST_FRAMES = 30
_smooth_corners: dict = {}   # {id: {'cx': float, 'cy': float, 'pts': ndarray, 'lost': int}}

# Scale EMA: very slow (α=0.05) so the "Table: X x Y cm" text is rock-stable.
# The old code loaded a constant from JSON; now we compute per-frame but smooth heavily.
SCALE_SMOOTH_ALPHA = 0.05

# ── Rectified (top-down) view size ────────────────────────────────────
# The rectified view must show the table at its TRUE aspect ratio with SQUARE
# pixels (1 px = the same real distance on both axes).  We don't know the
# table's real proportions until the 4 ArUco corners are seen, so we start with
# a provisional square and re-lock the dimensions once the per-axis scale is
# known (see _lock_aspect_ratio / the main loop).
WARP_BASE = 600                          # long-side resolution of the view
WARP_W, WARP_H = WARP_BASE, WARP_BASE    # updated once the real ratio is known
_aspect_locked = False                   # True after dims match the table ratio

# The track-segmentation pipeline (auto_detect_track_mask) is tuned for a SQUARE
# canvas, so it always runs at MASK_DIM×MASK_DIM regardless of the view's aspect
# ratio or WARP_BASE; only the final track-width expansion is mapped back to the
# real view.  Kept FIXED (not tied to WARP_BASE) so raising WARP_BASE for finer
# measurement never disturbs the mask tuning.
MASK_DIM = 600

# 轨道掩膜均匀化：骨架线 + 固定宽度膨胀
TRACK_HALF_WIDTH = 14   # px — final uniform half-width in the 600×600 mask

# ON/OFF TRACK 防抖：连续 N 帧一致才切换
DEBOUNCE_FRAMES = 8
_track_history = []   # 最近N帧的判断结果
_track_status = False  # 当前稳定状态

# ── STOP compliance ──────────────────────────────────────────────────
STOP_SPEED_THRESHOLD = 5.0  # px/s — below this counts as stopped
STOP_DISPLAY_SEC = 2.0      # how long to show STOP event message
_stop_event = {'text': '', 'color': (0, 0, 0), 'until': 0.0}

# ── YOLO sign detection ──────────────────────────────────────────────
SIGN_MODEL_PATH  = str(ROOT / "runs/train/track_signs/weights/best.pt")
SIGN_CONF        = 0.40
SIGN_EVERY_N     = 8     # run YOLO once every N frames; reuse result in between
_sign_model      = None
_sign_cache      = {'light': 'OFF', 'stop': False, 'speed': False, 'boxes': []}
_sign_frame_cnt  = 0
_sign_running    = False   # True while background YOLO thread is running
_sign_result     = [None]  # [0] holds the latest result from the background thread
_mask_running    = False   # True while background thread is computing track mask
_mask_result     = [None]  # [0] holds the latest mask from the background thread
_mask_M          = None    # M matrix used when last mask was computed
MASK_REPRO_THR   = 15.0   # pixels — re-detect if any corner moves more than this
_dynamic_sign_rois = []    # (x1,y1,x2,y2) boxes in warped 600×600 space; set by mask thread

# ── Distance measurement ──────────────────────────────────────────────
# The only hard-coded physical fact: each ArUco marker is a 10.5 cm square.
# All other distances are inferred automatically from this single constant.
ARUCO_REAL_SIZE_CM = 10.5

# Optional fixed scale-correction factor.  A 1.0101 factor was tried (one
# 19-point session read 1 % short) but validation showed the ArUco scale itself
# drifts ~3 % between sessions / after disturbing the setup — a FIXED factor
# can't correct a DRIFTING scale, so it is disabled (=1.0).  The real accuracy
# limit is scale repeatability; see the scale-stability study ('j' key →
# evaluation/scale_repeatability.csv).
SCALE_CALIBRATION = 1.0

# ── Baseline calibration (--baseline-x / --baseline-y) ────────────────
# Preferred way to fix the scale. Deriving cm/px from a marker's 10.5 cm edge
# means reading a ~37 px feature and extrapolating it across a 600 px field, so
# any local error is multiplied ~16x; measured against tape the result came out
# 5.7 % short in x and 2.4 % in y — not even a single factor, because a marker's
# imaged size varies across the field (a flat marker read 10.22 cm at the centre
# and 10.51 cm at the corners).
#
# The homography always maps the corner-marker centres onto the corners of the
# warped rectangle, so one tape measurement of the centre-to-centre distance
# fixes cm/px exactly, over a 181 cm baseline instead of 10.5 cm — the same
# absolute error diluted ~17x. It is also constant, so the scale stops drifting.
_baseline_x = None    # tape cm, ID0 centre → ID1 centre (spans WARP_W)
_baseline_y = None    # tape cm, ID0 centre → ID3 centre (spans WARP_H)

_dist_mode    = False      # True while user is clicking measurement points
_dist_pt_a    = None       # first click in warped 600×600 space
_dist_pt_b    = None       # second click
_dist_scale_x = None       # cm per pixel along x-axis (auto-derived)
_dist_scale_y = None       # cm per pixel along y-axis (auto-derived)
_logging_busy = False      # True while a background 'l' prompt is awaiting input
_mount_height = None       # camera height above table (cm); set via --height, logged by 'j'
_marker_height = None      # car marker height above the table (cm); enables parallax correction

# Live camera height, so the parallax term follows the rig instead of trusting a
# number typed once. The corner markers are a fixed physical distance apart, so
# their separation in the ORIGINAL image varies as 1/H: anchor that product
# against the measured --height once and H can be read off it every frame.
# The long baseline is used rather than a marker's own 10.5 cm edge because the
# imaged size of a marker is not reliable enough — it reads 10.22 cm at the
# centre of the field and 10.51 cm at the corners, which would put h anywhere
# between 8 and 15 cm.
_h_anchor = None           # H0 * baseline_px at the anchoring frame
_h_live = None             # current camera height (cm), tracked from the baseline
H_TRACK_ALPHA = 0.05       # heavy smoothing: H is physically near-constant

# ── CAR detection-rate monitor ────────────────────────────────────────
# The corner markers get EMA smoothing + persistence (_stabilise_corners), so
# their flicker is hidden; the car marker is deliberately raw — smoothing it
# would add speed-dependent lag and corrupt the very dynamic error we measure.
# This rolling hit-rate exposes the underlying detection reliability so lighting
# / marker changes can be judged live, and a bad run spotted before analysis.
CAR_RATE_WINDOW = 100
_car_hits: list = []

# ── Trajectory logging (dynamic tracking accuracy, 't' key) ───────────
_traj_recording = False    # True while logging the CAR trajectory
_traj_rows = []            # [(t_s, wx, wy, x_cm, y_cm), ...] for the current run
_traj_t0 = None            # perf_counter() at recording start

# Measurement log for accuracy (Task 2) + repeatability (Task 4) studies.
# Each 'l' keypress appends the current A–B measurement plus a label and the
# tape-measured ground truth; evaluation/accuracy_eval.py crunches the file.
MEASURE_LOG = str(ROOT / "evaluation" / "measurements.csv")

# Scale-stability log ('j' keypress): one row per read of the ArUco-derived
# scale, for the scale-repeatability study (read → disturb setup → read → …).
SCALE_LOG = str(ROOT / "evaluation" / "scale_repeatability.csv")

# Marker-survey log ('n' keypress): every detected marker's position and apparent
# size relative to the camera nadir, for the parallax / radial-distortion study.
# Both effects grow with distance from the nadir, so radius is the key variable.
SURVEY_LOG = str(ROOT / "evaluation" / "marker_survey.csv")


def _trigger_sign_detection(warped_snap):
    """Fire YOLO in a daemon thread; skip if a run is already in progress."""
    global _sign_running
    if _sign_running:
        return
    _sign_running = True
    def _run():
        # finally: an exception here would otherwise leave _sign_running True
        # forever, silently stopping all further sign detection.
        global _sign_running
        try:
            _sign_result[0] = detect_signs(warped_snap)
        except Exception as e:
            print(f"[YOLO] 检测线程出错: {e}")
        finally:
            _sign_running = False
    threading.Thread(target=_run, daemon=True).start()


def _get_sign_model():
    global _sign_model
    if _sign_model is None:
        from ultralytics import YOLO
        _sign_model = YOLO(SIGN_MODEL_PATH)
        print(f"[YOLO] Sign model loaded: {SIGN_MODEL_PATH}")
    return _sign_model


def list_cameras(max_index=5):
    """列出所有可用摄像头"""
    print("可用摄像头：")
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            release_camera(cap)   # timed — a plain release can hang here too
            time.sleep(0.3)   # let macOS release the device before next open
            print(f"  [{i}] {w}x{h}")
    print()


def open_camera(source, retries=5):
    """打开指定编号的摄像头，失败时自动重试（GoPro USB Webcam 在 Mac 上需要）"""
    for attempt in range(1, retries + 1):
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print(f"  [attempt {attempt}/{retries}] 驱动未就绪，3s 后重试…")
            time.sleep(3.0)
            continue

        # 等硬件开始推流（GoPro 比普通摄像头慢）
        time.sleep(2.0)

        ret, frame = cap.read()
        if ret and frame is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"使用摄像头 [{source}]: {w}x{h}")
            return cap

        print(f"  [attempt {attempt}/{retries}] 推流未就绪，3s 后重试…")
        release_camera(cap)
        time.sleep(3.0)

    print(f"无法打开摄像头 {source}（已重试 {retries} 次）")
    return None


class FrameGrabber:
    """
    Reads the camera on a daemon thread and publishes the newest frame.

    cv2's read() does not merely return False when a GoPro stops delivering
    frames on macOS — it can block forever, which freezes the whole UI: no
    display, no keys, only Ctrl-C. Confining that call to its own thread keeps
    the main loop responsive, so a stall becomes a visible warning the operator
    can quit out of cleanly rather than a hang that leaves the device claimed.

    Each frame carries a timestamp: the main loop processes a frame only when it
    is new (no duplicate trajectory samples) and can see how stale it has gone.
    """

    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = False
        self.fail_streak = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            try:
                ret, f = self._cap.read()
            except Exception:
                ret, f = False, None
            if ret and f is not None:
                with self._lock:
                    self._frame = f
                    self._stamp = time.perf_counter()
                    self.fail_streak = 0
            else:
                self.fail_streak += 1
                time.sleep(0.01)      # failed read: don't spin the CPU

    def read(self):
        """Return (frame, stamp); stamp is 0.0 until the first frame arrives."""
        with self._lock:
            return self._frame, self._stamp

    def stop(self):
        self._stop = True


def release_camera(cap, timeout=1.0):
    """
    Release the capture without risking a hang on exit.

    cap.release() can block indefinitely with a GoPro on macOS, which is why the
    original code skipped it and called os._exit — but then the device stays
    claimed and the next run cannot open it. Releasing on a thread gets the
    device freed in the normal case while still guaranteeing we exit.
    """
    if cap is None:
        return
    t = threading.Thread(target=cap.release, daemon=True)
    t.start()
    t.join(timeout)


def build_undistort_maps(h, w):
    """预计算畸变矫正映射表（只算一次，之后每帧直接 remap）"""
    fx = w * (0.5 + UNDISTORT_FX_SCALE / 100.0)
    K = np.array([[fx, 0, w / 2],
                  [0, fx, h / 2],
                  [0,  0,     1]], dtype=np.float64)
    dist = np.array([UNDISTORT_K1, UNDISTORT_K2, 0, 0, 0], dtype=np.float64)
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.5)
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (w, h), cv2.CV_16SC2)
    return map1, map2, roi


def undistort_frame(frame, map1, map2, roi):
    """用预计算的映射表矫正一帧"""
    out = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        out = out[y:y+rh, x:x+rw]
    return out


def detect_markers(frame):
    """检测画面中所有 ArUco 标记，返回 {id: center_point}"""
    corners, ids, _ = DETECTOR.detectMarkers(frame)
    result = {}
    if ids is None:
        return result, corners, ids
    for i, marker_id in enumerate(ids.flatten()):
        pts = corners[i][0]  # 4个角点
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        result[marker_id] = (cx, cy, pts)
    return result, corners, ids


def _stabilise_corners(raw_markers):
    """
    Apply EMA smoothing and short-term persistence to the 4 corner markers.

    Two sources of flicker are addressed:
      • Jitter: detected position jumps ±a few pixels each frame → EMA damps this.
      • Drop-out: marker not found for 1-CORNER_PERSIST_FRAMES frames → coast on
        last known smoothed position so M doesn't go None and cause a view switch.

    Non-corner markers (e.g. car ID 4) are passed through unchanged.
    """
    global _smooth_corners
    result = dict(raw_markers)
    for mid in CORNER_IDS:
        if mid in raw_markers:
            cx, cy, pts = raw_markers[mid]
            if mid in _smooth_corners:
                s = _smooth_corners[mid]
                s['cx']  = CORNER_SMOOTH_ALPHA * cx  + (1 - CORNER_SMOOTH_ALPHA) * s['cx']
                s['cy']  = CORNER_SMOOTH_ALPHA * cy  + (1 - CORNER_SMOOTH_ALPHA) * s['cy']
                s['pts'] = CORNER_SMOOTH_ALPHA * pts + (1 - CORNER_SMOOTH_ALPHA) * s['pts']
                s['lost'] = 0
            else:
                _smooth_corners[mid] = {
                    'cx': float(cx), 'cy': float(cy),
                    'pts': pts.astype(np.float32), 'lost': 0,
                }
            s = _smooth_corners[mid]
            result[mid] = (
                int(round(s['cx'])),
                int(round(s['cy'])),
                s['pts'].astype(np.float32),
            )
        elif mid in _smooth_corners:
            s = _smooth_corners[mid]
            s['lost'] += 1
            if s['lost'] <= CORNER_PERSIST_FRAMES:
                result[mid] = (
                    int(round(s['cx'])),
                    int(round(s['cy'])),
                    s['pts'].astype(np.float32),
                )
            else:
                del _smooth_corners[mid]
    return result


def get_perspective_transform(markers):
    """
    用4个桌角标记计算透视变换矩阵。
    ID 0=左上, 1=右上, 2=右下, 3=左下
    """
    if not all(i in markers for i in [0, 1, 2, 3]):
        return None

    # 每个标记的中心点作为对应桌角
    src = np.float32([
        markers[0][:2],   # 左上
        markers[1][:2],   # 右上
        markers[2][:2],   # 右下
        markers[3][:2],   # 左下
    ])

    dst = np.float32([
        [0, 0],
        [WARP_W, 0],
        [WARP_W, WARP_H],
        [0, WARP_H],
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    return M


def warp_point(pt, M):
    """将原图中的点变换到矫正图中的坐标"""
    p = np.float32([[pt]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(p, M)
    return int(warped[0][0][0]), int(warped[0][0][1])


def compute_scale_from_aruco(markers, M):
    """
    Auto-derive cm-per-pixel scale using only the known ArUco marker size.

    Each corner marker (IDs 0-3) is ARUCO_REAL_SIZE_CM on every side.
    Project its 4 detected corners through M into the warped space and measure
    the marker's EDGE lengths (rotation-invariant) in x and y — these give cm/px
    independently for both axes.  (Edge lengths, not the bounding box: a rotated
    marker inflates the bbox and biases the scale short.)  No manual input;
    works for any table size or camera height.
    """
    x_scales, y_scales = [], []
    for mid in CORNER_IDS:
        if mid not in markers:
            continue
        _, _, pts = markers[mid]          # pts: (4, 2) corners in original image
        wp = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        # Use the marker's actual EDGE lengths, not the axis-aligned bounding box.
        # A slightly rotated marker inflates the bbox extent (by cosθ+sinθ) →
        # scale comes out too small → every distance reads short, uniformly.
        # Corner order TL,TR,BR,BL: edges 0-1 & 2-3 span the x side, 1-2 & 3-0 the y side.
        side_x = 0.5 * (np.linalg.norm(wp[1] - wp[0]) + np.linalg.norm(wp[2] - wp[3]))
        side_y = 0.5 * (np.linalg.norm(wp[2] - wp[1]) + np.linalg.norm(wp[3] - wp[0]))
        if side_x > 1:
            x_scales.append(ARUCO_REAL_SIZE_CM / side_x)
        if side_y > 1:
            y_scales.append(ARUCO_REAL_SIZE_CM / side_y)
    if not x_scales or not y_scales:
        return None, None
    # Apply the empirical scale calibration so measured distances are unbiased.
    return (float(np.mean(x_scales)) * SCALE_CALIBRATION,
            float(np.mean(y_scales)) * SCALE_CALIBRATION)


def _lock_aspect_ratio(sx, sy):
    """
    Fix WARP_W / WARP_H so the rectified view has SQUARE pixels at the table's
    TRUE aspect ratio.

    Until now the view was a provisional square, which anisotropically stretched
    a non-square table (sx ≠ sy → a circle on the table renders as an ellipse).
    Given the per-axis cm/px from the ArUco markers, the real aspect ratio is
    sx : sy.  Keep the longer physical axis at WARP_BASE px and scale the shorter
    one down, so 1 px maps to the same real distance on both axes and the
    on-screen view is geometrically faithful.  Called once, then frozen.
    """
    global WARP_W, WARP_H, _aspect_locked
    # Tape baselines describe the same two spans (they ARE the rectangle edges),
    # so when available they set the ratio directly and the marker estimate — the
    # weaker of the two — is not consulted at all.
    if _baseline_x and _baseline_y:
        sx, sy = _baseline_x, _baseline_y
    if not sx or not sy or sx <= 0 or sy <= 0:
        return False
    if sx >= sy:                       # x is the longer / coarser axis
        WARP_W = WARP_BASE
        WARP_H = max(1, int(round(WARP_BASE * sy / sx)))
    else:                              # y is the longer / coarser axis
        WARP_H = WARP_BASE
        WARP_W = max(1, int(round(WARP_BASE * sx / sy)))
    _aspect_locked = True
    ratio = max(WARP_W, WARP_H) / max(1, min(WARP_W, WARP_H))
    print(f"[Aspect] Rectified view locked to {WARP_W}x{WARP_H} px "
          f"(true table ratio {ratio:.2f}:1, square pixels)")
    return True


def camera_gsd_from_markers(markers):
    """
    Ground sampling distance in cm per CAMERA pixel at the table plane, measured
    from the ArUco markers in the ORIGINAL (un-warped) frame.

    Each corner marker is ARUCO_REAL_SIZE_CM on a side; its mean side length in
    camera pixels gives cm/px at the table.  This is the sensor's true physical
    resolution — the analytical-precision anchor (GSD = H / f).  It is DISTINCT
    from the warp scale (_dist_scale_x/y), which is only as fine as the
    WARP_BASE-pixel rectified view and is usually coarser.
    """
    gsds = []
    for mid in CORNER_IDS:
        if mid not in markers:
            continue
        pts = markers[mid][2].astype(np.float32)   # 4 corners in the original frame
        sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
        side_px = float(np.mean(sides))
        if side_px > 1:
            gsds.append(ARUCO_REAL_SIZE_CM / side_px)
    if not gsds:
        return None
    return float(np.mean(gsds))


def draw_raw(frame, markers, ids, corners):
    """在原始画面上绘制检测结果"""
    out = frame.copy()

    # 画所有标记轮廓
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

    # 标注每个标记的身份
    for marker_id, (cx, cy, _) in markers.items():
        if marker_id in CORNER_IDS:
            label = f"ID{marker_id} {CORNER_IDS[marker_id]}"
            color = (0, 255, 0)
        elif marker_id == CAR_ID:
            label = "CAR"
            color = (0, 165, 255)
        else:
            label = f"ID{marker_id}"
            color = (200, 200, 200)
        cv2.putText(out, label, (cx - 30, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(out, (cx, cy), 5, color, -1)

    # 4角连线（如果都检测到）
    if all(i in markers for i in [0, 1, 2, 3]):
        pts = np.int32([markers[i][:2] for i in [0, 1, 2, 3]])
        cv2.polylines(out, [pts], True, (0, 255, 255), 2)
        cv2.putText(out, "4 corners OK - warp ready", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    return out


def _perspective_changed(M_new):
    """Return True if M_new differs from the stored _mask_M by more than MASK_REPRO_THR px."""
    global _mask_M
    if _mask_M is None or M_new is None:
        return False
    # Warp 4 image corners with both matrices and compare where they land
    corners = np.array([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]],
                       dtype=np.float32).reshape(-1, 1, 2)
    p_old = cv2.perspectiveTransform(corners, _mask_M)
    p_new = cv2.perspectiveTransform(corners, M_new)
    max_shift = float(np.max(np.linalg.norm(p_new - p_old, axis=2)))
    return max_shift > MASK_REPRO_THR



def _extract_track_ring(mask):
    """
    Find the largest ring-shaped region in a binary mask.

    Why gaps break naive ring detection:
      If the ring has any gap (missing segment, sign cutout, faint area), the
      track centre is topologically connected to the exterior background, so
      RETR_CCOMP never finds a parent-child pair → score = 0.

    Fix: apply a large MORPH_CLOSE (radius ~90 px on 600×600) to seal every
    gap before the topology check.  The sealed ring is used only to LOCATE the
    ring; the output is drawn from those sealed contours so that gaps are
    bridged.  The skeleton step that follows re-centres and normalises width,
    so artificially filled gaps don't survive into the final mask.

    Criteria for a valid ring (on the sealed mask):
      - outer contour encloses ≥ 8 % of image
      - band area (outer − hole)  in [4 %, 55 %]  ← the track strip
      - interior hole             ≥ 8 % of image  ← the track centre
    Score = hole_area / image_size.
    """
    # After perspective warp the track runs very close to the image borders.
    # findContours treats border-touching white regions as open (no closed outer
    # contour), so RETR_CCOMP never finds the parent-child hierarchy that
    # signals a ring.  Adding a black border forces the track's outer edge to
    # be fully enclosed within the image → proper ring topology.
    PAD = 10
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, PAD, PAD, PAD, PAD,
                                cv2.BORDER_CONSTANT, value=0)

    k_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    sealed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, k_seal)

    cnts, hier = cv2.findContours(sealed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return np.zeros_like(mask), 0.0
    hier = hier[0]
    total = float(mask.size)
    best_score, best_outer, best_children = 0.0, None, []

    for i in range(len(cnts)):
        if hier[i][3] != -1:          # skip inner contours
            continue
        outer_area = cv2.contourArea(cnts[i])
        if outer_area < total * 0.08:
            continue

        children, hole_area = [], 0.0
        ci = hier[i][2]
        while ci != -1:
            ha = cv2.contourArea(cnts[ci])
            hole_area += ha
            children.append(cnts[ci])
            ci = hier[ci][0]          # next sibling

        band_area = outer_area - hole_area
        if not (total * 0.04 <= band_area <= total * 0.55 and hole_area >= total * 0.08):
            continue

        score = hole_area / total
        if score > best_score:
            best_score = score
            best_outer, best_children = cnts[i], children

    if best_outer is None:
        return np.zeros_like(mask), 0.0

    # Draw ring on padded canvas, then crop back to original size.
    # Contour coords are in padded space, so the crop recovers the 600×600 ring.
    ring_padded = np.zeros_like(padded)
    cv2.drawContours(ring_padded, [best_outer], -1, 255, -1)
    for c in best_children:
        cv2.drawContours(ring_padded, [c], -1, 0, -1)
    ring = ring_padded[PAD:PAD + h, PAD:PAD + w]
    return ring, best_score


def _smooth_ring_contours(mask, sigma=12):
    """
    Remove small bumps (sign stickers, tape) from a ring mask by Gaussian-
    smoothing the outer and inner contour coordinates in-place (circular /
    wrap-around boundary so the closed loop is treated correctly).

    Why Gaussian instead of convex hull:
      Convex hull is GLOBAL — a single downward bump (e.g. STOP sticker) pulls
      the entire hull boundary downward, shifting the reconstructed ring ≥50 px.
      Gaussian smoothing is LOCAL: sigma=12 removes bumps ≤ ~30 px arc length
      while leaving the rounded corners (arc length ~100-150 px) intact.
    """
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or len(cnts) == 0:
        return mask
    hier = hier[0]

    # Largest outer contour = ring outer boundary
    outer_i, outer_a = -1, 0
    for i in range(len(cnts)):
        if hier[i][3] == -1:
            a = cv2.contourArea(cnts[i])
            if a > outer_a:
                outer_a, outer_i = a, i
    if outer_i < 0:
        return mask

    # Largest child = inner hole boundary
    inner_i, inner_a = -1, 0
    ci = hier[outer_i][2]
    while ci != -1:
        a = cv2.contourArea(cnts[ci])
        if a > inner_a:
            inner_a, inner_i = a, ci
        ci = hier[ci][0]

    def _gauss(cnt):
        pts = cnt.reshape(-1, 2).astype(np.float64)
        n   = len(pts)
        if n < 6:
            return cnt
        half_k = int(3 * sigma)
        kx     = np.arange(-half_k, half_k + 1, dtype=np.float64)
        kernel = np.exp(-kx ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        # Tile × 3 for circular boundary, convolve, keep middle copy
        tx = np.tile(pts[:, 0], 3)
        ty = np.tile(pts[:, 1], 3)
        sx = np.convolve(tx, kernel, mode='same')[n:2 * n]
        sy = np.convolve(ty, kernel, mode='same')[n:2 * n]
        return np.column_stack([sx, sy]).reshape(-1, 1, 2).astype(np.int32)

    result = np.zeros_like(mask)
    cv2.drawContours(result, [_gauss(cnts[outer_i])], -1, 255, -1)
    if inner_i >= 0:
        cv2.drawContours(result, [_gauss(cnts[inner_i])], -1, 0, -1)
    else:
        # No inner hole detected — derive it by erosion
        k_tw  = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (TRACK_HALF_WIDTH * 2 + 1, TRACK_HALF_WIDTH * 2 + 1))
        inner = cv2.erode(result, k_tw)
        result = cv2.bitwise_and(result, cv2.bitwise_not(inner))

    return result if result.any() else mask


def auto_detect_track_mask(warped, car_pos=None, sign_boxes=None):
    """
    Automatically segment the track from the warped top-down view.

    The whole segmentation pipeline is tuned for a SQUARE canvas, so the (now
    possibly non-square, true-aspect) input is squared to MASK_DIM×MASK_DIM here,
    processed exactly as before, and only the final uniform-width skeleton is
    resized back to the real view — where the dilation runs, so the track band
    is uniform in real units (square pixels).

    Approach:
      1. Mask car ArUco only before Otsu (preserves track through sign areas).
      2. Dual-polarity Otsu + morphological clean-up.
      3. Erase YOLO sign+tape regions, then distance-transform ridge skeleton.
    """
    # Square the inputs (true-aspect → square). car_pos and sign_boxes arrive in
    # true-aspect view coords, so scale them into the square canvas by the same
    # factors (sxr, syr); the mask is mapped back to the view at the end.
    src_w = max(1, warped.shape[1])
    src_h = max(1, warped.shape[0])
    sxr = MASK_DIM / src_w
    syr = MASK_DIM / src_h
    img = cv2.resize(warped, (MASK_DIM, MASK_DIM))

    if car_pos is not None:
        cv2.circle(img, (int(car_pos[0] * sxr), int(car_pos[1] * syr)),
                   40, (180, 180, 180), -1)

    # ── Downsample to 300×300 for speed ──────────────────────────────
    small = cv2.resize(img, (MASK_DIM // 2, MASK_DIM // 2), interpolation=cv2.INTER_AREA)

    # Blank the 4 ArUco corner markers in the downsampled warped image.
    # They are dark stickers → white in BINARY_INV → appear as track bumps.
    # Detect them here (they are still visible in the warped view) and paint
    # them medium-gray so they fall below the Otsu/adaptive dark threshold.
    _mc, _, _ = detect_markers(small)
    for _mid, (_cx, _cy, _) in _mc.items():
        if _mid in CORNER_IDS:
            cv2.circle(small, (_cx, _cy), 22, (180, 180, 180), -1)

    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Method A: CLAHE + global Otsu BINARY_INV.
    # CLAHE (4×4 tiles = 75×75 px at 300×300) reduces large-scale brightness
    # variation before the global threshold.  Works well for left/top/bottom.
    gray_clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    blurred_a   = cv2.GaussianBlur(gray_clahe, (7, 7), 0)
    _, seg_otsu = cv2.threshold(blurred_a, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Method B: adaptive threshold on raw gray (no CLAHE — let the local
    # window do its own normalisation).  blockSize=51 gives 51×51 px windows;
    # C=8 requires a pixel to be 8 grey levels darker than the local
    # Gaussian-weighted mean.  This catches the right-side track that appears
    # globally bright (glare) but is still locally darker than the white table.
    blurred_b  = cv2.GaussianBlur(gray, (7, 7), 0)
    seg_adapt  = cv2.adaptiveThreshold(blurred_b, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 8)

    # OR-combine: a pixel is track if flagged by either method.
    seg_raw = cv2.bitwise_or(seg_otsu, seg_adapt)

    # Morphological clean-up.
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    seg_raw = cv2.morphologyEx(seg_raw, cv2.MORPH_CLOSE, k_close)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seg_raw = cv2.morphologyEx(seg_raw, cv2.MORPH_OPEN,  k_open)

    # Clear image border (15 px at 300×300 = 30 px at 600×600).
    # The perspective warp fills out-of-bounds pixels with black; those
    # fill-pixels appear white in BINARY_INV and merge with the track edge,
    # causing the left/right "overflow" seen in the mask.
    _h, _w = seg_raw.shape
    # Border/corner clears scale with the (now possibly non-square) frame so the
    # short axis of a thin table isn't over-erased. 0.05·min-dim reproduces the
    # original 15 px at 300×300 and shrinks proportionally for a thin table.
    _brd = max(4, int(round(0.05 * min(_h, _w))))
    seg_raw[:_brd, :]  = 0
    seg_raw[-_brd:, :] = 0
    seg_raw[:, :_brd]  = 0
    seg_raw[:, -_brd:] = 0

    # Blank the four image corners — environment outside the table that leaks
    # into the warp corners after the border is cleared. 0.15·min-dim = 45 px at
    # 300×300, scaling down for a thin table so it doesn't eat the short axis.
    _cr = max(12, int(round(0.15 * min(_h, _w))))
    cv2.circle(seg_raw, (0,   0),   _cr, 0, -1)
    cv2.circle(seg_raw, (_w,  0),   _cr, 0, -1)
    cv2.circle(seg_raw, (0,   _h),  _cr, 0, -1)
    cv2.circle(seg_raw, (_w,  _h),  _cr, 0, -1)

    # Remove blobs disconnected from the main ring.
    # • The ring (all 4 sides connected) = ~13 000 px at 300×300.
    # • "COMSYS 306" text blob (adaptive detected) = ~2 000–3 000 px.
    # Threshold at ~3.3% of the frame (≈3 000 px at 300×300) safely drops text
    # without cutting ring sections, and adapts to a non-square frame.
    min_blob = int(0.033 * seg_raw.size)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(seg_raw, connectivity=8)
    seg_filt = np.zeros_like(seg_raw)
    for lbl in range(1, n_lbl):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_blob:
            seg_filt[lbl_map == lbl] = 255
    seg_raw = seg_filt

    # Debug: save segmentation before skeleton
    cv2.imwrite(str(ROOT / "debug_otsu.png"),
                cv2.resize(seg_raw, (MASK_DIM, MASK_DIM), interpolation=cv2.INTER_NEAREST))

    # ── Upsample (still square) ───────────────────────────────────────
    clean = cv2.resize(seg_raw, (MASK_DIM, MASK_DIM), interpolation=cv2.INTER_NEAREST)

    # ── Identify sign sticker boxes early (needed for tape removal) ──
    if sign_boxes is None:
        sign_boxes = []
    STICKER_CLASSES = {'stop', 'speed_55'}
    sticker_boxes = [b for b in sign_boxes
                     if len(b) < 7 or b[6] in STICKER_CLASSES]

    # ── Erase sign + tape regions from binary mask ───────────────────
    # The black tape holding each sign creates T/+ shaped blobs in the
    # Otsu mask.  Blank a padded rectangle so tape disappears before the
    # ridge is computed.  The gap is left empty — is_on_track() treats
    # anything inside _dynamic_sign_rois as on-track, so the car is never
    # flagged as off-track while crossing a sign area.
    TAPE_PAD = 20
    padded_rois = []
    for b in sticker_boxes:
        # sign boxes are in true-aspect view coords → scale into the square canvas
        x1, y1 = int(b[0] * sxr), int(b[1] * syr)
        x2, y2 = int(b[2] * sxr), int(b[3] * syr)
        px1 = max(0, x1 - TAPE_PAD);  py1 = max(0, y1 - TAPE_PAD)
        px2 = min(MASK_DIM, x2 + TAPE_PAD); py2 = min(MASK_DIM, y2 + TAPE_PAD)
        cv2.rectangle(clean, (px1, py1), (px2, py2), 0, -1)
        padded_rois.append((px1, py1, px2, py2))

    cv2.imwrite(str(ROOT / "debug_otsu.png"), clean)

    # ── Uniform-width skeleton: distance-transform ridge + fixed dilation ──
    dist         = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    dilated_dist = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    ridge        = ((dist >= dilated_dist * 0.85) & (clean > 0)).astype(np.uint8) * 255

    # Close tiny gaps (< 9 px) in the skeleton before expanding.
    ridge = cv2.morphologyEx(ridge, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    cv2.imwrite(str(ROOT / "debug_ridge.png"), ridge)

    # Map the thin skeleton back to the true-aspect view BEFORE expanding, so the
    # band comes out uniform in real units (square pixels), not uniform-in-square.
    ridge = cv2.resize(ridge, (WARP_W, WARP_H), interpolation=cv2.INTER_NEAREST)

    # Expand skeleton to uniform TRACK_HALF_WIDTH.
    dk    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (TRACK_HALF_WIDTH * 2 + 1, TRACK_HALF_WIDTH * 2 + 1)
    )
    clean = cv2.dilate(ridge, dk)

    # Seal remaining micro-gaps after dilation.
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    # ── Register sign ROIs for on-track short-circuit ────────────────
    # Use the original YOLO bounding box (no padding) — only the sign
    # itself counts as on-track, not the surrounding tape area.
    global _dynamic_sign_rois
    _dynamic_sign_rois = [(b[0], b[1], b[2], b[3]) for b in sticker_boxes]

    cv2.imwrite(str(ROOT / "track_mask.png"), clean)
    print(f"[Auto] Track mask detected ({len(sticker_boxes)} sign(s) bridged) → track_mask.png")
    return clean


def is_on_track(saved_mask, wx, wy, check_r=20):
    """
    用预先保存的轨道掩膜检测 (wx, wy) 是否在轨道上。
    Sign ROI positions always count as on-track (the track passes under the signs).
    """
    global _track_history, _track_status

    # Sign ROIs are on the track by definition — short-circuit mask check
    for rx1, ry1, rx2, ry2 in _dynamic_sign_rois:
        if rx1 <= wx <= rx2 and ry1 <= wy <= ry2:
            return True

    x1 = max(0, wx - check_r);  x2 = min(WARP_W, wx + check_r)
    y1 = max(0, wy - check_r);  y2 = min(WARP_H, wy + check_r)
    region = saved_mask[y1:y2, x1:x2]

    if region.size == 0:
        return _track_status

    track_ratio = float(np.sum(region > 0)) / region.size
    raw = track_ratio > 0.1  # 超过10%是轨道像素 → ON TRACK

    _track_history.append(raw)
    if len(_track_history) > DEBOUNCE_FRAMES:
        _track_history.pop(0)

    if sum(_track_history) > len(_track_history) * 0.6:
        _track_status = True
    elif sum(_track_history) < len(_track_history) * 0.4:
        _track_status = False

    return _track_status


def detect_signs(warped):
    """
    Detect all three signs using YOLOv11n (track-specific model).
    Runs YOLO on the full 600x600 warped image (same as training context),
    then filters detections by ROI region.
    Returns dict:
        'light': 'OFF'/'GREEN'/'RED'
        'stop':  bool
        'speed': bool
        'boxes': list of (x1,y1,x2,y2, label, color) for drawing
    """
    model = _get_sign_model()

    KNOWN_CLASSES = {'light_off', 'light_green', 'light_red', 'stop', 'speed_55'}
    BOX_COLOR = {
        'stop':        (0,   50, 255),
        'speed_55':    (255, 80,   0),
        'light_off':   (120, 120, 120),
        'light_green': (0,  200,   0),
        'light_red':   (0,   0,  220),
    }
    BOX_LABEL = {
        'stop':        'STOP',
        'speed_55':    '55',
        'light_off':   'OFF',
        'light_green': 'GREEN',
        'light_red':   'RED',
    }

    # Run YOLO at 1280px — upscales 600×600 internally so small signs on
    # the large table get enough pixels. Boxes are returned in 600×600 space.
    # No spatial ROI filter: sign positions differ between the two tables,
    # so the custom model's confidence threshold is the only gate.
    # Reject any detection whose bounding box touches the image border —
    # these are always artifacts of the perspective warp fill region.
    BORDER = 10

    best = {}      # class_name -> (conf, (x1,y1,x2,y2))
    _raw = []
    for r in model(warped, verbose=False, conf=0.01, imgsz=1280):
        for box, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            name = model.names[int(c)]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            conf_f = float(conf)
            border_fail = (x1 < BORDER or y1 < BORDER or x2 > WARP_W - BORDER or y2 > WARP_H - BORDER)
            _raw.append((name, conf_f, x1, y1, x2, y2, border_fail))
    for name, conf_f, x1, y1, x2, y2, border_fail in _raw:
        if name not in KNOWN_CLASSES or border_fail or conf_f < SIGN_CONF:
            continue
        if conf_f > best.get(name, (0.0, None))[0]:
            best[name] = (conf_f, (x1, y1, x2, y2))

    light_cls = max(
        (n for n in ('light_off', 'light_green', 'light_red') if n in best),
        key=lambda n: best[n][0],
        default=None,
    )
    light_map = {'light_green': 'GREEN', 'light_red': 'RED', 'light_off': 'OFF'}
    light = light_map.get(light_cls, 'OFF')

    # Build box list for drawing
    boxes = []
    for name, (conf_f, box) in best.items():
        x1, y1, x2, y2 = box
        label = f"{BOX_LABEL[name]} {conf_f:.2f}"
        color = BOX_COLOR[name]
        boxes.append((x1, y1, x2, y2, label, color, name))  # name used for bridge filtering

    return {
        'light': light,
        'stop':  'stop'     in best,
        'speed': 'speed_55' in best,
        'boxes': boxes,
    }


def draw_warped(frame, markers, M, saved_mask=None):
    """透视矫正后的画面，追踪小车位置并判断是否在轨道上"""
    global _stop_event
    warped = cv2.warpPerspective(frame, M, (WARP_W, WARP_H))

    # 'h' 隐藏信息文字/检测框，测距点标记点时不被遮挡（测距叠加层始终保留）
    hud = draw_warped._show_hud

    # 在矫正图上标出小车
    if CAR_ID in markers:
        cx, cy, _ = markers[CAR_ID]
        wx, wy = warp_point((cx, cy), M)
        # The car marker sits above the table, so pull it back onto the plane
        # before it is displayed, measured or logged.
        wxf, wyf = correct_parallax(wx, wy, M, frame.shape)
        wx, wy = int(round(wxf)), int(round(wyf))
        if 0 <= wx < WARP_W and 0 <= wy < WARP_H:
            # Trajectory logging: append this frame's position while recording
            if _traj_recording and _dist_scale_x and _dist_scale_y:
                _traj_rows.append((round(time.perf_counter() - _traj_t0, 3), wx, wy,
                                   round(wx * _dist_scale_x, 2), round(wy * _dist_scale_y, 2)))
            on_track = is_on_track(saved_mask, wx, wy) if saved_mask is not None else False
            status_text = "ON TRACK" if on_track else "OFF TRACK"
            status_color = (0, 255, 0) if on_track else (0, 0, 255)

            # 小车位置圆圈（始终保留）
            cv2.circle(warped, (wx, wy), 18, status_color, 3)
            cv2.circle(warped, (wx, wy), 5, status_color, -1)
            if hud:
                cv2.putText(warped, f"CAR ({wx},{wy})", (wx + 20, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(warped, f"CAR ({wx},{wy})", (wx + 20, wy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

                # 大字显示 ON/OFF TRACK
                cv2.putText(warped, status_text, (10, WARP_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
                cv2.putText(warped, status_text, (10, WARP_H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

    # STOP compliance event display (2-second timeout)
    if hud and time.perf_counter() < _stop_event['until']:
        ev = _stop_event
        cv2.putText(warped, ev['text'], (10, WARP_H - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
        cv2.putText(warped, ev['text'], (10, WARP_H - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, ev['color'], 2)

    # 画边框（始终保留，方便看到边界）
    cv2.rectangle(warped, (0, 0), (WARP_W - 1, WARP_H - 1), (0, 255, 255), 2)
    if hud:
        cv2.putText(warped, "Top-Down View", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # Show inferred table size — derived purely from the 10.5 cm ArUco markers
        if _dist_scale_x and _dist_scale_y:
            tw = WARP_W * _dist_scale_x
            th = WARP_H * _dist_scale_y
            size_text = f"Table: {tw:.0f} x {th:.0f} cm"
            cv2.putText(warped, size_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(warped, size_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Track mask status indicator (bottom-left)
    if hud:
        if saved_mask is not None:
            mask_text, mask_color = "TRACK: detected", (0, 220, 0)
        elif _mask_running:
            mask_text, mask_color = "TRACK: detecting...", (0, 165, 255)
        else:
            mask_text, mask_color = "TRACK: press c", (80, 80, 80)
        cv2.putText(warped, mask_text, (10, WARP_H - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(warped, mask_text, (10, WARP_H - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color, 1)

    # Sign detection: background thread fires every SIGN_EVERY_N frames
    global _sign_cache, _sign_frame_cnt
    if _sign_result[0] is not None:
        _sign_cache = _sign_result[0]
        _sign_result[0] = None
    _sign_frame_cnt += 1
    if _sign_frame_cnt % SIGN_EVERY_N == 0:
        _trigger_sign_detection(warped.copy())

    signs = _sign_cache
    light_color = {'GREEN': (0, 220, 0), 'RED': (0, 0, 255), 'OFF': (120, 120, 120)}
    hud_items = [
        (f"[LIGHT] {signs['light']}",
         light_color.get(signs['light'], (120, 120, 120))),
        (f"[STOP]  {'YES' if signs['stop']  else '---'}",
         (0, 0, 255) if signs['stop']  else (120, 120, 120)),
        (f"[55]    {'YES' if signs['speed'] else '---'}",
         (0, 0, 255) if signs['speed'] else (120, 120, 120)),
    ]
    if hud:
        for i, (text, color) in enumerate(hud_items):
            y = 30 + i * 28
            cv2.putText(warped, text, (WARP_W - 185, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(warped, text, (WARP_W - 185, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw YOLO bounding boxes on detected signs
        for x1, y1, x2, y2, label, color, *_ in signs['boxes']:
            cv2.rectangle(warped, (x1, y1), (x2, y2), color, 2)
            cv2.putText(warped, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4)
            cv2.putText(warped, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # Debug: draw dynamic sign ROIs detected by YOLO, press 'r' to toggle
    if draw_warped._show_rois:
        for rx1, ry1, rx2, ry2 in _dynamic_sign_rois:
            cv2.rectangle(warped, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(warped, 'SIGN ROI', (rx1 + 3, ry1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Camera nadir crosshair ('n' survey origin) — parallax and radial distortion
    # are both zero here and grow outward, so it's the reference for the survey.
    if draw_warped._show_nadir:
        nad = nadir_in_warp(M, frame.shape[1], frame.shape[0])
        if nad is not None:
            nx, ny = int(round(nad[0])), int(round(nad[1]))
            cv2.line(warped, (nx - 18, ny), (nx + 18, ny), (255, 0, 255), 2)
            cv2.line(warped, (nx, ny - 18), (nx, ny + 18), (255, 0, 255), 2)
            cv2.circle(warped, (nx, ny), 24, (255, 0, 255), 1)
            cv2.putText(warped, "(0,0)", (nx + 27, ny - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    warped = _draw_dist_overlay(warped)

    # 调试：左=俯视图，右=轨道掩膜，按 'm' 切换
    if draw_warped._show_mask and saved_mask is not None:
        mask_bgr = cv2.cvtColor(saved_mask, cv2.COLOR_GRAY2BGR)
        warped = np.hstack([warped, mask_bgr])

    return warped

draw_warped._show_mask = False
draw_warped._show_rois = False
draw_warped._show_hud  = True   # 'h' toggles info overlays (keeps measurement UI)
draw_warped._show_nadir = True  # 'x' toggles the camera-nadir crosshair


# ── Distance measurement helpers ──────────────────────────────────────


def _on_mouse(event, x, y, flags, param):
    global _dist_pt_a, _dist_pt_b
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if x >= WARP_W or y >= WARP_H:
        return
    if _dist_mode:
        pt = (x, y)
        if _dist_pt_a is None:
            _dist_pt_a = pt
        elif _dist_pt_b is None:
            _dist_pt_b = pt
        else:
            _dist_pt_a = pt
            _dist_pt_b = None


def _draw_dist_overlay(img):
    """Draw distance measurement UI onto the 600×600 warped image."""
    if not _dist_mode:
        return img

    # ── Measurement hint ──────────────────────────────────────────────
    hint = ('[DIST] Click A' if _dist_pt_a is None else
            '[DIST] Click B' if _dist_pt_b is None else
            '[DIST] Click to reset')
    # y=167 keeps clear of the main HUD, whose last line now sits at y=136
    cv2.putText(img, hint, (10, 167), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, hint, (10, 167), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ── Measurement points and result ─────────────────────────────────
    if _dist_pt_a is not None:
        cv2.circle(img, _dist_pt_a, 7, (0, 0, 210), -1)
        cv2.putText(img, 'A', (_dist_pt_a[0] + 9, _dist_pt_a[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 210), 2)
    if _dist_pt_b is not None:
        cv2.circle(img, _dist_pt_b, 7, (0, 180, 0), -1)
        cv2.putText(img, 'B', (_dist_pt_b[0] + 9, _dist_pt_b[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)

    if _dist_pt_a is not None and _dist_pt_b is not None:
        cv2.line(img, _dist_pt_a, _dist_pt_b, (220, 220, 220), 1)
        mx = (_dist_pt_a[0] + _dist_pt_b[0]) // 2
        my = (_dist_pt_a[1] + _dist_pt_b[1]) // 2
        if _dist_scale_x and _dist_scale_y:
            dx_r = (_dist_pt_b[0] - _dist_pt_a[0]) * _dist_scale_x
            dy_r = (_dist_pt_b[1] - _dist_pt_a[1]) * _dist_scale_y
            d_cm = float(np.sqrt(dx_r ** 2 + dy_r ** 2))
            label = f'{d_cm / 100:.2f} m' if d_cm >= 100 else f'{d_cm:.1f} cm'
        else:
            label = 'scale not ready'
        cv2.putText(img, label, (mx + 4, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, label, (mx + 4, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def _log_measurement():
    """
    Log the current A–B measurement (label + tape-measured ground truth) to
    evaluation/measurements.csv from a BACKGROUND thread.

    The terminal prompt runs in a daemon thread so it never blocks the main cv2
    loop — on macOS a blocking input() freezes the video window.  The measurement
    values are snapshotted at keypress time, so they can't drift while the prompt
    is open.  Only one prompt runs at a time.  Empty input (or Ctrl-D) cancels
    without writing a row.  Feeds both studies:
      • accuracy    — unique label per fixed point-pair + its true distance;
      • repeatability — reuse one label (e.g. "repeat-50") across disturbed runs.
    """
    global _logging_busy
    if _logging_busy:
        print("[Log] 请先在终端完成上一条记录（输入标签和真值）")
        return
    if _dist_pt_a is None or _dist_pt_b is None:
        print("[Log] 需要先按 'p' 进入测距并点击 A、B 两点再记录")
        return
    if not (_dist_scale_x and _dist_scale_y):
        print("[Log] 比例尺尚未就绪，请确保 4 个角标可见")
        return

    # Snapshot everything now so it can't change while the prompt is open.
    ax, ay = _dist_pt_a
    bx, by = _dist_pt_b
    sx, sy = _dist_scale_x, _dist_scale_y
    ww, wh = WARP_W, WARP_H
    measured_cm = float(np.hypot((bx - ax) * sx, (by - ay) * sy))
    print(f"[Log] 已记下 measured={measured_cm:.2f}cm — 在终端输入标签和真值"
          f"（空输入=跳过；窗口不会卡）")

    _logging_busy = True

    def _prompt_and_write():
        global _logging_busy
        try:
            try:
                label = input("  标签 (例: val-c1 / repeat-50): ").strip()
                gt    = input("  实际距离 cm (没有可留空): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Log] 已取消（未写入）")
                return
            if not label and not gt:
                print("[Log] 空输入，已跳过（未写入）")
                return
            is_new = not os.path.exists(MEASURE_LOG)
            os.makedirs(os.path.dirname(MEASURE_LOG), exist_ok=True)
            with open(MEASURE_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["timestamp", "label", "ax", "ay", "bx", "by",
                                     "scale_x_cmpx", "scale_y_cmpx", "warp_w", "warp_h",
                                     "measured_cm", "ground_truth_cm"])
                writer.writerow([datetime.now().isoformat(timespec="seconds"), label,
                                 ax, ay, bx, by, f"{sx:.5f}", f"{sy:.5f}",
                                 ww, wh, f"{measured_cm:.2f}", gt])
            print(f"[Log] 记录: measured={measured_cm:.2f}cm  label='{label}'  gt='{gt}'  → {MEASURE_LOG}")
        finally:
            _logging_busy = False

    threading.Thread(target=_prompt_and_write, daemon=True).start()


def nadir_in_warp(M, frame_w, frame_h):
    """
    Where the camera looks straight down, expressed in warped-view pixels.

    Both error terms under investigation are radial about this point: a marker
    raised h above the table is pushed outward by H/(H-h), and residual lens
    distortion also varies with radius. Directly beneath the camera both vanish,
    so this is the origin the corrections must be measured from.

    Approximated by the frame centre — the saved calibration puts the principal
    point within 0.4 % of it — projected through the homography.
    """
    if M is None:
        return None
    p = np.float32([[[frame_w / 2.0, frame_h / 2.0]]])
    w = cv2.perspectiveTransform(p, M).reshape(2)
    return float(w[0]), float(w[1])


def track_camera_height(markers):
    """
    Follow the camera height from the corner markers' separation in the original
    image, so moving the rig doesn't silently invalidate the parallax term.

    ID0 and ID1 are a fixed distance apart on the table, so their pixel
    separation scales as 1/H. The first stable frame anchors H0 * px against the
    measured --height; from then on H = anchor / px. Raise the camera and the
    markers draw closer together, and H rises to match.
    """
    global _h_anchor, _h_live
    if not _mount_height or 0 not in markers or 1 not in markers:
        return _h_live
    px = float(np.hypot(markers[0][0] - markers[1][0],
                        markers[0][1] - markers[1][1]))
    if px < 1:
        return _h_live
    if _h_anchor is None:
        _h_anchor = _mount_height * px
        _h_live = _mount_height
        return _h_live
    h_new = _h_anchor / px
    _h_live = (H_TRACK_ALPHA * h_new + (1 - H_TRACK_ALPHA) * _h_live
               if _h_live else h_new)
    return _h_live


def correct_parallax(wx, wy, M, frame_shape):
    """
    Project a raised marker back onto the table plane.

    The homography maps the table plane, so a marker riding h above it is seen
    along a slanted ray and lands too far out — by H/(H-h), measured outward from
    the point directly under the camera, which is why the error is nil there and
    grows toward the edges. Scaling the offset from the nadir by (H-h)/H undoes
    it. Survey confirmed the model: the car marker imaged 8.64 % larger raised
    than flat, giving h = 11.6 cm against 11 cm on a ruler.

    Needs --marker-height and --height; without either the position is unchanged.
    """
    H = _h_live or _mount_height          # tracked height, else the typed one
    if not (_marker_height and H) or M is None:
        return wx, wy
    nad = nadir_in_warp(M, frame_shape[1], frame_shape[0])
    if nad is None:
        return wx, wy
    k = (H - _marker_height) / H
    return (nad[0] + (wx - nad[0]) * k,
            nad[1] + (wy - nad[1]) * k)


def _log_marker_survey(markers, M, frame_shape):
    """
    Append every detected marker to evaluation/marker_survey.csv: its position
    relative to the camera nadir and its apparent size, both in cm.

    Feeds two experiments (see docs/next_phase_plan.md):
      • flat marker moved around  → does apparent size grow with radius?
        (residual radial distortion — the suspected source of the scale offset)
      • raised marker moved around → does position error grow with radius?
        (parallax, expected to follow H/(H-h))
    """
    if M is None or not (_dist_scale_x and _dist_scale_y):
        print("[Survey] 需要 4 个角标可见并进入俯视图")
        return
    nad = nadir_in_warp(M, frame_shape[1], frame_shape[0])
    if nad is None:
        return
    scale = 0.5 * (_dist_scale_x + _dist_scale_y)   # isotropic after the aspect fix

    is_new = not os.path.exists(SURVEY_LOG)
    os.makedirs(os.path.dirname(SURVEY_LOG), exist_ok=True)
    rows = []
    for mid, (cx, cy, pts) in sorted(markers.items()):
        wp = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        centre = wp.mean(axis=0)
        # Edge lengths are rotation-invariant; a bounding box would inflate with tilt
        sides = [float(np.linalg.norm(wp[i] - wp[(i + 1) % 4])) for i in range(4)]
        size_cm = float(np.mean(sides)) * scale
        dx_cm = (centre[0] - nad[0]) * scale
        dy_cm = (centre[1] - nad[1]) * scale
        rows.append((mid, dx_cm, dy_cm, float(np.hypot(dx_cm, dy_cm)), size_cm))

    if not rows:
        print("[Survey] 当前没有检测到任何标记")
        return
    with open(SURVEY_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "marker_id", "dx_cm", "dy_cm",
                             "radius_cm", "apparent_size_cm", "height_cm",
                             "scale_cmpx", "note"])
        for mid, dx, dy, r, s in rows:
            writer.writerow([datetime.now().isoformat(timespec="seconds"), mid,
                             f"{dx:.2f}", f"{dy:.2f}", f"{r:.2f}", f"{s:.3f}",
                             _mount_height if _mount_height is not None else "",
                             f"{scale:.5f}", ""])
    print(f"[Survey] 记录 {len(rows)} 个标记 → {SURVEY_LOG}")
    for mid, dx, dy, r, s in rows:
        print(f"    ID{mid}: 位置({dx:+.1f},{dy:+.1f})cm  半径={r:.1f}cm  表观尺寸={s:.2f}cm")


def _log_scale(markers):
    """
    One-key, non-blocking log of the CURRENT ArUco-derived scale to
    evaluation/scale_repeatability.csv — for the scale-stability study
    (read → disturb the setup → read → repeat ~10×).

    Records both the warp scale (the measurement ruler) and the raw camera GSD
    (the marker-based source), so the spread of these across disturbances is the
    system's scale repeatability — the true accuracy limit found in validation.
    """
    if not (_dist_scale_x and _dist_scale_y):
        print("[Scale] 比例尺尚未就绪，请确保 4 个角标可见")
        return
    cg = camera_gsd_from_markers(markers)
    tw, th = WARP_W * _dist_scale_x, WARP_H * _dist_scale_y

    trial = 1
    if os.path.exists(SCALE_LOG):
        with open(SCALE_LOG) as f:
            trial = max(1, sum(1 for _ in f))   # header + n data rows → next trial = n+1

    os.makedirs(os.path.dirname(SCALE_LOG), exist_ok=True)
    is_new = not os.path.exists(SCALE_LOG)
    with open(SCALE_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "trial", "scale_x_cmpx", "scale_y_cmpx",
                             "table_w_cm", "table_h_cm", "camera_gsd_cmpx",
                             "warp_w", "warp_h", "height_cm"])
            trial = 1
        writer.writerow([datetime.now().isoformat(timespec="seconds"), trial,
                         f"{_dist_scale_x:.5f}", f"{_dist_scale_y:.5f}",
                         f"{tw:.1f}", f"{th:.1f}",
                         f"{cg:.5f}" if cg else "", WARP_W, WARP_H,
                         _mount_height if _mount_height is not None else ""])
    hstr = f"  H={_mount_height}cm" if _mount_height is not None else ""
    print(f"[Scale] 记录 #{trial}: scale_x={_dist_scale_x:.4f} scale_y={_dist_scale_y:.4f}  "
          f"cam_gsd={cg:.4f}  桌面={tw:.1f}x{th:.1f}cm{hstr}  → {SCALE_LOG}")


def _toggle_trajectory():
    """
    Start/stop logging the tracked CAR trajectory to evaluation/traj_<time>.csv
    for the dynamic tracking-accuracy test: drive the robot along a KNOWN path,
    then compare the tracked displacement / path length against the tape truth.
    Positions are stored in warp pixels and cm (via the ArUco scale) per frame.
    """
    global _traj_recording, _traj_rows, _traj_t0
    if not _traj_recording:
        _traj_rows = []
        _traj_t0 = time.perf_counter()
        _traj_recording = True
        print("[Traj] 开始记录轨迹 — 让机器人走完路线后再按 't' 停止")
        return

    _traj_recording = False
    if len(_traj_rows) < 2:
        print("[Traj] 记录到的点太少（机器人码没被追到？）— 未保存")
        return
    # Date + time: a time-only name silently overwrites a run from another day.
    fn = str(ROOT / "evaluation" / f"traj_{datetime.now():%Y%m%d_%H%M%S}.csv")
    with open(fn, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "wx", "wy", "x_cm", "y_cm"])
        writer.writerows(_traj_rows)
    x0, y0 = _traj_rows[0][3], _traj_rows[0][4]
    x1, y1 = _traj_rows[-1][3], _traj_rows[-1][4]
    disp = float(np.hypot(x1 - x0, y1 - y0))
    path = sum(float(np.hypot(_traj_rows[i][3] - _traj_rows[i - 1][3],
                              _traj_rows[i][4] - _traj_rows[i - 1][4]))
               for i in range(1, len(_traj_rows)))
    dur = _traj_rows[-1][0]
    spd = path / dur if dur > 0 else 0.0
    print(f"[Traj] 停止：{len(_traj_rows)} 点，用时 {dur:.1f}s")
    print(f"[Traj] 首尾位移={disp:.1f}cm  路径长={path:.1f}cm  均速={spd:.1f}cm/s  → {fn}")


def main():
    # Rectified-view resolution is overridable at launch; declare the globals up
    # front (before the argparse default reads WARP_BASE).
    global WARP_BASE, WARP_W, WARP_H, _mount_height, _marker_height
    global _baseline_x, _baseline_y
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=None,
                        help="摄像头编号（不指定则列出所有摄像头）")
    parser.add_argument("--warp-base", type=int, default=WARP_BASE,
                        help="俯视图长边像素（默认600；测量研究可调大如1520，更细但更慢）")
    parser.add_argument("--height", type=float, default=None,
                        help="相机镜头到桌面高度(cm)，精密度-高度扫描用；按 'j' 记录时写入 height_cm 列")
    parser.add_argument("--baseline-x", type=float, default=None,
                        help="卷尺量的 ID0→ID1 中心距(cm)；用长基线定标比例尺，比用码尺寸准得多")
    parser.add_argument("--baseline-y", type=float, default=None,
                        help="卷尺量的 ID0→ID3 中心距(cm)；需与 --baseline-x 同时给出")
    parser.add_argument("--marker-height", type=float, default=None,
                        help="车标离桌面的高度(cm)，开启视差修正；需同时给 --height")
    args = parser.parse_args()

    _marker_height = args.marker_height
    if _marker_height and not args.height:
        print("[Parallax] 给了 --marker-height 但没给 --height，视差修正未启用")
    elif _marker_height:
        k = (args.height - _marker_height) / args.height
        print(f"[Parallax] 视差修正已启用: 车标高 {_marker_height}cm / 相机高 {args.height}cm "
              f"→ 系数 {k:.4f}")

    if bool(args.baseline_x) != bool(args.baseline_y):
        print("[Scale] --baseline-x 和 --baseline-y 必须同时给出，本次忽略")
    elif args.baseline_x:
        _baseline_x, _baseline_y = args.baseline_x, args.baseline_y
        print(f"[Scale] 长基线定标: x={_baseline_x}cm  y={_baseline_y}cm")

    # Higher WARP_BASE = finer measurement quantisation (approaches the camera
    # GSD) at the cost of FPS; the mask still runs at the fixed MASK_DIM canvas,
    # so its tuning is unaffected.
    WARP_BASE = args.warp_base
    WARP_W, WARP_H = WARP_BASE, WARP_BASE
    _mount_height = args.height

    if args.source is None:
        # Only scan when user hasn't specified a source
        list_cameras()
        print("请用 --source 指定摄像头编号，例如：")
        print("  python aruco_detect.py --source 0")
        return

    cap = open_camera(args.source)
    if cap is None:
        print(f"无法打开摄像头 {args.source}")
        return

    # Ctrl+C must always exit, but it should still hand the device back — an
    # abrupt os._exit leaves the GoPro claimed and the next run can't open it.
    def _sigint(*_):
        print("\n已中断，正在释放摄像头…")
        release_camera(cap)
        os._exit(0)
    signal.signal(signal.SIGINT, _sigint)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"使用摄像头: {w}x{h}")
    print("操作: 'w'=透视矫正  'c'=捕获掩膜  'm'=显示掩膜  'r'=ROI框  'h'=隐藏文字  'x'=中心十字  'd'=畸变对比  's'=截图")
    print("      'k'=显示比例  'j'=记比例尺  'n'=记录标记(实验用)  'p'=测距  'l'=记录测量  't'=记录轨迹  'q'=退出")

    # 预计算畸变矫正映射表
    map1, map2, roi = build_undistort_maps(h, w)

    show_warp = False
    show_distort_compare = False
    shot_count = 0
    frame_count = 0
    t_start = time.perf_counter()
    global _sign_cache, _sign_frame_cnt
    global _dist_mode, _dist_pt_a, _dist_pt_b, _dist_scale_x, _dist_scale_y
    M = None
    saved_mask = None   # auto-detected track mask
    _M_stable_frames = 0
    _first_detect_done = False
    def _run_mask_detection(warped_snap, car_pos_snap, M_snap):
        """Run YOLO then mask in one background thread — guarantees fresh sign positions."""
        # finally: without it an exception leaves _mask_running (and possibly
        # _sign_running) stuck True, permanently disabling both re-detections.
        global _mask_running, _mask_M, _sign_running
        try:
            # Step 1: fresh YOLO — gives accurate sign positions for tape blanking
            fresh = detect_signs(warped_snap)
            _sign_result[0] = fresh   # let main loop pick up updated sign state
            _sign_running = False
            # Step 2: mask with guaranteed-fresh boxes
            result = auto_detect_track_mask(warped_snap, car_pos_snap, fresh['boxes'])
            _mask_result[0] = result
            if result is not None:
                _mask_M = M_snap
                print("✓ 轨道掩膜已更新")
            else:
                print("[Auto] 轨道识别失败，请检查光照或按 'c' 重试")
        except Exception as e:
            print(f"[Auto] 掩膜线程出错: {e}")
        finally:
            _sign_running = False
            _mask_running = False

    def _trigger_mask(warped_now, markers_now):
        """Snapshot current frame and kick off background detection (non-blocking)."""
        global _mask_running, _sign_running
        if _mask_running:
            return
        car_pos = None
        if CAR_ID in markers_now:
            cx, cy, _ = markers_now[CAR_ID]
            car_pos = warp_point((cx, cy), M)
        _mask_running = True
        _sign_running = True   # block concurrent YOLO until mask thread finishes its own
        threading.Thread(target=_run_mask_detection,
                         args=(warped_now.copy(), car_pos, M.copy()), daemon=True).start()

    print("使用步骤: 按'w'进入俯视图 → 轨道自动识别（后台运行，不影响画面）")
    print("'c'键可随时重新识别轨道  |  'r'显示ROI框  |  'm'显示掩膜")
    print("测距: 按 'w' 进入俯视图，4个角标可见后比例尺自动推算，按 'p' 进入测距模式，鼠标点 A / B 两点")

    cv2.namedWindow("ArUco Detection")
    cv2.setMouseCallback("ArUco Detection", _on_mouse)

    grabber = FrameGrabber(cap)
    last_stamp = 0.0
    STALL_SEC = 2.0          # no new frame for this long → warn, keep UI alive

    while True:
        frame, stamp = grabber.read()
        now = time.perf_counter()

        # No new frame: keep the window and keys alive instead of blocking.
        if frame is None or stamp == last_stamp:
            stalled_for = now - stamp if stamp else now - t_start
            if stalled_for > STALL_SEC:
                warn = np.zeros((160, 760, 3), dtype=np.uint8)
                cv2.putText(warn, "CAMERA STALLED", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                cv2.putText(warn, f"no frame for {stalled_for:.0f}s - press 'q' to quit",
                            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("ArUco Detection", warn)
            if (cv2.waitKey(30) & 0xFF) == ord('q'):
                grabber.stop()
                release_camera(cap)
                cv2.destroyAllWindows()
                os._exit(0)
            continue
        last_stamp = stamp

        frame_count += 1
        elapsed = time.perf_counter() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0

        # 畸变矫正（当前关闭，GoPro Wide 俯拍畸变极小）
        corrected = undistort_frame(frame, map1, map2, roi) if UNDISTORT_ENABLED else frame

        # 'd' 模式：左右对比原图 vs 矫正图
        if show_distort_compare:
            h1, w1 = frame.shape[:2]
            h2, w2 = corrected.shape[:2]
            target_h = min(h1, h2)
            left  = cv2.resize(frame,     (int(w1 * target_h / h1), target_h))
            right = cv2.resize(corrected, (int(w2 * target_h / h2), target_h))
            display = np.hstack([left, right])
            cv2.putText(display, "ORIGINAL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2)
            cv2.putText(display, "UNDISTORTED", (left.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
            cv2.imshow("ArUco Detection", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_distort_compare = False
            elif key == ord('s'):
                shot_count += 1
                fname = os.path.join(SCREENSHOT_DIR, f"aruco_{shot_count:03d}.jpg")
                cv2.imwrite(fname, display)
                print(f"截图保存: {fname}")
            continue

        # 用矫正后的图做 ArUco 检测，并对角标做 EMA 平滑 + persistence
        markers, corners, ids = detect_markers(corrected)
        markers = _stabilise_corners(markers)
        frame = corrected  # 后续显示都用矫正图

        # 有4个角就更新透视矩阵，并从ArUco标记尺寸自动推算比例尺
        if all(i in markers for i in [0, 1, 2, 3]):
            M = get_perspective_transform(markers)
            _M_stable_frames += 1
            track_camera_height(markers)   # keeps the parallax term current
            sx, sy = compute_scale_from_aruco(markers, M)

            # Once the corners are stable, lock the rectified view to the real
            # table aspect ratio (square pixels). Done once; afterwards sx ≈ sy.
            if sx is not None and not _aspect_locked and _M_stable_frames >= 15:
                if _lock_aspect_ratio(sx, sy):
                    M = get_perspective_transform(markers)         # rebuild at locked dims
                    sx, sy = compute_scale_from_aruco(markers, M)  # now ~isotropic
                    _dist_scale_x = _dist_scale_y = None           # re-init EMA to new scale
                    saved_mask = None                              # geometry changed → re-detect
                    _first_detect_done = False

            if _baseline_x and _baseline_y and _aspect_locked:
                # The homography pins the marker centres to the rectangle corners,
                # so the tape baselines give cm/px outright — exact, and constant,
                # so it neither drifts nor needs smoothing.
                _dist_scale_x = _baseline_x / WARP_W
                _dist_scale_y = _baseline_y / WARP_H
            elif sx is not None:
                if _dist_scale_x is None:
                    _dist_scale_x, _dist_scale_y = sx, sy          # first frame: init directly
                else:
                    _dist_scale_x = SCALE_SMOOTH_ALPHA * sx + (1 - SCALE_SMOOTH_ALPHA) * _dist_scale_x
                    _dist_scale_y = SCALE_SMOOTH_ALPHA * sy + (1 - SCALE_SMOOTH_ALPHA) * _dist_scale_y
        else:
            _M_stable_frames = 0

        # Pick up mask result from background thread
        if _mask_result[0] is not None:
            saved_mask = _mask_result[0]
            _mask_result[0] = None

        # Auto-detect: first time corners are stable in warp mode, or perspective shifted
        if show_warp and M is not None and not _mask_running and _M_stable_frames >= 30:
            need_detect = (not _first_detect_done) or \
                          (_first_detect_done and _perspective_changed(M))
            if need_detect:
                _first_detect_done = True
                warped_now = cv2.warpPerspective(frame, M, (WARP_W, WARP_H))
                saved_mask = None   # clear stale mask while new one computes
                _trigger_mask(warped_now, markers)

        # 决定显示内容
        if show_warp and M is not None:
            display = draw_warped(frame, markers, M, saved_mask)
        else:
            display = draw_raw(frame, markers, ids, corners)
            if show_warp and M is None:
                cv2.putText(display, "Need all 4 corner markers", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Track the CAR hit-rate every frame, whether or not the HUD is shown.
        car_found = CAR_ID in markers
        _car_hits.append(car_found)
        if len(_car_hits) > CAR_RATE_WINDOW:
            _car_hits.pop(0)
        car_rate = 100.0 * sum(_car_hits) / len(_car_hits)

        # HUD（'h' 可隐藏，避免遮挡左上角的标记点）
        if draw_warped._show_hud:
            found_corners = [i for i in [0, 1, 2, 3] if i in markers]
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Corners: {len(found_corners)}/4  {found_corners}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if len(found_corners) == 4 else (0, 165, 255), 2)
            cv2.putText(display, f"Car: {'Detected' if car_found else 'Not found'}",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if car_found else (80, 80, 80), 2)
            # Rolling detection rate — green ≥90%, orange ≥70%, red below
            rate_color = ((0, 255, 0) if car_rate >= 90 else
                          (0, 165, 255) if car_rate >= 70 else (0, 0, 255))
            cv2.putText(display, f"Detect: {car_rate:.0f}% (last {len(_car_hits)}f)",
                        (10, 109), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rate_color, 2)
            mode_text = "Mode: Warp" if (show_warp and M is not None) else "Mode: Raw"
            if _h_live and _marker_height:
                mode_text += f"  H:{_h_live:.0f}cm"
            cv2.putText(display, mode_text, (10, 136),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("ArUco Detection", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            grabber.stop()
            release_camera(cap)   # timed: frees the device, never hangs the exit
            cv2.destroyAllWindows()
            os._exit(0)
        elif key == ord('w'):
            show_warp = not show_warp
            if not show_warp:       # leaving warp view — cancel any active dist mode
                _dist_mode = False
                _dist_pt_a = None
                _dist_pt_b = None
        elif key == ord('d'):
            show_distort_compare = True
        elif key == ord('c'):
            # Re-run auto track detection (car can be present — its position is masked out)
            if show_warp and M is not None:
                warped_now = cv2.warpPerspective(frame, M, (WARP_W, WARP_H))
                _mask_result[0] = None
                saved_mask = None
                _trigger_mask(warped_now, markers)
                print("轨道识别中（后台）…")
            else:
                print("请先按 'w' 进入俯视图模式再按 'c'")
        elif key == ord('m'):
            draw_warped._show_mask = not draw_warped._show_mask
        elif key == ord('r'):
            draw_warped._show_rois = not draw_warped._show_rois
        elif key == ord('h'):
            # Hide/show info overlays so they don't occlude the marker points
            draw_warped._show_hud = not draw_warped._show_hud
            print(f"[HUD] 信息文字 {'显示' if draw_warped._show_hud else '已隐藏'}")
        elif key == ord('s'):
            shot_count += 1
            fname = os.path.join(SCREENSHOT_DIR, f"aruco_{shot_count:03d}.jpg")
            cv2.imwrite(fname, display)
            print(f"截图保存: {fname}")
        elif key == ord('k'):
            if _dist_scale_x and _dist_scale_y:
                tw = WARP_W * _dist_scale_x
                th = WARP_H * _dist_scale_y
                print(f"[Dist] warp 比例尺: x={_dist_scale_x:.4f} cm/px  y={_dist_scale_y:.4f} cm/px  (600px 俯视图分辨率)")
                print(f"[Dist] 推算桌面尺寸: {tw:.1f} cm × {th:.1f} cm  (仅凭 ArUco {ARUCO_REAL_SIZE_CM} cm 推算)")
                cg = camera_gsd_from_markers(markers)
                if cg:
                    print(f"[Dist] 相机 GSD(原始帧): {cg:.4f} cm/px  ← 精密度分析用这个 --measured-gsd")
            else:
                print("[Dist] 尚未推算比例尺，请先按 'w' 进入俯视图并确保4个角标可见")
        elif key == ord('p'):
            if show_warp and M is not None:
                _dist_mode = not _dist_mode
                if not _dist_mode:
                    _dist_pt_a = None
                    _dist_pt_b = None
                print(f"[Dist] 测距模式 {'已开启 — 点击 A、B 两点' if _dist_mode else '已关闭'}")
        elif key == ord('l'):
            # Log current A–B measurement (+ ground truth) for accuracy/repeatability
            _log_measurement()
        elif key == ord('j'):
            # Log the current ArUco scale for the scale-stability / repeatability study
            _log_scale(markers)
        elif key == ord('t'):
            # Start/stop logging the CAR trajectory (dynamic tracking accuracy)
            _toggle_trajectory()
        elif key == ord('n'):
            # Survey every visible marker (position + apparent size vs nadir)
            _log_marker_survey(markers, M, frame.shape)
        elif key == ord('x'):
            draw_warped._show_nadir = not draw_warped._show_nadir

    grabber.stop()
    release_camera(cap)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n已退出")
