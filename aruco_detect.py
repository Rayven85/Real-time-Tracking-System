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

# ── 畸变矫正开关 ──────────────────────────────────────────────────
# GoPro Wide 在当前俯拍高度畸变极小，关闭矫正效果更好
UNDISTORT_ENABLED = False
UNDISTORT_K1 = -0.462
UNDISTORT_K2 = -0.054
UNDISTORT_FX_SCALE = 23   # %

SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ArUco 字典（与生成时一致）
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# ID 定义
CORNER_IDS = {0: '左上', 1: '右上', 2: '右下', 3: '左下'}
CAR_ID = 4

# 矫正后输出画面大小
WARP_W, WARP_H = 600, 600

# ON/OFF TRACK 防抖：连续 N 帧一致才切换
DEBOUNCE_FRAMES = 8
_track_history = []   # 最近N帧的判断结果
_track_status = False  # 当前稳定状态

# ── STOP compliance ──────────────────────────────────────────────────
STOP_SPEED_THRESHOLD = 5.0  # px/s — below this counts as stopped
STOP_ZONE_PAD = 20          # px padding around ROI_STOP for detection
STOP_DISPLAY_SEC = 2.0      # how long to show STOP event message
_stop_event = {'text': '', 'color': (0, 0, 0), 'until': 0.0}

# ── Sign detection ROIs in warped 600×600 view ──────────────────────
# Format: (x1, y1, x2, y2)  — press 'r' to visualize, tune as needed
ROI_LIGHT = (10,  252, 72,  315)   # Left: circular LED indicator
ROI_STOP  = (305, 472, 402, 550)   # Bottom: STOP sign
ROI_55    = (468, 283, 568, 362)   # Right: speed limit 55 sign

# ── YOLO sign detection ──────────────────────────────────────────────
SIGN_MODEL_PATH = "runs/detect/runs/train/track_signs/weights/best.pt"
SIGN_CONF       = 0.20
_sign_model     = None   # lazy-loaded on first detection frame


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
            cap.release()
            time.sleep(0.3)   # let macOS release the device before next open
            print(f"  [{i}] {w}x{h}")
    print()


def open_camera(source):
    """打开指定编号的摄像头"""
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    # GoPro Webcam 在 Mac 上需要先丢弃几帧才能稳定
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"摄像头 [{source}] 打开但无法读取帧")
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"使用摄像头 [{source}]: {w}x{h}")
    return cap


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


def capture_track_mask(warped_gray, threshold=80):
    """
    从没有 car 标记的俯视图中提取轨道掩膜并保存。
    只需运行一次（按 'c' 触发）。
    """
    _, mask = cv2.threshold(warped_gray, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("track_mask.png", mask)
    print("轨道掩膜已保存: track_mask.png")
    return mask


def is_on_track(saved_mask, wx, wy, check_r=20):
    """
    用预先保存的轨道掩膜检测 (wx, wy) 是否在轨道上。
    saved_mask 是没有 car 标记时拍的，所以轨道完整无遮挡。
    """
    global _track_history, _track_status

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

    CLASS_ROI = {
        'light_off':   ROI_LIGHT,
        'light_green': ROI_LIGHT,
        'light_red':   ROI_LIGHT,
        'stop':        ROI_STOP,
        'speed_55':    ROI_55,
    }
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

    # Run YOLO on full warped image — same context as training data
    best = {}      # class_name -> (conf, xyxy)
    for r in model(warped, verbose=False, conf=SIGN_CONF):
        for box, c, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            name = model.names[int(c)]
            conf_f = float(conf)
            if conf_f <= best.get(name, (0.0, None))[0]:
                continue
            roi = CLASS_ROI.get(name)
            if roi is None:
                continue
            bx = float((box[0] + box[2]) / 2)
            by = float((box[1] + box[3]) / 2)
            if roi[0] <= bx <= roi[2] and roi[1] <= by <= roi[3]:
                best[name] = (conf_f, box)

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
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        label = f"{BOX_LABEL[name]} {conf_f:.2f}"
        color = BOX_COLOR[name]
        boxes.append((x1, y1, x2, y2, label, color))

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

    # 在矫正图上标出小车
    if CAR_ID in markers:
        cx, cy, _ = markers[CAR_ID]
        wx, wy = warp_point((cx, cy), M)
        if 0 <= wx < WARP_W and 0 <= wy < WARP_H:
            on_track = is_on_track(saved_mask, wx, wy) if saved_mask is not None else False
            status_text = "ON TRACK" if on_track else "OFF TRACK"
            status_color = (0, 255, 0) if on_track else (0, 0, 255)

            # 小车位置圆圈
            cv2.circle(warped, (wx, wy), 18, status_color, 3)
            cv2.circle(warped, (wx, wy), 5, status_color, -1)
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
    if time.perf_counter() < _stop_event['until']:
        ev = _stop_event
        cv2.putText(warped, ev['text'], (10, WARP_H - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
        cv2.putText(warped, ev['text'], (10, WARP_H - 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, ev['color'], 2)

    # 画边框
    cv2.rectangle(warped, (0, 0), (WARP_W - 1, WARP_H - 1), (0, 255, 255), 2)
    cv2.putText(warped, "Top-Down View", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Sign detection HUD (top-right corner) + bounding boxes
    signs = detect_signs(warped)
    light_color = {'GREEN': (0, 220, 0), 'RED': (0, 0, 255), 'OFF': (120, 120, 120)}
    hud_items = [
        (f"[LIGHT] {signs['light']}",
         light_color.get(signs['light'], (120, 120, 120))),
        (f"[STOP]  {'YES' if signs['stop']  else '---'}",
         (0, 0, 255) if signs['stop']  else (120, 120, 120)),
        (f"[55]    {'YES' if signs['speed'] else '---'}",
         (0, 0, 255) if signs['speed'] else (120, 120, 120)),
    ]
    for i, (text, color) in enumerate(hud_items):
        y = 30 + i * 28
        cv2.putText(warped, text, (WARP_W - 185, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(warped, text, (WARP_W - 185, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw YOLO bounding boxes on detected signs
    for x1, y1, x2, y2, label, color in signs['boxes']:
        cv2.rectangle(warped, (x1, y1), (x2, y2), color, 2)
        cv2.putText(warped, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 4)
        cv2.putText(warped, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    # Debug: draw ROI rectangles, press 'r' to toggle
    if draw_warped._show_rois:
        for roi, label, col in [
            (ROI_LIGHT, 'LIGHT', (0, 255, 255)),
            (ROI_STOP,  'STOP',  (0, 80,  255)),
            (ROI_55,    '55',    (255, 80, 0)),
        ]:
            x1, y1, x2, y2 = roi
            cv2.rectangle(warped, (x1, y1), (x2, y2), col, 2)
            cv2.putText(warped, label, (x1 + 3, y1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # 调试：左=俯视图，右=轨道掩膜，按 'm' 切换
    if draw_warped._show_mask and saved_mask is not None:
        mask_bgr = cv2.cvtColor(saved_mask, cv2.COLOR_GRAY2BGR)
        warped = np.hstack([warped, mask_bgr])

    return warped

draw_warped._show_mask = False
draw_warped._show_rois = False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=int, default=None,
                        help="摄像头编号（不指定则列出所有摄像头）")
    args = parser.parse_args()

    list_cameras()
    time.sleep(1.0)   # give macOS time to fully release devices after scan

    if args.source is None:
        print("请用 --source 指定摄像头编号，例如：")
        print("  python aruco_detect.py --source 1")
        return

    cap = open_camera(args.source)
    if cap is None:
        print(f"无法打开摄像头 {args.source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"使用摄像头: {w}x{h}")
    print("操作: 'w'=透视矫正  'c'=捕获掩膜  'm'=显示掩膜  'r'=ROI框  't'=清轨迹  'd'=畸变对比  's'=截图  'q'=退出")

    # 预计算畸变矫正映射表
    map1, map2, roi = build_undistort_maps(h, w)

    show_warp = False
    show_distort_compare = False
    shot_count = 0
    frame_count = 0
    t_start = time.perf_counter()
    M = None
    saved_mask = None  # 预先扫描保存的轨道掩膜

    print("使用步骤: 1)先不放car标记，按'w'进入俯视图  2)按'c'捕获轨道  3)放上car标记")
    print("标识检测: 'r'键显示ROI框，确认框住LIGHT/STOP/55后关闭")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        # 用矫正后的图做 ArUco 检测
        markers, corners, ids = detect_markers(corrected)
        frame = corrected  # 后续显示都用矫正图

        # 有4个角就更新透视矩阵
        if all(i in markers for i in [0, 1, 2, 3]):
            M = get_perspective_transform(markers)

        # 决定显示内容
        if show_warp and M is not None:
            display = draw_warped(frame, markers, M, saved_mask)
        else:
            display = draw_raw(frame, markers, ids, corners)
            if show_warp and M is None:
                cv2.putText(display, "Need all 4 corner markers", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # HUD
        found_corners = [i for i in [0, 1, 2, 3] if i in markers]
        car_found = CAR_ID in markers
        cv2.putText(display, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Corners: {len(found_corners)}/4  {found_corners}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if len(found_corners) == 4 else (0, 165, 255), 2)
        cv2.putText(display, f"Car: {'Detected' if car_found else 'Not found'}",
                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if car_found else (80, 80, 80), 2)
        mode_text = "Mode: Warp" if (show_warp and M is not None) else "Mode: Raw"
        cv2.putText(display, mode_text, (10, 109),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("ArUco Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            show_warp = not show_warp
        elif key == ord('d'):
            show_distort_compare = True
        elif key == ord('c'):
            # 捕获当前俯视图作为轨道掩膜（此时不要放 car 标记）
            if show_warp and M is not None:
                warped_now = cv2.warpPerspective(frame, M, (WARP_W, WARP_H))
                warped_gray = cv2.cvtColor(warped_now, cv2.COLOR_BGR2GRAY)
                saved_mask = capture_track_mask(warped_gray)
                print("✓ 轨道掩膜已捕获！现在可以放上 car 标记了")
            else:
                print("请先按 'w' 进入俯视图模式再按 'c'")
        elif key == ord('m'):
            draw_warped._show_mask = not draw_warped._show_mask
        elif key == ord('r'):
            draw_warped._show_rois = not draw_warped._show_rois
        elif key == ord('s'):
            shot_count += 1
            fname = os.path.join(SCREENSHOT_DIR, f"aruco_{shot_count:03d}.jpg")
            cv2.imwrite(fname, display)
            print(f"截图保存: {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
