"""
Real-time Camera Testing & Detection Pipeline
Usage:
  python test.py --mode camera_test          # 摄像头驱动调试
  python test.py --mode detect               # 实时检测 pipeline
  python test.py --mode fps_test             # FPS 压测
  python test.py --mode conf_sweep           # 置信度阈值交互调整
"""

import argparse
import time
import cv2
from ultralytics import YOLO

# ─── 颜色配置（BGR） ────────────────────────────────────────────────────────
CLASS_COLORS = [
    (0, 0, 255),    # danger      → 红
    (0, 200, 0),    # mandatory   → 绿
    (255, 150, 0),  # other       → 蓝橙
    (0, 165, 255),  # prohibitory → 橙
]


def open_camera(source: int):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 (source={source})")
    return cap


def draw_boxes(frame, results, class_names):
    boxes = results[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names.get(cls_id, cls_id)} {conf:.2f}"
        color = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


# ─── Mode 1: 摄像头驱动调试 ─────────────────────────────────────────────────
def camera_test(source: int, duration: float = 5.0):
    cap = open_camera(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[camera_test] 分辨率: {w}x{h}  |  驱动报告 FPS: {reported_fps:.1f}")

    count = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration:
        ret, frame = cap.read()
        if not ret:
            print("[camera_test] 取帧失败，退出")
            break
        count += 1
        elapsed = time.perf_counter() - t_start
        fps_live = count / elapsed
        cv2.putText(frame, f"FPS: {fps_live:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Camera Test (press q to quit early)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    elapsed = time.perf_counter() - t_start
    actual_fps = count / elapsed if elapsed > 0 else 0
    print(f"[camera_test] 共取帧 {count} 帧，用时 {elapsed:.2f}s，实际 FPS: {actual_fps:.1f}")
    cap.release()
    cv2.destroyAllWindows()


# ─── Mode 2: 实时检测 Pipeline ───────────────────────────────────────────────
def detect(source: int, model_path: str, conf: float):
    model = YOLO(model_path)
    class_names = model.names
    cap = open_camera(source)
    print(f"[detect] 模型: {model_path}  |  conf_threshold: {conf}")
    print("[detect] 按 q 退出")

    prev_time = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        frame = draw_boxes(frame, results, class_names)

        now = time.perf_counter()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Real-time Detection (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─── Mode 3: FPS 压测 ────────────────────────────────────────────────────────
def fps_test(source: int, model_path: str, conf: float, n_frames: int):
    model = YOLO(model_path)
    class_names = model.names
    cap = open_camera(source)
    print(f"[fps_test] 模型: {model_path}  |  测试帧数: {n_frames}")

    latencies = []
    count = 0
    while count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        results = model.predict(frame, conf=conf, verbose=False)
        frame = draw_boxes(frame, results, class_names)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)
        count += 1

        fps_cur = 1.0 / (latencies[-1] + 1e-9)
        cv2.putText(frame, f"FPS: {fps_cur:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("FPS Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        avg_fps = 1.0 / avg_lat
        min_fps = 1.0 / max(latencies)
        max_fps = 1.0 / min(latencies)
        print(f"\n[fps_test] 结果 ({len(latencies)} 帧):")
        print(f"  平均延迟: {avg_lat*1000:.1f} ms")
        print(f"  平均 FPS: {avg_fps:.1f}")
        print(f"  最低 FPS: {min_fps:.1f}")
        print(f"  最高 FPS: {max_fps:.1f}")
        if avg_fps >= 15:
            print("  ✓ 达到 15 FPS 目标")
        else:
            print("  ✗ 未达到 15 FPS，建议降低分辨率或换更小模型")


# ─── Mode 4: 置信度交互调整 ──────────────────────────────────────────────────
def conf_sweep(source: int, model_path: str, conf_init: float):
    model = YOLO(model_path)
    class_names = model.names
    cap = open_camera(source)
    conf = conf_init
    print(f"[conf_sweep] 按 '+' 提高阈值，按 '-' 降低阈值，按 'q' 退出")

    prev_time = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        frame = draw_boxes(frame, results, class_names)

        now = time.perf_counter()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now
        n_det = len(results[0].boxes)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"conf: {conf:.2f}  dets: {n_det}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(frame, "+/- adjust conf | q quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Conf Sweep (press +/- to adjust)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            conf = min(conf + 0.05, 0.95)
            print(f"  conf → {conf:.2f}")
        elif key == ord('-'):
            conf = max(conf - 0.05, 0.05)
            print(f"  conf → {conf:.2f}")

    print(f"[conf_sweep] 最终阈值: {conf:.2f}")
    cap.release()
    cv2.destroyAllWindows()


# ─── Entry Point ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="摄像头测试 & 实时检测")
    parser.add_argument('--mode', choices=['camera_test', 'detect', 'fps_test', 'conf_sweep'],
                        default='detect', help='运行模式')
    parser.add_argument('--source', type=int, default=0, help='摄像头索引 (默认 0)')
    parser.add_argument('--model', default='yolov8n.pt', help='模型路径')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--frames', type=int, default=100, help='fps_test 测试帧数')
    args = parser.parse_args()

    if args.mode == 'camera_test':
        camera_test(args.source)
    elif args.mode == 'detect':
        detect(args.source, args.model, args.conf)
    elif args.mode == 'fps_test':
        fps_test(args.source, args.model, args.conf, args.frames)
    elif args.mode == 'conf_sweep':
        conf_sweep(args.source, args.model, args.conf)


if __name__ == '__main__':
    main()
