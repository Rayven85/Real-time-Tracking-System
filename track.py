"""
实时交通标志跟踪 — YOLOv11n + ByteTrack
用法：
    python track.py                          # 默认摄像头 0
    python track.py --source 1               # 外接摄像头索引 1
    python track.py --source video.mp4       # 视频文件
    python track.py --conf 0.4               # 调整置信度阈值
    python track.py --save                   # 保存输出视频
"""

import argparse
import time
import cv2
from ultralytics import YOLO

WEIGHTS = "runs/detect/runs/train/yolov11n/weights/best.pt"

CLASS_COLORS = [
    (0, 0, 255),    # danger      → 红
    (0, 200, 0),    # mandatory   → 绿
    (255, 150, 0),  # other       → 蓝橙
    (0, 165, 255),  # prohibitory → 橙
]


def run(source, conf, save):
    model = YOLO(WEIGHTS)
    class_names = model.names  # {0: 'danger', 1: 'mandatory', ...}

    # 打开摄像头 / 视频
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[track] 分辨率: {w}x{h}  |  模型: {WEIGHTS}  |  conf: {conf}")
    print("[track] 按 q 退出")

    writer = None
    if save:
        out_path = "runs/track_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
        print(f"[track] 输出将保存到 {out_path}")

    prev_time = time.perf_counter()
    track_history = {}   # id → 轨迹点列表，用于画运动轨迹

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── 跟踪推理 ──────────────────────────────────────────────────────────
        results = model.track(
            frame,
            conf=conf,
            tracker="bytetrack.yaml",   # ultralytics 内置 ByteTrack
            persist=True,               # 跨帧保持 ID
            verbose=False,
            device="cpu",
        )

        # ── 画框 + 轨迹 ───────────────────────────────────────────────────────
        boxes = results[0].boxes
        active_ids = set()

        if boxes is not None and boxes.id is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id  = int(box.cls[0])
                conf_v  = float(box.conf[0])
                track_id = int(box.id[0])
                color   = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                label   = f"ID{track_id} {class_names[cls_id]} {conf_v:.2f}"

                # 画检测框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                # 记录轨迹中心点
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:   # 只保留最近 30 帧
                    track_history[track_id].pop(0)
                active_ids.add(track_id)

                # 画轨迹线
                pts = track_history[track_id]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)   # 越新越亮
                    c = tuple(int(ch * alpha) for ch in color)
                    cv2.line(frame, pts[i - 1], pts[i], c, 2)

        # 清理消失目标的轨迹，避免内存无限增长
        for tid in list(track_history.keys()):
            if tid not in active_ids:
                del track_history[tid]

        # ── HUD 信息 ──────────────────────────────────────────────────────────
        now = time.perf_counter()
        fps = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        n_tracks = len(active_ids)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking: {n_tracks}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("YOLOv11n + ByteTrack (press q to quit)", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[track] 退出")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",  help="摄像头索引或视频路径")
    parser.add_argument("--conf",   type=float, default=0.4)
    parser.add_argument("--save",   action="store_true", help="保存输出视频")
    opt = parser.parse_args()
    run(opt.source, opt.conf, opt.save)


if __name__ == "__main__":
    main()
