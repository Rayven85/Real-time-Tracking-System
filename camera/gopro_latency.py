"""
GoPro Frame Rate & Latency Analyser
=====================================
Measures GoPro USB Webcam real-time streaming performance:
  - Actual FPS (mean + std dev)
  - Frame interval jitter (dropped frame detection)
  - cap.read() blocking time vs total processing time

Usage:
  python gopro_latency.py --source 0 --duration 30
"""

import cv2
import numpy as np
import time
import argparse

PLOT_W, PLOT_H = 700, 300   # size of the FPS history chart


def open_camera(source):
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera [{source}]: {w}x{h}")
    return cap


def draw_fps_plot(intervals_ms, target_ms):
    """Draw a scrolling bar chart of frame intervals (ms). Red line = target."""
    canvas = np.zeros((PLOT_H, PLOT_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    if not intervals_ms:
        return canvas

    n = min(len(intervals_ms), PLOT_W)
    recent = intervals_ms[-n:]
    max_ms = max(max(recent), target_ms * 2, 1)

    bar_w = max(1, PLOT_W // n)
    for i, ms in enumerate(recent):
        bar_h = int((ms / max_ms) * (PLOT_H - 40))
        x = i * bar_w
        color = (0, 200, 0) if ms < target_ms * 1.5 else (0, 50, 255)
        cv2.rectangle(canvas, (x, PLOT_H - 20 - bar_h), (x + bar_w - 1, PLOT_H - 20), color, -1)

    # Target line
    target_y = PLOT_H - 20 - int((target_ms / max_ms) * (PLOT_H - 40))
    cv2.line(canvas, (0, target_y), (PLOT_W, target_y), (0, 180, 255), 1)
    cv2.putText(canvas, f"Target {target_ms:.1f}ms", (5, target_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)

    # Stats
    mean_ms = float(np.mean(recent))
    std_ms  = float(np.std(recent))
    max_ms2 = float(np.max(recent))
    cv2.putText(canvas, f"mean={mean_ms:.1f}ms  std={std_ms:.1f}ms  max={max_ms2:.1f}ms",
                (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",   type=int, default=0,  help="Camera index")
    parser.add_argument("--duration", type=int, default=30, help="Test duration (seconds)")
    args = parser.parse_args()

    cap = open_camera(args.source)
    if cap is None:
        print(f"Cannot open camera {args.source}")
        return

    declared_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_ms = 1000.0 / declared_fps
    print(f"Declared FPS: {declared_fps:.1f}  (target interval: {target_ms:.1f} ms)")
    print(f"Running for {args.duration}s — press 'q' to stop early\n")

    intervals_ms = []       # frame-to-frame intervals
    read_times_ms = []      # cap.read() blocking duration
    t_last = time.perf_counter()
    t_end  = t_last + args.duration

    while time.perf_counter() < t_end:
        t_before = time.perf_counter()
        ret, frame = cap.read()
        t_after  = time.perf_counter()

        if not ret:
            break

        read_ms = (t_after - t_before) * 1000
        interval_ms = (t_after - t_last) * 1000
        t_last = t_after

        read_times_ms.append(read_ms)
        if len(intervals_ms) > 0 or interval_ms < 500:  # skip first huge gap
            intervals_ms.append(interval_ms)

        elapsed = t_after - (t_end - args.duration)
        remaining = t_end - t_after

        # Build display
        plot = draw_fps_plot(intervals_ms, target_ms)
        info = np.zeros((120, PLOT_W, 3), dtype=np.uint8)
        inst_fps = 1000.0 / interval_ms if interval_ms > 0 else 0
        cv2.putText(info, f"Instant FPS: {inst_fps:.1f}    read() time: {read_ms:.1f} ms",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(info, f"Elapsed: {elapsed:.1f}s    Remaining: {remaining:.1f}s",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        if intervals_ms:
            mean_fps = 1000.0 / np.mean(intervals_ms)
            dropped = sum(1 for x in intervals_ms if x > target_ms * 2)
            cv2.putText(info, f"Mean FPS: {mean_fps:.2f}    Dropped frames (>2x interval): {dropped}",
                        (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(info, "Press 'q' to stop early", (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        display = np.vstack([plot, info])
        cv2.imshow("GoPro Latency Analysis", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # ── Final report ─────────────────────────────────────────────────
    if not intervals_ms:
        print("No data collected.")
        return

    arr = np.array(intervals_ms)
    read_arr = np.array(read_times_ms)
    mean_fps  = 1000.0 / arr.mean()
    std_fps   = 1000.0 / arr.std() if arr.std() > 0 else float('inf')
    dropped   = int(np.sum(arr > target_ms * 2))

    print("\n" + "=" * 50)
    print("  GoPro Frame Rate & Latency Report")
    print("=" * 50)
    print(f"  Frames captured    : {len(arr)}")
    print(f"  Declared FPS       : {declared_fps:.1f}")
    print(f"  Actual mean FPS    : {mean_fps:.2f}")
    print(f"  Frame interval     : mean={arr.mean():.1f}ms  std={arr.std():.1f}ms  max={arr.max():.1f}ms")
    print(f"  Dropped frames     : {dropped}  ({100*dropped/len(arr):.1f}%)")
    print(f"  cap.read() latency : mean={read_arr.mean():.1f}ms  max={read_arr.max():.1f}ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
