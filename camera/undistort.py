"""
鱼眼畸变矫正工具（无需标定，手动调参）
============================================
用法:
  python undistort.py --mode wide
  python undistort.py --mode superview
  python undistort.py --mode rpi

操作:
  拖动滑条调整矫正强度
  's' 保存矫正后的图片
  'r' 重置参数
  'q' 退出
"""

import cv2
import numpy as np
import glob
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── GoPro Hero 各焦段的经验初始参数 ──────────────────────────────
# k1: 桶形畸变主系数（负值=桶形，正值=枕形）
# k2: 高阶修正
# 这些是 GoPro Wide 的典型估算值，通过滑条微调
PRESETS = {
    "wide":      {"k1": -25, "k2": 5,  "fx_scale": 90, "fisheye": False},
    "superview": {"k1": -35, "k2": 10, "fx_scale": 85, "fisheye": False},
    "rpi":       {"k1": 50,  "k2": 50, "fx_scale": 70, "fisheye": True},
}

WIN = "Undistort (drag sliders)"
# 滑条范围（内部存 0~100，映射到实际参数）
SLIDER_MAX = 100


def load_image(mode):
    folder = str(ROOT / "RPi_samples") if mode == "rpi" else str(ROOT / f"gopro_samples/{mode}")
    for ext in ("*.JPG", "*.jpg", "*.jpeg", "*.png", "*.PNG"):
        files = sorted(glob.glob(os.path.join(folder, ext)))
        if files:
            img = cv2.imread(files[0])
            if img is not None:
                print(f"加载: {files[0]}")
                return img
    print(f"❌ 未找到图片: {folder}/")
    return None


def apply_fisheye_undistort(img, k1_pct, k2_pct, fx_scale_pct):
    """
    RPi 鱼眼专用矫正，使用 cv2.fisheye 模块。
    k1_pct / k2_pct: 0~100 → 实际 k1/k2: 0.0 ~ 2.0
    """
    h, w = img.shape[:2]
    k1 = k1_pct * 0.02        # 0→0.0, 100→2.0
    k2 = k2_pct * 0.02
    D = np.array([[k1], [k2], [0.0], [0.0]], dtype=np.float64)
    fx = w * (0.3 + fx_scale_pct * 0.007)
    K = np.array([[fx, 0, w / 2],
                  [0, fx, h / 2],
                  [0,  0,     1]], dtype=np.float64)
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=0.5)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    return undistorted


def build_camera_matrix(h, w, fx_scale_pct):
    """
    fx_scale_pct: 0~100，映射到焦距比例 0.5~1.5 * 图像宽
    """
    fx = w * (0.5 + fx_scale_pct / 100.0)
    fy = fx
    cx, cy = w / 2, h / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def apply_undistort(img, k1_pct, k2_pct, fx_scale_pct):
    """
    k1_pct: 0~100 → 实际 k1: -0.6 ~ 0.0（负=矫正桶形）
    k2_pct: 0~100 → 实际 k2: -0.1 ~ 0.1
    """
    h, w = img.shape[:2]
    k1 = -0.6 + k1_pct * 0.006       # 0→-0.6, 100→0.0
    k2 = -0.1 + k2_pct * 0.002       # 0→-0.1, 100→0.1
    dist = np.array([k1, k2, 0, 0, 0], dtype=np.float64)
    K = build_camera_matrix(h, w, fx_scale_pct)
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.5)
    undistorted = cv2.undistort(img, K, dist, None, K_new)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y+rh, x:x+rw]
    return undistorted


def draw_grid(img, step=60):
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (0, 255, 0), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 255, 0), 1)
    return out


def make_compare(orig, fixed, target_w=1400):
    """左=原图, 右=矫正后，并排，宽度固定"""
    h_o, w_o = orig.shape[:2]
    h_f, w_f = fixed.shape[:2]
    target_h = int(h_o * (target_w // 2) / w_o)
    left  = cv2.resize(draw_grid(orig),  (target_w // 2, target_h))
    right = cv2.resize(draw_grid(fixed), (target_w // 2, target_h))

    # 补齐高度
    max_h = max(left.shape[0], right.shape[0])
    def pad(im):
        dh = max_h - im.shape[0]
        return cv2.copyMakeBorder(im, 0, dh, 0, 0, cv2.BORDER_CONSTANT)

    out = np.hstack([pad(left), pad(right)])
    cv2.putText(out, "ORIGINAL", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2)
    cv2.putText(out, "UNDISTORTED", (target_w // 2 + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["wide", "superview", "rpi"], default="wide")
    args = parser.parse_args()

    img = load_image(args.mode)
    if img is None:
        return

    preset = PRESETS[args.mode]
    use_fisheye = preset["fisheye"]
    if use_fisheye:
        k1_init = preset["k1"]
        k2_init = preset["k2"]
    else:
        k1_init = int((preset["k1"] / 100.0 + 0.6) / 0.006)
        k2_init = int((preset["k2"] / 100.0 + 0.1) / 0.002)
    fx_init = preset["fx_scale"]

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1400, 600)
    cv2.createTrackbar("K1 (桶形矫正)", WIN, k1_init, SLIDER_MAX, lambda x: None)
    cv2.createTrackbar("K2 (高阶修正)", WIN, k2_init, SLIDER_MAX, lambda x: None)
    cv2.createTrackbar("焦距缩放",      WIN, fx_init,  SLIDER_MAX, lambda x: None)

    print(f"\n模式: {args.mode.upper()}")
    print("拖动滑条观察矫正效果，直线越直越好")
    print("'s'=保存  'r'=重置  'q'=退出\n")

    shot = 0
    while True:
        k1_v  = cv2.getTrackbarPos("K1 (桶形矫正)", WIN)
        k2_v  = cv2.getTrackbarPos("K2 (高阶修正)", WIN)
        fx_v  = cv2.getTrackbarPos("焦距缩放",      WIN)

        if use_fisheye:
            undistorted = apply_fisheye_undistort(img, k1_v, k2_v, fx_v)
            k1_real = k1_v * 0.02
            k2_real = k2_v * 0.02
        else:
            undistorted = apply_undistort(img, k1_v, k2_v, fx_v)
            k1_real = -0.6 + k1_v * 0.006
            k2_real = -0.1 + k2_v * 0.002
        display = make_compare(img, undistorted)

        param_text = f"k1={k1_real:.3f}  k2={k2_real:.3f}  fx_scale={fx_v}%"
        # 黑色描边 + 白色文字，确保在任何背景下可读
        tx, ty = 10, display.shape[0] - 15
        cv2.putText(display, param_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(display, param_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # 同时打印到终端
        print(f"\r{param_text}", end="", flush=True)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            cv2.setTrackbarPos("K1 (桶形矫正)", WIN, k1_init)
            cv2.setTrackbarPos("K2 (高阶修正)", WIN, k2_init)
            cv2.setTrackbarPos("焦距缩放",      WIN, fx_init)
        elif key == ord('s'):
            shot += 1
            fname = f"undistorted_{args.mode}_{shot:02d}.jpg"
            cv2.imwrite(fname, undistorted)
            print(f"保存: {fname}  (k1={k1_real:.3f}, k2={k2_real:.3f}, fx={fx_v}%)")
        elif key == ord('n') and args.mode == "rpi":
            # RPi 模式下按 'n' 切换下一张图
            pass  # 扩展预留

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
