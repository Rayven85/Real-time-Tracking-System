"""
GoPro FOV 畸变对比工具
========================
用途：对比 GoPro 不同 FOV 模式（Linear / Wide / SuperView）的畸变程度，
      帮你选出覆盖范围与图像质量的最佳平衡点。

用法：
  1. 在 GoPro 上用不同 FOV 模式分别拍赛道俯视照片
  2. 把照片放进对应文件夹：
       gopro_samples/linear/
       gopro_samples/wide/
       gopro_samples/superview/
  3. 运行：python gopro_distortion.py

输出：
  - 每张图叠加网格线（看直线是否弯曲）
  - 检测图像边缘的直线度评分
  - 三种模式并排对比图 → 保存为 fov_comparison.jpg
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_DIRS = {
    "Linear":    str(ROOT / "gopro_samples/linear"),
    "Wide":      str(ROOT / "gopro_samples/wide"),
    "SuperView": str(ROOT / "gopro_samples/superview"),
}
OUTPUT_FILE = str(ROOT / "fov_comparison.jpg")
GRID_STEP = 80   # 网格间距（像素），越小越密


def load_first_image(folder):
    """从文件夹中加载第一张图片（支持 jpg/png）"""
    for ext in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png"):
        files = glob.glob(os.path.join(folder, ext))
        if files:
            img = cv2.imread(files[0])
            if img is not None:
                print(f"  加载: {files[0]}")
                return img
    return None


def draw_grid(img, step=GRID_STEP, color=(0, 255, 0), thickness=1):
    """在图像上叠加等间距网格线（直线弯曲 = 有畸变）"""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), color, thickness)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), color, thickness)
    return out


def distortion_score(img):
    """
    用 Hough 直线检测估算畸变程度。
    思路：真正的直线（桌子边缘）在畸变图中会变弯，
         Hough 检测到的短线段越多、角度越散乱 → 畸变越大。
    返回：(检测到的线段数, 角度标准差)，数值越大表示畸变越严重。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=80, minLineLength=60, maxLineGap=15)
    if lines is None:
        return 0, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        angles.append(angle)

    angles = np.array(angles)
    # 只看接近水平/垂直的线（赛道/桌子边缘）
    hv = angles[(angles < 20) | (angles > 160) |
                 ((angles > 70) & (angles < 110))]
    std = float(np.std(hv)) if len(hv) > 1 else 0.0
    return len(lines), round(std, 2)


def add_label(img, mode_name, n_lines, angle_std, target_h=500):
    """缩放图片并在底部添加文字说明"""
    h, w = img.shape[:2]
    scale = target_h / h
    resized = cv2.resize(img, (int(w * scale), target_h))

    label_h = 70
    canvas = np.zeros((target_h + label_h, resized.shape[1], 3), dtype=np.uint8)
    canvas[:target_h] = resized

    cv2.putText(canvas, mode_name, (10, target_h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(canvas, f"Lines: {n_lines}  AngleStd: {angle_std}",
                (10, target_h + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return canvas


def main():
    # 建立样本文件夹（首次运行提示用户放图）
    for folder in SAMPLE_DIRS.values():
        os.makedirs(folder, exist_ok=True)

    panels = []
    any_found = False

    for mode_name, folder in SAMPLE_DIRS.items():
        print(f"\n[{mode_name}]")
        img = load_first_image(folder)

        if img is None:
            print(f"  ⚠️  未找到图片，请把 {mode_name} 模式拍的照片放入: {folder}/")
            # 占位灰图
            placeholder = np.full((500, 700, 3), 40, dtype=np.uint8)
            cv2.putText(placeholder, f"{mode_name}: No image",
                        (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            panels.append(placeholder)
            continue

        any_found = True
        gridded = draw_grid(img)
        n_lines, angle_std = distortion_score(img)
        print(f"  检测到线段数: {n_lines}  角度标准差: {angle_std}")
        panel = add_label(gridded, mode_name, n_lines, angle_std)
        panels.append(panel)

        # 单独保存带网格的图
        out_name = f"fov_{mode_name.lower()}_grid.jpg"
        cv2.imwrite(out_name, gridded)
        print(f"  网格图保存: {out_name}")

    # 三图并排
    if any_found:
        # 统一高度后拼接
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            dh = max_h - p.shape[0]
            padded.append(cv2.copyMakeBorder(p, 0, dh, 0, 0,
                                              cv2.BORDER_CONSTANT, value=0))
        comparison = np.hstack(padded)
        cv2.imwrite(OUTPUT_FILE, comparison)
        print(f"\n对比图已保存: {OUTPUT_FILE}")

        cv2.imshow("FOV Comparison (press any key to close)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n请先把三种 FOV 模式的 GoPro 照片放入对应文件夹，再运行此脚本。")
        print("文件夹结构：")
        for mode, folder in SAMPLE_DIRS.items():
            print(f"  {folder}/  ← {mode} 模式照片")


if __name__ == "__main__":
    main()
