"""
GoPro Wide 相机标定工具
========================
步骤：
  1. 打印 calibration_checkerboard.png（A4纸，贴在硬纸板上）
  2. 用 GoPro Wide 模式拍20-30张棋盘格照片，放入 calib_images/ 文件夹
  3. 运行：python calibrate_gopro.py

输出：
  - calib_images/gopro_calib.npz  ← 精确相机内参，供 aruco_detect.py 使用
  - 每张图的角点检测结果（目视确认）
"""

import cv2
import numpy as np
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 棋盘格参数（必须和打印的一致）──────────────────────────────────
# 内角点数：格子数-1，例如 9x6 的棋盘格有 8x5 个内角点
BOARD_W = 9   # 横向内角点数
BOARD_H = 6   # 纵向内角点数
SQUARE_MM = 25  # 每个格子的边长（毫米），打印后用尺子量一下

CALIB_DIR = str(ROOT / "calib_images")
OUTPUT_FILE = str(ROOT / "calib_images/gopro_calib.npz")


def generate_checkerboard():
    """生成棋盘格图片用于打印"""
    squares_x = BOARD_W + 1
    squares_y = BOARD_H + 1
    square_px = 80  # 每格像素大小（打印时按A4缩放）
    w = squares_x * square_px
    h = squares_y * square_px

    img = np.ones((h, w), dtype=np.uint8) * 255
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                x1, y1 = col * square_px, row * square_px
                img[y1:y1+square_px, x1:x1+square_px] = 0

    fname = str(ROOT / "calib_images/calibration_checkerboard.png")
    cv2.imwrite(fname, img)
    print(f"棋盘格已生成: {fname}")
    print(f"  规格: {squares_x}x{squares_y} 格, 内角点 {BOARD_W}x{BOARD_H}")
    print(f"  打印说明: A4纸打印，贴在硬纸板上保持平整")
    print(f"  打印后用尺子量一个格子的实际边长，更新 SQUARE_MM 参数\n")
    return fname


def calibrate():
    os.makedirs(CALIB_DIR, exist_ok=True)

    # 找棋盘格图片
    images = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG"):
        images += glob.glob(os.path.join(CALIB_DIR, ext))
    images = sorted(images)

    if len(images) == 0:
        print(f"未找到图片！请把 GoPro 拍的棋盘格照片放入: {CALIB_DIR}/")
        return

    print(f"找到 {len(images)} 张图片，开始标定...\n")

    # 3D 世界坐标（棋盘格平面，Z=0）
    objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2)
    objp *= SQUARE_MM

    objpoints = []  # 3D 点
    imgpoints = []  # 2D 像素点
    img_size = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    good = 0
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)

        if ret:
            good += 1
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # 画角点预览
            preview = img.copy()
            cv2.drawChessboardCorners(preview, (BOARD_W, BOARD_H), corners2, ret)
            cv2.putText(preview, f"OK #{good}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow("Calibration", cv2.resize(preview, (800, 500)))
            cv2.waitKey(300)
            print(f"  [{i+1:02d}/{len(images)}] ✓ {os.path.basename(fname)}")
        else:
            print(f"  [{i+1:02d}/{len(images)}] ✗ 未检测到角点: {os.path.basename(fname)}")

    cv2.destroyAllWindows()

    if good < 10:
        print(f"\n⚠️  只有 {good} 张图成功检测到角点，建议至少15张。")
        print("请补拍更多角度的照片后重新运行。")
        return

    print(f"\n成功使用 {good} 张图，正在计算标定参数...")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # 计算重投影误差（越小越好，通常 < 1.0 像素）
    total_err = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        total_err += cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total_err / len(objpoints)

    print(f"\n标定完成！")
    print(f"  相机矩阵 K:\n{K}")
    print(f"  畸变系数 dist: {dist.ravel()}")
    print(f"  重投影误差: {mean_err:.4f} px  {'✓ 良好' if mean_err < 1.0 else '⚠️ 偏大，建议补拍'}")

    np.savez(OUTPUT_FILE, K=K, dist=dist, img_size=img_size)
    print(f"\n参数已保存: {OUTPUT_FILE}")
    print("下一步：运行 aruco_detect.py 时会自动加载此文件")


def main():
    # 先生成棋盘格图片
    if not os.path.exists(str(ROOT / "calib_images/calibration_checkerboard.png")):
        generate_checkerboard()
    else:
        print(f"棋盘格图片已存在: {ROOT / 'calib_images/calibration_checkerboard.png'}")

    # 如果 calib_images/ 有图就直接标定
    calib_files = glob.glob(os.path.join(CALIB_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(CALIB_DIR, "*.JPG")) + \
                  glob.glob(os.path.join(CALIB_DIR, "*.png"))
    if calib_files:
        calibrate()
    else:
        print(f"\n请把 GoPro Wide 模式拍的棋盘格照片放入: {CALIB_DIR}/")
        print("拍摄建议：")
        print("  - 20-30张，每张改变棋盘格的角度和位置")
        print("  - 覆盖画面四个角、中间、不同倾斜角度")
        print("  - 棋盘格占画面至少 1/4")
        print("  - 保持棋盘格平整，不要遮挡任何角点")
        print("\n放好照片后重新运行此脚本")


if __name__ == "__main__":
    main()
