"""
生成 ArUco 标记图片，打印后贴到桌角和小车上。

用法:  python generate_markers.py

输出:
  markers/corner_0.png  ← 桌子左上角
  markers/corner_1.png  ← 桌子右上角
  markers/corner_2.png  ← 桌子右下角
  markers/corner_3.png  ← 桌子左下角
  markers/car.png       ← 贴在小车上
"""

import cv2
import os

OUTPUT_DIR = "markers"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用 4x4_50 字典（标记简单，摄像头容易识别）
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

MARKER_SIZE_PX = 400  # 图片像素大小
BORDER_BITS = 1       # 黑色边框宽度

# ID 分配：0-3 = 桌角，4 = 小车
CORNER_NAMES = ["corner_0_左上", "corner_1_右上", "corner_2_右下", "corner_3_左下"]
CAR_ID = 4

print("生成桌角标记...")
for i, name in enumerate(CORNER_NAMES):
    img = cv2.aruco.generateImageMarker(aruco_dict, i, MARKER_SIZE_PX)
    # 加白边，打印时方便裁剪
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20,
                              cv2.BORDER_CONSTANT, value=255)
    path = os.path.join(OUTPUT_DIR, f"corner_{i}.png")
    cv2.imwrite(path, img)
    print(f"  ✓ ID={i}  {name}  → {path}")

print("生成小车标记...")
img = cv2.aruco.generateImageMarker(aruco_dict, CAR_ID, MARKER_SIZE_PX)
img = cv2.copyMakeBorder(img, 20, 20, 20, 20,
                          cv2.BORDER_CONSTANT, value=255)
path = os.path.join(OUTPUT_DIR, "car.png")
cv2.imwrite(path, img)
print(f"  ✓ ID={CAR_ID}  小车  → {path}")

print(f"\n完成！共生成 5 张标记图，保存在 {OUTPUT_DIR}/ 文件夹")
print("打印说明：")
print("  桌角标记：建议打印成 4~6cm 正方形，贴在桌子四个角")
print("  小车标记：建议打印成 3~4cm 正方形，贴在小车顶部")
