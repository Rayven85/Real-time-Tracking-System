"""
可视化测试结果：真实框 vs 各模型预测框
输出：
  1. 每个模型单独的预测图（真实框 + 预测框）
  2. 所有模型对比图（同一张图上用不同颜色画各模型预测框 + 真实框）

用法：
    python visualize.py                     # 随机取 10 张测试图
    python visualize.py --n 20              # 取 20 张
    python visualize.py --n 5 --seed 42     # 固定随机种子，结果可复现
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

# ── 配置 ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['danger', 'mandatory', 'other', 'prohibitory']

MODELS = {
    "yolov5nu": str(ROOT / "runs/detect/runs/train/yolov5nu/weights/best.pt"),
    "yolov8n":  str(ROOT / "runs/detect/runs/train/yolov8n/weights/best.pt"),
    "yolov9t":  str(ROOT / "runs/detect/runs/train/yolov9t/weights/best.pt"),
    "yolov10n": str(ROOT / "runs/detect/runs/train/yolov10n/weights/best.pt"),
    "yolov11n": str(ROOT / "runs/detect/runs/train/yolov11n/weights/best.pt"),
}

# 对比图中每个模型的颜色（BGR）
MODEL_COLORS = {
    "yolov5nu": (255, 100,   0),   # 蓝
    "yolov8n":  (  0, 200,   0),   # 绿
    "yolov9t":  (  0, 100, 255),   # 橙
    "yolov10n": (200,   0, 200),   # 紫
    "yolov11n": (  0, 200, 255),   # 黄
}
GT_COLOR     = (0, 0, 255)         # 真实框：红色
CONF_THRESH  = 0.3

IMG_DIR   = ROOT / "datasets/test/images"
LABEL_DIR = ROOT / "datasets/test/labels"
OUT_DIR   = ROOT / "runs/visualization"


# ── 工具函数 ──────────────────────────────────────────────────────────────────
def yolo2xyxy(cx, cy, w, h, W, H):
    """YOLO 归一化坐标 → 像素绝对坐标"""
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)
    return x1, y1, x2, y2


def load_gt(label_path: Path, W: int, H: int) -> list:
    """读取 YOLO 格式真实框，返回 [(cls_id, x1,y1,x2,y2), ...]"""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            boxes.append((cls_id, *yolo2xyxy(cx, cy, bw, bh, W, H)))
    return boxes


def draw_box(img, x1, y1, x2, y2, color, label, thickness=2):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    # 文字背景
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_legend(img, items: list, start_y=20):
    """在图片右上角画图例，items = [(name, color), ...]"""
    x = img.shape[1] - 160
    for i, (name, color) in enumerate(items):
        y = start_y + i * 22
        cv2.rectangle(img, (x, y - 12), (x + 18, y + 2), color, -1)
        cv2.putText(img, name, (x + 24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ── 单模型可视化 ──────────────────────────────────────────────────────────────
def visualize_single(model_name: str, model: YOLO,
                     img_paths: list, out_dir: Path):
    """为每张图生成：真实框(红) + 该模型预测框(绿)"""
    save_dir = out_dir / "per_model" / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # 画真实框
        gt_boxes = load_gt(LABEL_DIR / (img_path.stem + ".txt"), W, H)
        for cls_id, x1, y1, x2, y2 in gt_boxes:
            draw_box(img, x1, y1, x2, y2, GT_COLOR,
                     f"GT:{CLASS_NAMES[cls_id]}", thickness=2)

        # 画预测框
        results = model.predict(str(img_path), conf=CONF_THRESH,
                                verbose=False, device="cpu")
        pred_color = (0, 220, 0)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            draw_box(img, x1, y1, x2, y2, pred_color,
                     f"P:{CLASS_NAMES[cls_id]} {conf:.2f}")

        # 图例
        draw_legend(img, [("GT (ground truth)", GT_COLOR),
                          (f"{model_name} pred", pred_color)])

        cv2.imwrite(str(save_dir / img_path.name), img)

    print(f"  [{model_name}] 单模型图已保存到 {save_dir}")


# ── 多模型对比图 ──────────────────────────────────────────────────────────────
def visualize_comparison(models: dict, img_paths: list, out_dir: Path):
    """所有模型预测框 + 真实框画在同一张图上"""
    save_dir = out_dir / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 预加载所有模型
    loaded = {name: YOLO(w) for name, w in models.items()}

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]

        # 画真实框（最粗，画在最底层）
        gt_boxes = load_gt(LABEL_DIR / (img_path.stem + ".txt"), W, H)
        for cls_id, x1, y1, x2, y2 in gt_boxes:
            draw_box(img, x1, y1, x2, y2, GT_COLOR,
                     f"GT:{CLASS_NAMES[cls_id]}", thickness=3)

        # 画各模型预测框
        for name, model in loaded.items():
            color = MODEL_COLORS[name]
            results = model.predict(str(img_path), conf=CONF_THRESH,
                                    verbose=False, device="cpu")
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                draw_box(img, x1, y1, x2, y2, color,
                         f"{name}:{CLASS_NAMES[cls_id]} {conf:.2f}")

        # 图例
        legend_items = [("GT (ground truth)", GT_COLOR)] + \
                       [(name, MODEL_COLORS[name]) for name in models]
        draw_legend(img, legend_items)

        cv2.imwrite(str(save_dir / img_path.name), img)

    print(f"  [comparison] 对比图已保存到 {save_dir}")


# ── 拼接对比图（grid）─────────────────────────────────────────────────────────
def make_grid(models: dict, img_paths: list, out_dir: Path):
    """
    每张测试图生成一个 grid 图：
    左上：原图+GT   其余格子：各模型预测
    布局：2行3列（1张GT + 5张模型 = 6格）
    """
    save_dir = out_dir / "grid"
    save_dir.mkdir(parents=True, exist_ok=True)

    loaded = {name: YOLO(w) for name, w in models.items()}
    CELL_W, CELL_H = 640, 480

    for img_path in img_paths:
        orig = cv2.imread(str(img_path))
        H, W = orig.shape[:2]

        panels = []

        # 面板0：原图 + 真实框
        gt_panel = orig.copy()
        gt_boxes = load_gt(LABEL_DIR / (img_path.stem + ".txt"), W, H)
        for cls_id, x1, y1, x2, y2 in gt_boxes:
            draw_box(gt_panel, x1, y1, x2, y2, GT_COLOR,
                     f"GT:{CLASS_NAMES[cls_id]}", thickness=2)
        cv2.putText(gt_panel, "Ground Truth", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, GT_COLOR, 2)
        panels.append(cv2.resize(gt_panel, (CELL_W, CELL_H)))

        # 面板1-5：各模型
        for name, model in loaded.items():
            panel = orig.copy()
            color = MODEL_COLORS[name]
            results = model.predict(str(img_path), conf=CONF_THRESH,
                                    verbose=False, device="cpu")
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                draw_box(panel, x1, y1, x2, y2, color,
                         f"{CLASS_NAMES[cls_id]} {conf:.2f}")
            cv2.putText(panel, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            panels.append(cv2.resize(panel, (CELL_W, CELL_H)))

        # 拼成 2×3 grid
        row1 = np.hstack(panels[:3])
        row2 = np.hstack(panels[3:])
        grid = np.vstack([row1, row2])

        cv2.imwrite(str(save_dir / img_path.name), grid)

    print(f"  [grid] 拼接对比图已保存到 {save_dir}")


# ── 主函数 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=10, help="取几张测试图")
    parser.add_argument("--seed", type=int, default=0,  help="随机种子")
    parser.add_argument("--mode", default="all",
                        choices=["all", "single", "comparison", "grid"],
                        help="输出模式")
    opt = parser.parse_args()

    # 随机选 n 张测试图
    all_imgs = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    random.seed(opt.seed)
    imgs = random.sample(all_imgs, min(opt.n, len(all_imgs)))
    print(f"选取 {len(imgs)} 张测试图，seed={opt.seed}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if opt.mode in ("all", "single"):
        print("\n── 生成单模型图 ──")
        for name, weights in MODELS.items():
            model = YOLO(weights)
            visualize_single(name, model, imgs, OUT_DIR)

    if opt.mode in ("all", "comparison"):
        print("\n── 生成多模型对比图 ──")
        visualize_comparison(MODELS, imgs, OUT_DIR)

    if opt.mode in ("all", "grid"):
        print("\n── 生成 Grid 拼接图 ──")
        make_grid(MODELS, imgs, OUT_DIR)

    print(f"\n全部完成，结果在 {OUT_DIR}/")
    print("  per_model/   — 每个模型单独的预测图")
    print("  comparison/  — 所有模型叠加在同一张图上")
    print("  grid/        — 2×3 拼接对比图（推荐用于汇报）")


if __name__ == "__main__":
    main()
