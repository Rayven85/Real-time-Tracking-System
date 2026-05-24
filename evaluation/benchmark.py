"""
轻量化指标基准测试
测量各模型的：参数量、GFLOPs、模型大小、推理延迟、FPS
结果保存到 runs/lightweight_results.csv，并与 runs/comparison_results.csv 合并输出

用法：
    python benchmark.py                    # 测所有模型
    python benchmark.py --models yolov8n yolov11n
    python benchmark.py --warmup 20 --runs 100
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "runs/train"

MODELS = {
    "yolov5nu": BASE / "yolov5nu" / "weights" / "best.pt",
    "yolov8n":  BASE / "yolov8n"  / "weights" / "best.pt",
    "yolov9t":  BASE / "yolov9t"  / "weights" / "best.pt",
    "yolov10n": BASE / "yolov10n" / "weights" / "best.pt",
    "yolov11n": BASE / "yolov11n" / "weights" / "best.pt",
}

# 读取已有的精度指标（来自 collect_results.py 生成的 comparison_results.csv）
def load_accuracy_results(csv_path: Path) -> dict:
    results = {}
    if not csv_path.exists():
        return results
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            results[row["model"]] = row
    return results


def get_test_images(img_dir: Path, n: int = 100) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    # 如果不足 n 张则循环复用
    if len(imgs) == 0:
        raise FileNotFoundError(f"测试图片目录为空: {img_dir}")
    if len(imgs) < n:
        imgs = (imgs * ((n // len(imgs)) + 1))[:n]
    return imgs[:n]


def benchmark_model(name: str, weights: Path, img_dir: Path,
                    warmup: int, runs: int, device: str) -> dict:
    print(f"\n  [{name}] 加载模型 ...")
    model = YOLO(str(weights))

    # ── 模型大小（文件体积）──
    size_mb = os.path.getsize(weights) / (1024 ** 2)

    # ── 参数量 & GFLOPs（通过 ultralytics LOGGER handler 捕获输出）──
    import io, re, logging
    from ultralytics.utils import LOGGER as _UL_LOGGER
    buf = io.StringIO()
    _handler = logging.StreamHandler(buf)
    _handler.setLevel(logging.DEBUG)
    _UL_LOGGER.addHandler(_handler)
    try:
        model.info(verbose=True)
    except Exception:
        pass
    _UL_LOGGER.removeHandler(_handler)
    info_str = buf.getvalue()

    n_params, gflops = None, None
    # 匹配 "Model summary: 130 layers, 3,011,628 parameters, 0 gradients, 8.2 GFLOPs"
    m = re.search(r"([\d,]+)\s+parameters.*?([\d.]+)\s+GFLOPs", info_str)
    if m:
        n_params = int(m.group(1).replace(",", ""))
        gflops   = float(m.group(2))
    else:
        n_params = sum(p.numel() for p in model.model.parameters())

    params_m = n_params / 1e6  # 换算为 M

    # ── 推理延迟（在真实图片上实测）──
    imgs = get_test_images(img_dir, warmup + runs)

    print(f"  [{name}] warmup {warmup} 帧 ...", end="", flush=True)
    for img in imgs[:warmup]:
        model.predict(str(img), device=device, verbose=False)
    print(" done")

    print(f"  [{name}] 计时 {runs} 帧 ...", end="", flush=True)
    latencies = []
    for img in imgs[warmup:warmup + runs]:
        t0 = time.perf_counter()
        model.predict(str(img), device=device, verbose=False)
        latencies.append(time.perf_counter() - t0)
    print(" done")

    avg_ms  = (sum(latencies) / len(latencies)) * 1000
    min_ms  = min(latencies) * 1000
    max_ms  = max(latencies) * 1000
    fps     = 1000.0 / avg_ms

    return {
        "model":      name,
        "params_M":   round(params_m, 2),
        "GFLOPs":     round(gflops, 2) if gflops is not None else "N/A",
        "size_MB":    round(size_mb, 2),
        "latency_ms": round(avg_ms, 2),
        "latency_min_ms": round(min_ms, 2),
        "latency_max_ms": round(max_ms, 2),
        "FPS":        round(fps, 1),
    }


def print_table(bench_results: list, acc_map: dict) -> None:
    header = f"{'Model':<12} {'Params(M)':>10} {'GFLOPs':>8} {'Size(MB)':>9} " \
             f"{'Lat(ms)':>9} {'FPS':>7} {'mAP50':>8} {'mAP50-95':>10}"
    sep = "─" * len(header)
    print(f"\n{sep}")
    print("  轻量化 + 精度综合对比")
    print(sep)
    print(" ", header)
    print(" ", "─" * (len(header) - 2))

    for r in bench_results:
        acc = acc_map.get(r["model"], {})
        map50    = float(acc.get("mAP50",    0)) if acc.get("mAP50")    else 0
        map5095  = float(acc.get("mAP50-95", 0)) if acc.get("mAP50-95") else 0
        print(f"  {r['model']:<12} {r['params_M']:>10} {str(r['GFLOPs']):>8} "
              f"{r['size_MB']:>9} {r['latency_ms']:>9} {r['FPS']:>7} "
              f"{map50:>8.4f} {map5095:>10.4f}")
    print(f"{sep}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",  nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--img_dir", default=str(ROOT / "datasets/test/images"))
    parser.add_argument("--warmup",  type=int, default=10, help="预热帧数")
    parser.add_argument("--runs",    type=int, default=100, help="计时帧数")
    parser.add_argument("--device",  default="cpu", help="推理设备（建议 cpu 保证可比性）")
    opt = parser.parse_args()

    img_dir = Path(opt.img_dir)
    if not img_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {img_dir}")

    acc_map = load_accuracy_results(ROOT / "runs/comparison_results.csv")

    bench_results = []
    for name in opt.models:
        weights = MODELS.get(name)
        if weights is None:
            print(f"[SKIP] {name} — 未在 MODELS 中注册")
            continue
        if not weights.exists():
            print(f"[SKIP] {name} — 权重文件不存在: {weights}")
            continue

        r = benchmark_model(name, weights, img_dir, opt.warmup, opt.runs, opt.device)
        bench_results.append(r)

    if not bench_results:
        print("没有可测试的模型。")
        return

    # ── 保存轻量化结果 ──
    out_dir = Path("runs")
    out_dir.mkdir(exist_ok=True)
    bench_csv = out_dir / "lightweight_results.csv"
    with open(bench_csv, "w", newline="") as f:
        fields = ["model", "params_M", "GFLOPs", "size_MB",
                  "latency_ms", "latency_min_ms", "latency_max_ms", "FPS"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bench_results)
    print(f"\n轻量化指标已保存到 {bench_csv}")

    # ── 合并保存到 comparison_results_full.csv ──
    if acc_map:
        full_csv = out_dir / "comparison_results_full.csv"
        fields_full = ["model", "mAP50", "mAP50-95", "precision", "recall",
                       "params_M", "GFLOPs", "size_MB", "latency_ms", "FPS"]
        with open(full_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields_full)
            writer.writeheader()
            for r in bench_results:
                acc = acc_map.get(r["model"], {})
                merged = {
                    "model":      r["model"],
                    "mAP50":      acc.get("mAP50", ""),
                    "mAP50-95":   acc.get("mAP50-95", ""),
                    "precision":  acc.get("precision", ""),
                    "recall":     acc.get("recall", ""),
                    "params_M":   r["params_M"],
                    "GFLOPs":     r["GFLOPs"],
                    "size_MB":    r["size_MB"],
                    "latency_ms": r["latency_ms"],
                    "FPS":        r["FPS"],
                }
                writer.writerow(merged)
        print(f"完整对比（精度+效率）已保存到 {full_csv}")

    print_table(bench_results, acc_map)


if __name__ == "__main__":
    main()
