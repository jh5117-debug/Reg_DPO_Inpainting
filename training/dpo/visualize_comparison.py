#!/usr/bin/env python3
"""
五合一对比可视化视频生成工具

将 GT / Mask / ProPainter / Win / Lose 五路帧合成为一个对比视频。
默认 8 FPS 慢速播放，方便人眼观察各方案差异。

布局:
  ┌─────────┬─────────┬─────────┐
  │  GT     │  Mask   │ProPainter│
  ├─────────┴────┬────┴─────────┤
  │  Win (Best)  │ Lose (Worst) │
  └──────────────┴──────────────┘

  上排 3 格等宽，下排 2 格居中（两侧留黑边）。

用法:
    python visualize_comparison.py \
        --gt_dir clip_000/gt_frames \
        --mask_dir clip_000/masks \
        --pp_dir clip_000/pp_frames \
        --win_dir clip_000/win_frames \
        --lose_dir clip_000/lose_frames \
        --output clip_000/comparison.mp4 \
        --fps 8
"""

import argparse
import os
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _read_frames_from_dir(frame_dir: str) -> List[np.ndarray]:
    """从目录读取 RGB 帧列表。"""
    files = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    frames = []
    for fp in files:
        img = cv2.imread(fp)
        if img is not None:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def _read_from_path(path: str) -> List[np.ndarray]:
    """从目录或视频文件读取 RGB 帧。"""
    if os.path.isdir(path):
        return _read_frames_from_dir(path)
    # 视频文件
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _ffmpeg_remux_h264(src_mp4: str, dst_mp4: str) -> bool:
    """Re-encode mp4v → H.264 via ffmpeg."""
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", src_mp4,
             "-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-pix_fmt", "yuv420p", "-an", dst_mp4],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
        return os.path.exists(dst_mp4) and os.path.getsize(dst_mp4) > 0
    except Exception:
        return False


def create_3in1_comparison_video(
    gt_frames: List[np.ndarray],
    mask_frames: List[np.ndarray],
    neg_frames: List[np.ndarray],
    output_path: str,
    fps: int = 8,
    cell_w: int = 320,
    cell_h: int = 240,
    neg_type: str = "neg",
):
    """
    生成三合一对比视频：GT | Mask | Negative。

    Args:
        gt_frames: GT RGB 帧列表
        mask_frames: Mask 帧列表（可为灰度或 RGB）
        neg_frames: 负样本帧列表
        output_path: 输出 mp4 路径
        fps: 播放帧率（默认 8，慢速）
        cell_w: 每个 cell 的宽度
        cell_h: 每个 cell 的高度
        neg_type: 负样本类型（blur/hallucination/flicker），标注在视频上
    """
    n = min(len(gt_frames), len(mask_frames), len(neg_frames))
    if n == 0:
        print("[WARN] 3in1: 无可用帧")
        return

    canvas_w = cell_w * 3
    canvas_h = cell_h
    canvas_w = canvas_w - canvas_w % 2
    canvas_h = canvas_h - canvas_h % 2

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (canvas_w, canvas_h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(cell_w, cell_h) / 500)
    font_thickness = max(1, int(font_scale * 2))

    labels = ["GT", "Mask", f"Neg ({neg_type})"]
    # 负样本用红色边框
    neg_border_color = (0, 0, 200)

    for i in range(n):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        sources = [gt_frames[i], mask_frames[i], neg_frames[i]]

        for col, (src, label) in enumerate(zip(sources, labels)):
            cell = _prepare_cell(src, cell_w, cell_h)
            _draw_label(cell, label, font, font_scale, font_thickness)
            _draw_frame_number(cell, i, n, font, font_scale)
            # 负样本 cell 加红色边框
            if col == 2:
                cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), neg_border_color, 3)
            x0 = col * cell_w
            canvas[0:cell_h, x0:x0 + cell_w] = cell

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()

    if _ffmpeg_remux_h264(tmp_path, output_path):
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, output_path)

    print(f"  [3in1] 保存: {output_path} ({n} frames, {fps} fps)")


def create_5in1_comparison_video(
    gt_frames: List[np.ndarray],
    mask_frames: List[np.ndarray],
    pp_frames: List[np.ndarray],
    win_frames: List[np.ndarray],
    lose_frames: List[np.ndarray],
    output_path: str,
    fps: int = 8,
    cell_w: int = 320,
    cell_h: int = 240,
    win_score: Optional[float] = None,
    lose_score: Optional[float] = None,
):
    """
    生成五合一对比视频。

    Args:
        gt_frames: GT RGB 帧列表
        mask_frames: Mask 帧列表（可为灰度或 RGB）
        pp_frames: ProPainter 结果帧
        win_frames: 最佳候选帧 (Win)
        lose_frames: 最差候选帧 (Lose)
        output_path: 输出 mp4 路径
        fps: 播放帧率（默认 8，慢速）
        cell_w: 每个 cell 的宽度
        cell_h: 每个 cell 的高度
        win_score: Win 的 inpainting_score（可选，标注用）
        lose_score: Lose 的 inpainting_score（可选，标注用）
    """
    n = min(len(gt_frames), len(mask_frames), len(pp_frames),
            len(win_frames), len(lose_frames))
    if n == 0:
        print("[WARN] 5in1: 无可用帧")
        return

    # 画布尺寸: 上排 3 格, 下排 2 格居中
    canvas_w = cell_w * 3
    canvas_h = cell_h * 2
    # 确保偶数
    canvas_w = canvas_w - canvas_w % 2
    canvas_h = canvas_h - canvas_h % 2

    # 下排两格总宽度 = 2 * cell_w, 居中偏移
    bottom_total_w = cell_w * 2
    bottom_offset_x = (canvas_w - bottom_total_w) // 2

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (canvas_w, canvas_h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(cell_w, cell_h) / 500)
    font_thickness = max(1, int(font_scale * 2))

    # 标签
    labels_top = ["GT", "Mask", "ProPainter"]
    win_label = f"Win ({win_score:.3f})" if win_score is not None else "Win (Best)"
    lose_label = f"Lose ({lose_score:.3f})" if lose_score is not None else "Lose (Worst)"

    for i in range(n):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # 准备上排 3 个 cell
        top_sources = [gt_frames[i], mask_frames[i], pp_frames[i]]
        for col, (src, label) in enumerate(zip(top_sources, labels_top)):
            cell = _prepare_cell(src, cell_w, cell_h)
            _draw_label(cell, label, font, font_scale, font_thickness)
            # 帧号
            _draw_frame_number(cell, i, n, font, font_scale)
            x0 = col * cell_w
            canvas[0:cell_h, x0:x0 + cell_w] = cell

        # 准备下排 2 个 cell
        bottom_sources = [win_frames[i], lose_frames[i]]
        bottom_labels = [win_label, lose_label]
        # Win 边框绿色，Lose 边框红色
        border_colors = [(0, 200, 0), (0, 0, 200)]

        for col, (src, label, bcolor) in enumerate(
            zip(bottom_sources, bottom_labels, border_colors)
        ):
            cell = _prepare_cell(src, cell_w, cell_h)
            _draw_label(cell, label, font, font_scale, font_thickness)
            _draw_frame_number(cell, i, n, font, font_scale)
            # 彩色边框（3px）
            cv2.rectangle(cell, (0, 0), (cell_w - 1, cell_h - 1), bcolor, 3)
            x0 = bottom_offset_x + col * cell_w
            canvas[cell_h:cell_h * 2, x0:x0 + cell_w] = cell

        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()

    # 转码 H.264
    if _ffmpeg_remux_h264(tmp_path, output_path):
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, output_path)

    print(f"  [5in1] 保存: {output_path} ({n} frames, {fps} fps)")


def _prepare_cell(src: np.ndarray, w: int, h: int) -> np.ndarray:
    """将源帧 resize 到 cell 尺寸，处理灰度/RGB。"""
    if src.ndim == 2:
        # 灰度 → RGB
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    elif src.shape[2] == 1:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)

    resized = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
    return resized.copy()


def _draw_label(cell, label, font, font_scale, font_thickness):
    """在 cell 左上角绘制半透明背景标签。"""
    ts = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    # 半透明黑色背景
    overlay = cell.copy()
    cv2.rectangle(overlay, (4, 4), (12 + ts[0], 14 + ts[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, cell, 0.4, 0, cell)
    cv2.putText(cell, label, (8, 10 + ts[1]), font, font_scale,
                (255, 255, 255), font_thickness, cv2.LINE_AA)


def _draw_frame_number(cell, frame_idx, total, font, font_scale):
    """在 cell 右下角绘制帧号。"""
    h, w = cell.shape[:2]
    text = f"{frame_idx + 1}/{total}"
    ts = cv2.getTextSize(text, font, font_scale * 0.7, 1)[0]
    x = w - ts[0] - 8
    y = h - 8
    cv2.putText(cell, text, (x, y), font, font_scale * 0.7,
                (200, 200, 200), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────
# 独立运行模式
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="生成五合一对比视频")
    parser.add_argument("--gt_dir", required=True, help="GT 帧目录或视频路径")
    parser.add_argument("--mask_dir", required=True, help="Mask 帧目录或视频路径")
    parser.add_argument("--pp_dir", required=True, help="ProPainter 帧目录或视频路径")
    parser.add_argument("--win_dir", required=True, help="Win 帧目录或视频路径")
    parser.add_argument("--lose_dir", required=True, help="Lose 帧目录或视频路径")
    parser.add_argument("--output", default="comparison_5in1.mp4", help="输出路径")
    parser.add_argument("--fps", type=int, default=8, help="播放帧率")
    parser.add_argument("--cell_w", type=int, default=320, help="Cell 宽度")
    parser.add_argument("--cell_h", type=int, default=240, help="Cell 高度")
    parser.add_argument("--win_score", type=float, default=None, help="Win inpainting_score")
    parser.add_argument("--lose_score", type=float, default=None, help="Lose inpainting_score")
    args = parser.parse_args()

    gt = _read_from_path(args.gt_dir)
    mask = _read_from_path(args.mask_dir)
    pp = _read_from_path(args.pp_dir)
    win = _read_from_path(args.win_dir)
    lose = _read_from_path(args.lose_dir)

    create_5in1_comparison_video(
        gt, mask, pp, win, lose,
        args.output, fps=args.fps,
        cell_w=args.cell_w, cell_h=args.cell_h,
        win_score=args.win_score, lose_score=args.lose_score,
    )


if __name__ == "__main__":
    main()
