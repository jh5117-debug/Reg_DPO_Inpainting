#!/usr/bin/env python
# coding=utf-8
"""
validation_metrics.py  ─  轻量级验证指标 (PSNR / SSIM)

从 inference/compare_all.py + inference/metrics.py 精简而来。
仅保留纯 CPU/numpy 计算的 PSNR 和 SSIM，适用于训练中 validation step 的定量指标输出。
不加载任何额外模型 (LPIPS/VBench/RAFT/I3D 等)。

Usage:
    from validation_metrics import compute_validation_metrics
    results = compute_validation_metrics(pred_frames, gt_frames)
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def calc_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算单帧 PSNR (Peak Signal-to-Noise Ratio)。"""
    return peak_signal_noise_ratio(img1, img2, data_range=255)


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算单帧 SSIM (Structural Similarity Index)。"""
    return structural_similarity(
        img1, img2, multichannel=True, channel_axis=-1, data_range=255
    )


def compute_validation_metrics(
    pred_frames: list, gt_frames: list, video_name: str = "unknown"
) -> dict:
    """
    对一组预测帧与 GT 帧逐帧计算 PSNR / SSIM 并取平均。

    Args:
        pred_frames: list of np.ndarray (H, W, 3), uint8 range [0, 255]
        gt_frames:   list of np.ndarray (H, W, 3), uint8 range [0, 255]
        video_name:  视频名称 (用于日志)

    Returns:
        dict: {"video_name": str, "psnr": float, "ssim": float, "num_frames": int}
    """
    assert len(pred_frames) == len(gt_frames), (
        f"Frame count mismatch: pred={len(pred_frames)}, gt={len(gt_frames)}"
    )

    psnr_list = []
    ssim_list = []
    for pred, gt in zip(pred_frames, gt_frames):
        pred = np.asarray(pred, dtype=np.uint8)
        gt = np.asarray(gt, dtype=np.uint8)

        # 确保尺寸一致
        if pred.shape != gt.shape:
            from PIL import Image
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.BILINEAR
            ))

        psnr_list.append(calc_psnr(pred, gt))
        ssim_list.append(calc_ssim(pred, gt))

    return {
        "video_name": video_name,
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "num_frames": len(pred_frames),
    }


def compute_batch_metrics(
    all_pred_frames: list, all_gt_frames: list, video_names: list
) -> dict:
    """
    批量计算多个视频的 PSNR / SSIM。

    Returns:
        dict: {
            "per_video": [{"video_name": str, "psnr": float, "ssim": float}, ...],
            "psnr_mean": float,
            "ssim_mean": float,
        }
    """
    per_video = []
    for pred_frames, gt_frames, name in zip(
        all_pred_frames, all_gt_frames, video_names
    ):
        result = compute_validation_metrics(pred_frames, gt_frames, name)
        per_video.append(result)

    psnr_mean = float(np.mean([r["psnr"] for r in per_video]))
    ssim_mean = float(np.mean([r["ssim"] for r in per_video]))

    return {
        "per_video": per_video,
        "psnr_mean": psnr_mean,
        "ssim_mean": ssim_mean,
    }


def format_metrics_table(batch_results: dict, step: int) -> str:
    """
    格式化指标为精美的 ASCII 表格。

    Returns:
        str: 可直接打印的表格字符串
    """
    sep = "─" * 45
    header = f"{'Video':>12s}   {'PSNR ↑':>10s}   {'SSIM ↑':>10s}"
    lines = [
        "",
        f"  {sep}",
        f"  Validation Metrics @ Step {step}",
        f"  {sep}",
        f"  {header}",
        f"  {sep}",
    ]
    for r in batch_results["per_video"]:
        lines.append(
            f"  {r['video_name']:>12s}   {r['psnr']:10.4f}   {r['ssim']:10.4f}"
        )
    lines.append(f"  {sep}")
    lines.append(
        f"  {'Average':>12s}   {batch_results['psnr_mean']:10.4f}   "
        f"{batch_results['ssim_mean']:10.4f}"
    )
    lines.append(f"  {sep}")
    lines.append("")
    return "\n".join(lines)
