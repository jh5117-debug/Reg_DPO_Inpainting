#!/usr/bin/env python3
"""
InpaintingScore — 视频修复质量自动评分模块

受 VideoDPO OmniScore 思想启发，但为 Video Inpainting 场景独立设计。
使用 VBench 6 个评分维度加权汇总为 inpainting_score，
为 DPO 数据集构建提供自动化的 Best-vs-Worst 配对选择和动态权重。

与 OmniScore 的区别:
  - OmniScore 面向 Text-to-Video，侧重 Prompt 匹配度
  - InpaintingScore 面向 Video Inpainting，侧重背景一致性和时序稳定性
  - 评分引擎不同：OmniScore=CoTracker+CLIP+DOVER, InpaintingScore=VBench
  - 权重分配不同：background_consistency 和 temporal_flickering 各占 25%

核心功能:
  1. InpaintingScorer: 封装 VBench 子模块，score_video() 接口
  2. select_best_worst(): 从 N 个候选中选出最高/最低分
  3. compute_dupfactor(): 动态训练权重（分差越大权重越高）

使用:
    scorer = InpaintingScorer(device='cuda')
    scores = [scorer.score_video(v) for v in candidate_videos]
    best_idx, worst_idx, dupfactor = select_best_worst(scores)
"""

import contextlib
import importlib
import io
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# VBench 路径
VBENCH_ROOT = "/home/hj/VBench"
if VBENCH_ROOT not in sys.path:
    sys.path.insert(0, VBENCH_ROOT)

# 适用于视频修复场景的评分维度及权重
DIMENSION_WEIGHTS = {
    "subject_consistency":    0.15,
    "background_consistency": 0.25,   # 背景恢复最关键
    "temporal_flickering":    0.25,   # 时序一致性最关键
    "motion_smoothness":     0.15,
    "aesthetic_quality":     0.10,
    "imaging_quality":       0.10,
}


class InpaintingScorer:
    """
    封装 VBench 子模块，提供 video → inpainting_score 的一站式评分接口。

    初始化时加载全部子模块（仅一次），后续 score_video() 调用为纯推理。
    """

    def __init__(
        self,
        device: str = "cuda",
        dimensions: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.device = device
        self.dimensions = dimensions or list(DIMENSION_WEIGHTS.keys())
        self.weights = weights or DIMENSION_WEIGHTS

        # 只保留有权重的维度
        self.dimensions = [d for d in self.dimensions if d in self.weights]
        assert len(self.dimensions) > 0, "至少需要一个有效评分维度"

        self._init_submodules()

    def _init_submodules(self):
        """一次性加载 VBench 子模块。"""
        from vbench.utils import init_submodules
        print(f"[InpaintingScorer] 初始化 {len(self.dimensions)} 个维度: "
              f"{', '.join(self.dimensions)}")
        self.submodules = init_submodules(self.dimensions, local=False, read_frame=False)
        print("[InpaintingScorer] 子模块就绪")

    def score_video(self, video_path: str, name: str = "video") -> Dict:
        """
        对单个视频文件计算 InpaintingScore。

        Args:
            video_path: mp4 视频路径
            name: 视频名称（VBench info 的 prompt 字段）

        Returns:
            dict: {
                "path": str,
                "per_dim": {dim: score, ...},      ← 各维度子分数
                "per_dim_weighted": {dim: w*s, ...}, ← 各维度加权后分数
                "inpainting_score": float            ← 加权汇总总分
            }
        """
        from vbench.utils import save_json

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频不存在: {video_path}")

        per_dim = {}

        for dim in self.dimensions:
            try:
                score = self._eval_dimension(video_path, name, dim)
                per_dim[dim] = float(score)
            except Exception as e:
                logging.warning(f"[InpaintingScorer] {dim} 评分失败: {e}")
                per_dim[dim] = -1.0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 加权汇总（跳过失败维度）
        valid = {d: s for d, s in per_dim.items() if s >= 0}
        if valid:
            w_sum = sum(self.weights[d] for d in valid)
            inpainting_score = sum(self.weights[d] * s / w_sum for d, s in valid.items())
        else:
            inpainting_score = 0.0

        # 各维度加权后的贡献值
        per_dim_weighted = {}
        for d, s in per_dim.items():
            if s >= 0 and w_sum > 0:
                per_dim_weighted[d] = round(self.weights[d] * s / w_sum, 6)
            else:
                per_dim_weighted[d] = 0.0

        return {
            "path": video_path,
            "per_dim": per_dim,
            "per_dim_weighted": per_dim_weighted,
            "inpainting_score": inpainting_score,
        }

    def _eval_dimension(self, video_path: str, name: str, dim: str) -> float:
        """调用 VBench 单维度评分。"""
        from vbench.utils import save_json

        # 构造 VBench 要求的 info JSON
        info_list = [{
            "prompt_en": name,
            "dimension": [dim],
            "video_list": [video_path],
        }]
        info_path = video_path + f"._vbench_{dim}_info.json"
        save_json(info_list, info_path)

        try:
            dim_module = importlib.import_module(f"vbench.{dim}")
            compute_fn = getattr(dim_module, f"compute_{dim}")

            # 静默内部输出
            prev_level = logging.root.level
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                avg_score, _ = compute_fn(info_path, self.device, self.submodules[dim])
            logging.disable(prev_level)

            score = float(avg_score) if not isinstance(avg_score, bool) else (1.0 if avg_score else 0.0)
            return score
        finally:
            logging.disable(logging.NOTSET)
            try:
                os.remove(info_path)
            except OSError:
                pass

    def score_videos_batch(self, video_paths: List[str], name: str = "video") -> List[Dict]:
        """批量评分多个视频，打印每个维度子分数。"""
        results = []
        for i, vp in enumerate(video_paths):
            print(f"  [InpaintingScorer] 评分 {i+1}/{len(video_paths)}: {os.path.basename(vp)}")
            r = self.score_video(vp, name=name)
            # 打印各维度子分数 + 加权贡献
            for dim in self.dimensions:
                raw = r["per_dim"].get(dim, -1)
                weighted = r["per_dim_weighted"].get(dim, 0)
                w = self.weights.get(dim, 0)
                print(f"    {dim}: {raw:.4f} (weight={w:.0%}, contribution={weighted:.4f})")
            print(f"    >> inpainting_score = {r['inpainting_score']:.4f}")
            results.append(r)
        return results


def select_best_worst(
    scores: List[Dict],
    dupfactor_beta: float = 1.0,
    dupfactor_baseline: Optional[float] = None,
) -> Tuple[int, int, float, Dict]:
    """
    从 N 个候选评分中选出 Best (Win) 和 Worst (Lose)。

    Args:
        scores: score_video() 返回值的列表
        dupfactor_beta: 动态权重指数，越大越强调高分差样本
        dupfactor_baseline: 归一化基线（默认取中位数分差）

    Returns:
        (best_idx, worst_idx, dupfactor, pair_info)
    """
    all_scores_val = [s["inpainting_score"] for s in scores]

    best_idx = int(np.argmax(all_scores_val))
    worst_idx = int(np.argmin(all_scores_val))

    # 分差
    pair_score_diff = all_scores_val[best_idx] - all_scores_val[worst_idx]

    # dupfactor: 分差越大权重越高（受 VideoDPO 启发）
    if dupfactor_baseline is None:
        # 默认基线 = 0.05（经验值，VBench 各维度标准差约 0.03~0.08）
        dupfactor_baseline = 0.05

    if pair_score_diff > 0:
        dupfactor = (pair_score_diff / dupfactor_baseline) ** dupfactor_beta
        dupfactor = max(1.0, dupfactor)  # 至少为 1
    else:
        dupfactor = 1.0

    pair_info = {
        "best_idx": best_idx,
        "worst_idx": worst_idx,
        "best_inpainting_score": all_scores_val[best_idx],
        "worst_inpainting_score": all_scores_val[worst_idx],
        "pair_score_diff": pair_score_diff,
        "dupfactor": dupfactor,
        "all_inpainting_scores": all_scores_val,
        "all_scores": scores,
    }

    return best_idx, worst_idx, dupfactor, pair_info


def compute_batch_dupfactors(
    all_pair_infos: List[Dict],
    beta: float = 1.0,
) -> List[float]:
    """
    对所有 clip 的 pair_info 重新计算 dupfactor（使用全局中位数作 baseline）。

    训练前调用，确保 dupfactor 在全局分布上有意义。

    Args:
        all_pair_infos: 所有 clip 的 pair_info (select_best_worst 返回值)
        beta: 指数

    Returns:
        更新后的 dupfactor 列表
    """
    diffs = [p["pair_score_diff"] for p in all_pair_infos if p["pair_score_diff"] > 0]
    if not diffs:
        return [1.0] * len(all_pair_infos)

    median_diff = float(np.median(diffs))
    if median_diff <= 0:
        median_diff = 0.05

    dupfactors = []
    for p in all_pair_infos:
        diff = p["pair_score_diff"]
        if diff > 0:
            df = (diff / median_diff) ** beta
            df = max(1.0, df)
        else:
            df = 1.0
        dupfactors.append(df)

    return dupfactors


# ──────────────────────────────────────────────────────────────────────
# 独立运行模式：对指定目录下的视频打分
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="InpaintingScore 视频修复质量评分")
    parser.add_argument("video_paths", nargs="+", help="一个或多个视频路径")
    parser.add_argument("--device", default="cuda", help="计算设备")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    scorer = InpaintingScorer(device=args.device)

    results = []
    for vp in args.video_paths:
        r = scorer.score_video(vp, name=os.path.splitext(os.path.basename(vp))[0])
        results.append(r)
        print(f"\n  {os.path.basename(vp)}: inpainting_score={r['inpainting_score']:.4f}")
        for dim, s in r["per_dim"].items():
            w = DIMENSION_WEIGHTS.get(dim, 0)
            wc = r["per_dim_weighted"].get(dim, 0)
            print(f"    {dim}: {s:.4f} (weight={w:.0%}, contribution={wc:.4f})")

    if len(results) >= 2:
        best_idx, worst_idx, dupfactor, pair_info = select_best_worst(results)
        print(f"\n  Best: [{best_idx}] {os.path.basename(results[best_idx]['path'])} "
              f"({pair_info['best_inpainting_score']:.4f})")
        print(f"  Worst: [{worst_idx}] {os.path.basename(results[worst_idx]['path'])} "
              f"({pair_info['worst_inpainting_score']:.4f})")
        print(f"  Δ = {pair_info['pair_score_diff']:.4f}, dupfactor = {dupfactor:.2f}")

    if args.output:
        out_data = {
            "scores": results,
        }
        if len(results) >= 2:
            out_data["pair_info"] = {
                k: v for k, v in pair_info.items()
                if k != "all_scores"  # 避免过大
            }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存至 {args.output}")


if __name__ == "__main__":
    main()
