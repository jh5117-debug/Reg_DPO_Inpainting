#!/usr/bin/env python3
"""
DPO 偏好对数据集生成脚本（视频级三路对决策略）

策略设计：
  正样本 = GT（Ground Truth 原始帧）
  负样本 = 每个视频跑 3 种管线 → 评分 → 取最差的一路:
    1. blur          — 仅 ProPainter → 天然模糊
    2. hallucination — 仅 DiffuEraser (dummy priori) → 幻觉
    3. flicker       — DiffuEraser 分段多 seed (dummy priori) → 闪烁

  评分方式: mask 区域 SSIM + 时序闪烁惩罚 → score 最低者胜出为负样本

输出目录结构:
  dpo_data/
  ├── {video_name}/
  │   ├── gt_frames/        ← 正样本 (GT)
  │   ├── masks/            ← mask 序列
  │   ├── neg_frames/       ← 三路中最差的
  │   ├── meta.json         ← neg_type, 三路分数
  │   └── comparison.mp4    ← 对比视频 (GT | Mask | Neg)
  └── manifest.json
"""

import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

# 将项目根目录加入 sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_inference_root = "/home/hj/DiffuEraser_new"
if _inference_root not in sys.path:
    sys.path.insert(0, _inference_root)
# from dataset.utils import create_random_shape_with_random_motion


# ──────────────────────────────────────────────────────────────────────
# CenterCrop：与训练集 FinetuneDataset 保持完全一致
# ──────────────────────────────────────────────────────────────────────

def center_crop_frame(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    short = min(target_h, target_w)
    scale = short / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    new_h, new_w = img.shape[:2]
    top  = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return img[top: top + target_h, left: left + target_w]


def center_crop_mask(mask: Image.Image, target_h: int, target_w: int) -> Image.Image:
    w, h = mask.size
    short = min(target_h, target_w)
    scale = short / min(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    mask = mask.resize((new_w, new_h), Image.NEAREST)
    new_h, new_w = mask.size[1], mask.size[0]
    top  = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return mask.crop((left, top, left + target_w, top + target_h))


# ──────────────────────────────────────────────────────────────────────
# 评分工具：将帧列表写为临时 mp4 → InpaintingScorer 评分
# ──────────────────────────────────────────────────────────────────────

def save_rgb_frames_as_mp4(frames: List[np.ndarray], out_path: str, fps: int = 24):
    """将 RGB 帧列表通过 ffmpeg stdin 管道直接编码为 H.264 mp4，绕过所有 Python 编解码器兼容性问题。"""
    import subprocess
    
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    
    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{w}x{h}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        '-preset', 'fast',
        '-an',
        out_path,
    ]
    
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in frames:
        f = np.asarray(frame)
        if f.dtype != np.uint8:
            f = (f * 255).clip(0, 255).astype(np.uint8)
        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def score_candidate_frames(
    scorer, frames: List[np.ndarray], video_name: str, neg_type: str,
    work_dir: str, fps: int = 24,
) -> float:
    """
    将候选帧列表保存为临时 mp4 → 用 InpaintingScorer 评分 → 返回 inpainting_score。
    分数越低 = 质量越差 = 越适合做负样本。
    """
    tmp_mp4 = os.path.join(work_dir, f"_score_{neg_type}.mp4")
    save_rgb_frames_as_mp4(frames, tmp_mp4, fps=fps)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = scorer.score_video(tmp_mp4, name=f"{video_name}_{neg_type}")
        score = result.get("inpainting_score", 0.0)
    except Exception as e:
        print(f"    [score] Error scoring {neg_type}: {e}", flush=True)
        score = 0.0
    _cleanup(tmp_mp4)
    return score


# ──────────────────────────────────────────────────────────────────────
# 帧 ↔ 视频 / 目录 工具
# ──────────────────────────────────────────────────────────────────────

def frames_to_temp_video(frames, fps=24, tmp_dir=None, suffix="_input"):
    h, w = frames[0].shape[:2]
    tmp_path = os.path.join(tmp_dir or tempfile.gettempdir(), f"_dpo{suffix}.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return tmp_path


def masks_to_temp_video(masks, fps=24, tmp_dir=None):
    arr0 = np.array(masks[0])
    h, w = arr0.shape[:2]
    tmp_path = os.path.join(tmp_dir or tempfile.gettempdir(), "_dpo_mask.mp4")
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=False)
    for m in masks:
        writer.write(np.array(m))
    writer.release()
    return tmp_path


def save_frames_to_dir(frames, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.png"),
                     cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def create_dpo_hard_mask(video_length, imageHeight, imageWidth):
    import random
    import numpy as np
    from PIL import Image
    from dataset.utils import get_random_shape
    
    # 尺寸控制：保证面积大概占据 40%-50%
    # 随机多边形约占外接矩形的 70%，因此矩形需占 55%~70% 的面积
    # 宽、高占比 0.75~0.85，乘积约为 0.56~0.72，折算实际多边形即为 40%~50%
    h_ratio = random.uniform(0.75, 0.85)
    w_ratio = random.uniform(0.75, 0.85)
    height = int(imageHeight * h_ratio)
    width = int(imageWidth * w_ratio)
    
    edge_num = random.randint(5, 8)
    ratio = random.randint(6, 8) / 10

    region = get_random_shape(edge_num=edge_num, ratio=ratio, height=height, width=width)
    region_width, region_height = region.size
    
    # 位置控制：必须在画面中部，禁止触碰四周 (预留15%的边缘空气墙)
    margin_x = int(imageWidth * 0.15)
    margin_y = int(imageHeight * 0.15)
    
    min_x = margin_x
    max_x = max(margin_x, imageWidth - margin_x - region_width)
    min_y = margin_y
    max_y = max(margin_y, imageHeight - margin_y - region_height)
    
    pos_x = random.randint(min_x, max_x)
    pos_y = random.randint(min_y, max_y)
    
    # 50% 概率静止，50% 概率缓慢移动
    is_static = random.random() < 0.5
    masks = []
    
    if is_static:
        m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (pos_x, pos_y, pos_x + region_width, pos_y + region_height))
        mask_final = m.convert('L')
        return [mask_final] * video_length
        
    # 缓慢移动 (速度设为 0.5 ~ 1.5 像素/帧)
    speed = random.uniform(0.5, 1.5)
    angle = random.uniform(0, 2 * np.pi)
    
    for _ in range(video_length):
        m = Image.fromarray(np.zeros((imageHeight, imageWidth)).astype(np.uint8))
        m.paste(region, (int(pos_x), int(pos_y), int(pos_x) + region_width, int(pos_y) + region_height))
        masks.append(m.convert('L'))
        
        pos_x += speed * np.cos(angle)
        pos_y += speed * np.sin(angle)
        
        # 触碰“空气墙”时反弹
        if pos_x < min_x or pos_x > max_x:
            angle = np.pi - angle
            pos_x = np.clip(pos_x, min_x, max_x)
        if pos_y < min_y or pos_y > max_y:
            angle = -angle
            pos_y = np.clip(pos_y, min_y, max_y)
            
    return masks


def save_masks_to_dir(masks, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, m in enumerate(masks):
        m.save(os.path.join(out_dir, f"{i:05d}.png"))


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────
# 模型封装
# ──────────────────────────────────────────────────────────────────────

class ModelManager:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffueraser = None
        self.propainter = None

    def load_models(self):
        if self.propainter is None:
            print("[Loading ProPainter ...]", flush=True)
            from propainter.inference_OR import Propainter
            self.propainter = Propainter(self.args.propainter_model_dir, device=self.device)
        if self.diffueraser is None:
            print("[Loading DiffuEraser ...]", flush=True)
            from diffueraser.diffueraser_OR_DPO import DiffuEraser
            self.diffueraser = DiffuEraser(
                self.device, self.args.base_model_path,
                self.args.vae_path, self.args.diffueraser_path,
                ckpt="2-Step", pcm_weights_path=self.args.pcm_weights_path,
            )

    def run_propainter(self, video_path, mask_path, out_dir):
        priori_path = os.path.join(out_dir, "propainter.mp4")
        priori_frames = self.propainter.forward(
            video=video_path, mask=mask_path, output_path=priori_path,
            resize_ratio=1.0, height=-1, width=-1,
            video_length=self.args.video_length,
            mask_dilation=self.args.mask_dilation_iter,
            ref_stride=self.args.ref_stride,
            neighbor_length=self.args.neighbor_length,
            subvideo_length=self.args.subvideo_length,
            raft_iter=20, save_fps=24, save_frames=False,
            fp16=True, return_frames=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return priori_path, priori_frames

    def run_diffueraser(self, video_path, mask_path, priori_path, out_path,
                        nframes, seed=None):
        try:
            self.diffueraser.forward(
                validation_image=video_path, validation_mask=mask_path,
                priori=priori_path, output_path=out_path,
                max_img_size=max(self.args.height, self.args.width) + 100,
                video_length=self.args.video_length,
                mask_dilation_iter=self.args.mask_dilation_iter,
                nframes=nframes, seed=seed, blended=True,
                prompt="", n_prompt="",
            )
        except Exception as e:
            import traceback
            print(f"    [DiffuEraser] Error (seed={seed}): {e}\n{traceback.format_exc()}", flush=True)
            return []
        frames = read_video_frames(out_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return frames


# ──────────────────────────────────────────────────────────────────────
# 三种管线
# ──────────────────────────────────────────────────────────────────────

def pipeline_blur(models, gt_frames, masks, work_dir, nframes):
    """仅 ProPainter → blur"""
    video_tmp = frames_to_temp_video(gt_frames, tmp_dir=work_dir, suffix="_gt")
    mask_tmp = masks_to_temp_video(masks, tmp_dir=work_dir)

    print(f"    [blur] ProPainter ...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        pp_path, pp_frames = models.run_propainter(video_tmp, mask_tmp, work_dir)
    print(f"    [blur] Done", flush=True)

    if isinstance(pp_frames, list) and len(pp_frames) > 0:
        if isinstance(pp_frames[0], np.ndarray):
            result = pp_frames[:nframes]
        else:
            result = read_video_frames(pp_path)[:nframes]
    else:
        result = read_video_frames(pp_path)[:nframes]

    _cleanup(video_tmp, mask_tmp, pp_path)
    return result if len(result) >= nframes else None


def pipeline_hallucination(models, gt_frames, masks, work_dir, nframes, seed):
    """仅 DiffuEraser (无 priori) → hallucination"""
    h, w = gt_frames[0].shape[:2]
    video_tmp = frames_to_temp_video(gt_frames, tmp_dir=work_dir, suffix="_gt")
    mask_tmp = masks_to_temp_video(masks, tmp_dir=work_dir)
    out_path = os.path.join(work_dir, "_halluc.mp4")

    print(f"    [hallucination] DiffuEraser (seed={seed}, no priori) ...", flush=True)
    result = models.run_diffueraser(video_tmp, mask_tmp, None, out_path,
                                    nframes=22, seed=seed)
    n_ok = len(result) if result else 0
    print(f"    [hallucination] => {n_ok} frames", flush=True)

    _cleanup(video_tmp, mask_tmp, out_path)
    return result[:nframes] if result and len(result) >= nframes else None


def pipeline_flicker(models, gt_frames, masks, work_dir, nframes, base_seed):
    """DiffuEraser 双 seed 交错拼接 (无 priori) → 密集 flicker
    
    策略：跑 2 次完整的 DiffuEraser（不同 seed），然后每 8 帧交替取帧。
    这样每个 16 帧评分窗口都恰好包含 1 个闪烁切点，
    同时计算量和原来的 2-chunk 方案完全一样。
    """
    h, w = gt_frames[0].shape[:2]
    video_tmp = frames_to_temp_video(gt_frames, tmp_dir=work_dir, suffix="_gt")
    mask_tmp = masks_to_temp_video(masks, tmp_dir=work_dir)

    seed_a = base_seed + 7
    seed_b = base_seed + 1007

    # ── Run 1: seed_A 全片 ──
    out_a = os.path.join(work_dir, "_flick_a.mp4")
    print(f"      run A (seed={seed_a}, full {nframes} frames)", flush=True)
    full_a = models.run_diffueraser(video_tmp, mask_tmp, None, out_a,
                                    nframes=22, seed=seed_a)
    _cleanup(out_a)
    if not full_a or len(full_a) < nframes:
        print(f"      run A FAILED", flush=True)
        _cleanup(video_tmp, mask_tmp)
        return None

    # ── Run 2: seed_B 全片 ──
    out_b = os.path.join(work_dir, "_flick_b.mp4")
    print(f"      run B (seed={seed_b}, full {nframes} frames)", flush=True)
    full_b = models.run_diffueraser(video_tmp, mask_tmp, None, out_b,
                                    nframes=22, seed=seed_b)
    _cleanup(out_b)
    if not full_b or len(full_b) < nframes:
        print(f"      run B FAILED", flush=True)
        _cleanup(video_tmp, mask_tmp)
        return None

    # ── 每 8 帧交替拼接：A-B-A-B-... ──
    interleave_step = 8
    assembled = []
    sources = [full_a, full_b]
    n_transitions = 0
    for i in range(0, nframes, interleave_step):
        chunk_end = min(i + interleave_step, nframes)
        src_idx = (i // interleave_step) % 2
        assembled.extend(sources[src_idx][i:chunk_end])
        if i > 0:
            n_transitions += 1

    _cleanup(video_tmp, mask_tmp)
    if len(assembled) < nframes:
        return None
    print(f"    [flicker] Interleaved {nframes} frames (A/B every {interleave_step}f, {n_transitions} transitions)", flush=True)
    return assembled


# ──────────────────────────────────────────────────────────────────────
# 数据集扫描
# ──────────────────────────────────────────────────────────────────────

def scan_davis(davis_root, min_frames):
    entries = []
    train_list = os.path.join(davis_root, "ImageSets", "2017", "train.txt")
    if not os.path.exists(train_list):
        print(f"Warning: {train_list} not found, skipping DAVIS")
        return entries
    with open(train_list) as f:
        names = [line.strip() for line in f if line.strip()]
    for vname in names:
        d = os.path.join(davis_root, "JPEGImages", "480p", vname)
        if not os.path.isdir(d):
            continue
        flist = sorted(f for f in os.listdir(d) if f.endswith((".jpg", ".png")))
        if len(flist) >= min_frames:
            entries.append((f"davis_{vname}", d, flist))
    print(f"DAVIS: {len(entries)} videos (>= {min_frames} frames)", flush=True)
    return entries


def scan_ytvos(ytvos_root, min_frames):
    entries = []
    base = os.path.join(ytvos_root, "JPEGImages")
    if not os.path.isdir(base):
        print(f"Warning: {base} not found, skipping YTBV")
        return entries
    for vid in sorted(os.listdir(base)):
        d = os.path.join(base, vid)
        if not os.path.isdir(d):
            continue
        flist = sorted(f for f in os.listdir(d) if f.endswith((".jpg", ".png")))
        if len(flist) >= min_frames:
            entries.append((f"ytbv_{vid}", d, flist))
    print(f"YTBV: {len(entries)} videos (>= {min_frames} frames)", flush=True)
    return entries


# ──────────────────────────────────────────────────────────────────────
# 核心：处理一个视频
# ──────────────────────────────────────────────────────────────────────

def process_one_video(
    models: ModelManager,
    scorer,
    video_name: str,
    jpeg_dir: str,
    frame_list: List[str],
    output_dir: str,
    args,
) -> Optional[Dict]:
    """
    对一个视频跑 3 种管线 → InpaintingScorer 评分 → 取最差做负样本。
    短视频 (<22 帧) 直接用 blur 作为负样本（不评分）。
    """
    video_out_dir = os.path.join(output_dir, video_name)

    # resume 检查
    meta_path = os.path.join(video_out_dir, "meta.json")
    if args.resume and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                existing = json.load(f)
            if "neg_type" in existing:
                print(f"  [skip] already completed ({existing['neg_type']})", flush=True)
                return _build_entry(video_out_dir, existing, len(frame_list))
        except Exception:
            pass

    os.makedirs(video_out_dir, exist_ok=True)
    nframes = len(frame_list)

    # 读取帧 + CenterCrop
    gt_frames = []
    for fname in frame_list:
        img = cv2.imread(os.path.join(jpeg_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = center_crop_frame(img, args.height, args.width)
        gt_frames.append(img)

    # 生成随机 mask (满足DPO困难度需求: 面积30~40%, 居中, 慢速或静止)
    masks = create_dpo_hard_mask(nframes, imageHeight=args.height, imageWidth=args.width)
    masks = [center_crop_mask(m, args.height, args.width) for m in masks]

    # 保存 GT 和 mask
    save_frames_to_dir(gt_frames, os.path.join(video_out_dir, "gt_frames"))
    save_masks_to_dir(masks, os.path.join(video_out_dir, "masks"))

    # RNG for seed
    rng = random.Random(args.seed + hash(video_name))
    seed = rng.randint(0, 2**31)

    # 临时工作目录
    work_dir = os.path.join(video_out_dir, "_work")
    os.makedirs(work_dir, exist_ok=True)

    can_run_de = nframes >= 22

    # ── 短视频：直接用 blur ──
    if not can_run_de:
        print(f"  Short video ({nframes} < 22): blur only", flush=True)
        blur_frames = pipeline_blur(models, gt_frames, masks, work_dir, nframes)
        _cleanup_work(work_dir)
        if not blur_frames:
            print(f"  [FAIL] blur pipeline failed", flush=True)
            return None
        save_frames_to_dir(blur_frames, os.path.join(video_out_dir, "neg_frames"))
        meta = {"neg_type": "blur", "selected_score": -1, "all_scores": {}, "seed": seed, "num_frames": nframes}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        _make_comparison(gt_frames, masks, blur_frames, video_out_dir, "blur", args.comparison_fps)
        return [_build_entry(video_out_dir, meta, nframes)]

    # ── 正常视频：三路并行首先生满全片 ──
    candidates = {}  

    # 1. blur
    print(f"  [1/3] blur", flush=True)
    blur_frames = pipeline_blur(models, gt_frames, masks, work_dir, nframes)
    if blur_frames: candidates["blur"] = blur_frames
    else: print(f"    FAILED", flush=True)

    # 2. hallucination
    print(f"  [2/3] hallucination", flush=True)
    hall_frames = pipeline_hallucination(models, gt_frames, masks, work_dir, nframes, seed)
    if hall_frames: candidates["hallucination"] = hall_frames
    else: print(f"    FAILED", flush=True)

    # 3. flicker
    print(f"  [3/3] flicker", flush=True)
    flick_frames = pipeline_flicker(models, gt_frames, masks, work_dir, nframes, seed)
    if flick_frames: candidates["flicker"] = flick_frames
    else: print(f"    FAILED", flush=True)

    if not candidates:
        print(f"  [FAIL] No candidates generated", flush=True)
        _cleanup_work(work_dir)
        return None

    # ── 依据 16 帧切小段分别评分并缝合抽出 最差两名 ──
    chunk_size = 16
    start = 0
    neg_frames_1 = []
    neg_frames_2 = []
    chunk_records = []

    print(f"\n  [Scoring] Evaluating in {chunk_size}-frame chunks...", flush=True)
    while start < nframes:
        end = start + chunk_size
        # 如果剩下的不够 16 帧，直接并入本次一并评分
        if nframes - end < chunk_size:
            end = nframes

        print(f"    chunk [{start}:{end}]", flush=True)

        chunk_scores = {}
        for cname, cframes in candidates.items():
            if cframes is not None and len(cframes) >= end:
                c_chunk = cframes[start:end]
                sc = score_candidate_frames(scorer, c_chunk, video_name,
                                            f"{cname}_{start}_{end}", work_dir)
                chunk_scores[cname] = sc
                print(f"      {cname:15s}: {sc:.4f}", flush=True)

        if not chunk_scores:
            print("      [FAIL] No candidates available for this chunk", flush=True)
            break

        # 根据 inpainting_score 排序挑选最差的两名（分数越低越差）
        sorted_methods = sorted(chunk_scores.keys(), key=lambda t: chunk_scores[t])
        worst_1 = sorted_methods[0]
        worst_2 = sorted_methods[1] if len(sorted_methods) > 1 else worst_1

        print(f"      => Worst 1: {worst_1}, Worst 2: {worst_2}", flush=True)

        chunk_records.append({
            "start": start, "end": end,
            "worst_1": worst_1, "score_1": round(chunk_scores[worst_1], 5),
            "worst_2": worst_2, "score_2": round(chunk_scores.get(worst_2, chunk_scores[worst_1]), 5),
        })

        # 截取两家的那几帧，追加进最终集合（缝合成长视频）
        neg_frames_1.extend(candidates[worst_1][start:end])
        neg_frames_2.extend(candidates[worst_2][start:end])

        start = end

    # 确保帧数完整
    if len(neg_frames_1) < nframes:
        print(f"  [FAIL] Failed to construct chimera videos ({len(neg_frames_1)}/{nframes})", flush=True)
        _cleanup_work(work_dir)
        return None

    # 保存两套缝合怪负样本
    save_frames_to_dir(neg_frames_1, os.path.join(video_out_dir, "neg_frames_1"))
    save_frames_to_dir(neg_frames_2, os.path.join(video_out_dir, "neg_frames_2"))

    # 写 meta.json
    meta = {
        "neg_type": "chimera_chunked",
        "chunks": chunk_records,
        "seed": seed,
        "num_frames": nframes,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 对照视频
    _make_comparison(gt_frames, masks, neg_frames_1, video_out_dir, "worst_1_chimera", args.comparison_fps)

    # 构建两条记录
    entry_1 = _build_entry(video_out_dir, meta, nframes)
    entry_1["neg_frames"] = os.path.join(video_out_dir, "neg_frames_1")
    entry_1["score"] = sum(c["score_1"] for c in chunk_records) / len(chunk_records)

    entry_2 = _build_entry(video_out_dir, meta, nframes)
    entry_2["neg_frames"] = os.path.join(video_out_dir, "neg_frames_2")
    entry_2["score"] = sum(c["score_2"] for c in chunk_records) / len(chunk_records)

    _cleanup_work(work_dir)
    return [entry_1, entry_2]

def _cleanup_work(work_dir):
    try:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass


def _make_comparison(gt_frames, masks, neg_frames, video_out_dir, neg_type, fps):
    try:
        from visualize_comparison import create_3in1_comparison_video
        mask_np = [np.array(m) for m in masks]
        create_3in1_comparison_video(
            gt_frames=gt_frames, mask_frames=mask_np, neg_frames=neg_frames,
            output_path=os.path.join(video_out_dir, "comparison.mp4"),
            fps=fps, neg_type=neg_type,
        )
    except Exception as e:
        print(f"  [3in1] Error: {e}", flush=True)


def _build_entry(video_out_dir, meta, nframes):
    return {
        "gt_frames": os.path.join(video_out_dir, "gt_frames"),
        "masks": os.path.join(video_out_dir, "masks"),
        "neg_frames": os.path.join(video_out_dir, "neg_frames"),
        "neg_type": meta.get("neg_type", "unknown"),
        "num_frames": nframes,
        "score": meta.get("selected_score", 0),
    }


# ──────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPO dataset: video-level 3-way scoring")
    p.add_argument("--davis_root", type=str, default="dataset/DAVIS")
    p.add_argument("--ytvos_root", type=str, default="dataset/YTBV")
    p.add_argument("--output_dir", type=str, default="dpo_data")
    p.add_argument("--min_video_frames", type=int, default=22)
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--comparison_fps", type=int, default=8)
    # 模型路径
    p.add_argument("--base_model_path", type=str,
                    default="/home/hj/DiffuEraser_new/weights/stable-diffusion-v1-5")
    p.add_argument("--vae_path", type=str,
                    default="/home/hj/DiffuEraser_new/weights/sd-vae-ft-mse")
    p.add_argument("--diffueraser_path", type=str,
                    default="/home/hj/DiffuEraser_new/weights/diffuEraser")
    p.add_argument("--propainter_model_dir", type=str,
                    default="/home/hj/DiffuEraser_new/weights/propainter")
    p.add_argument("--pcm_weights_path", type=str,
                    default="/home/hj/DiffuEraser_new/weights/PCM_Weights")
    # 推理参数
    p.add_argument("--mask_dilation_iter", type=int, default=8)
    p.add_argument("--ref_stride", type=int, default=10)
    p.add_argument("--neighbor_length", type=int, default=10)
    p.add_argument("--subvideo_length", type=int, default=80)
    p.add_argument("--video_length", type=int, default=-1)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 扫描
    all_videos = []
    all_videos.extend(scan_davis(args.davis_root, args.min_video_frames))
    all_videos.extend(scan_ytvos(args.ytvos_root, args.min_video_frames))
    print(f"\nTotal: {len(all_videos)} videos\n", flush=True)

    if args.max_videos and len(all_videos) > args.max_videos:
        random.shuffle(all_videos)
        all_videos = all_videos[:args.max_videos]
        print(f"Sampled {args.max_videos} videos\n", flush=True)

    # 加载模型
    models = ModelManager(args)
    models.load_models()

    # 加载评分器 (InpaintingScorer)
    print("[Loading InpaintingScorer ...]", flush=True)
    from score_inpainting_quality import InpaintingScorer
    scorer = InpaintingScorer(device="cuda" if torch.cuda.is_available() else "cpu")
    print("[InpaintingScorer ready]", flush=True)

    # 处理
    manifest = {}
    neg_type_counts = {"blur": 0, "hallucination": 0, "flicker": 0}
    total_t0 = time.time()

    for i, (video_name, jpeg_dir, frame_list) in enumerate(all_videos):
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(all_videos)}] {video_name} ({len(frame_list)} frames)", flush=True)
        print(f"{'='*60}", flush=True)

        entries = process_one_video(models, scorer, video_name, jpeg_dir, frame_list,
                                   args.output_dir, args)
        if entries:
            for idx, entry in enumerate(entries):
                manifest[f"{video_name}_part{idx+1}"] = entry
                nt = entry.get("neg_type", "unknown")
                if nt not in neg_type_counts:
                    neg_type_counts[nt] = 0
                neg_type_counts[nt] += 1

        # 每 5 个视频 checkpoint
        if (i + 1) % 5 == 0:
            _save_manifest(args.output_dir, manifest, neg_type_counts)

    _save_manifest(args.output_dir, manifest, neg_type_counts)
    elapsed = time.time() - total_t0
    print(f"\n{'='*60}", flush=True)
    print(f"Done! {len(manifest)} videos in {elapsed:.0f}s", flush=True)
    print(f"  blur:{neg_type_counts['blur']} hall:{neg_type_counts['hallucination']} flick:{neg_type_counts['flicker']}", flush=True)
    print(f"{'='*60}\n", flush=True)


def _save_manifest(output_dir, manifest, counts):
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    total = sum(counts.values())
    print(f"\n  [checkpoint] {total} videos (blur:{counts['blur']} hall:{counts['hallucination']} flick:{counts['flicker']})", flush=True)


if __name__ == "__main__":
    main()
