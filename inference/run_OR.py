#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object Removal (OR) Inference Script  —  Simplified
=====================================================

ProPainter + DiffuEraser pipeline for Object Removal.
Text guidance via unified YAML or per-video YAML.

Example (batch, DAVIS, with text guidance):

    CUDA_VISIBLE_DEVICES=0 python run_OR.py \
      --dataset davis \
      --video_root /path/to/DAVIS/JPEGImages/Full-Resolution \
      --mask_root  /path/to/DAVIS/Annotations/Full-Resolution \
      --save_path  results_OR \
      --video_length 60 \
      --ref_stride 6 --neighbor_length 25 --subvideo_length 80 \
      --mask_dilation_iter 4 \
      --save_comparison \
      --base_model_path  weights/stable-diffusion-v1-5 \
      --vae_path         weights/sd-vae-ft-mse \
      --diffueraser_path weights/diffuEraser \
      --propainter_model_dir weights/propainter \
      --pcm_weights_path     weights/PCM_Weights \
      --height 360 --width 640 \
      --use_text \
      --unified_prompt_yaml prompt_cache/all_captions.yaml \
      --text_guidance_scale 3.5

Example (single video, no text):

    CUDA_VISIBLE_DEVICES=0 python run_OR.py \
      --input_video examples/example3/video.mp4 \
      --input_mask  examples/example3/mask.mp4 \
      --save_path results_single \
      --video_length 10 \
      --height 480 --width 856
"""

import os
import sys
import gc
import json
import time
import argparse
import yaml
import warnings
import logging
from pathlib import Path
from typing import Optional, List, Tuple

# ── Suppress noisy warnings globally ──
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import cv2
import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# 0)  Ensure repo root is on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# VBench — 视频质量评估（位于 /home/hj/VBench，非 pip 安装）
VBENCH_ROOT = Path("/home/hj/VBench")
if str(VBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VBENCH_ROOT))

from propainter.inference_OR import Propainter, get_device          # noqa: E402
from diffueraser.diffueraser_OR import DiffuEraser                  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION A — YAML Prompt Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_prompt_from_yaml(yaml_path: str):
    """从单个 per-video YAML 加载 prompt / n_prompt / text_guidance_scale。"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    prompt = ""
    if 'prompt' in config and config['prompt']:
        p = config['prompt']
        prompt = p[0] if isinstance(p, list) else str(p)

    n_prompt = ""
    if 'n_prompt' in config and config['n_prompt']:
        np_ = config['n_prompt']
        n_prompt = np_[0] if isinstance(np_, list) else str(np_)

    text_guidance_scale = config.get('text_guidance_scale', 2.0)
    return prompt, n_prompt, text_guidance_scale


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION B — Video / Frame I/O Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_image_dir(path: str, exts=(".jpg", ".jpeg", ".png")) -> List[str]:
    """Return sorted list of image file paths in *path*."""
    return sorted(
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.lower().endswith(exts)
    )


def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: int = 24,
    size: Optional[Tuple[int, int]] = None,
) -> str:
    """Convert a directory of frames to an mp4 video."""
    files = _read_image_dir(frames_dir)
    if not files:
        raise ValueError(f"No image frames found in: {frames_dir}")

    first = cv2.imread(files[0])
    if first is None:
        raise ValueError(f"Cannot read image: {files[0]}")

    if size is not None:
        w, h = size
        w, h = w - w % 8, h - h % 8
    else:
        h, w = first.shape[:2]
        w, h = w - w % 8, h - h % 8

    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        writer.write(img)

    writer.release()

    _ffmpeg_remux_h264(tmp_path, output_path)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, output_path)
    return output_path


def _ffmpeg_remux_h264(src_mp4, dst_mp4):
    """Re-encode mp4v → H.264 via ffmpeg for universal player compatibility."""
    import subprocess, shutil as _shutil
    ffmpeg = _shutil.which("ffmpeg")
    if not ffmpeg:
        return
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src_mp4),
             "-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-pix_fmt", "yuv420p", "-an", str(dst_mp4)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except Exception:
        pass


def create_comparison_video_from_frames(in_frames, mask_frames, pp_frames, de_frames,
                                        output_path, fps=24,
                                        max_cell_w=960, max_cell_h=540):
    """Create 2x2 comparison video from frame lists (RGB numpy arrays).

    Layout:  Input      | Mask
             ProPainter | DiffuEraser
    """
    n = min(len(in_frames), len(mask_frames), len(pp_frames), len(de_frames))
    if n == 0:
        print(f"[WARN] No frames to create comparison video")
        return

    h_orig, w_orig = in_frames[0].shape[:2]

    w, h = w_orig, h_orig
    if w > max_cell_w or h > max_cell_h:
        scale = min(max_cell_w / w, max_cell_h / h)
        w = int(w * scale)
        h = int(h * scale)
    w = w - w % 2
    h = h - h % 2

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = str(out_path) + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(tmp_path, fourcc, fps, (w * 2, h * 2))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(w, h) / 600)
    font_thickness = max(1, int(font_scale * 2))
    labels = ['Input', 'Mask', 'ProPainter', 'DiffuEraser']

    for i in range(n):
        f_in = cv2.cvtColor(cv2.resize(in_frames[i], (w, h)), cv2.COLOR_RGB2BGR)

        m = mask_frames[i]
        if m.ndim == 2:
            m_bgr = cv2.cvtColor(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        else:
            m_bgr = cv2.resize(m, (w, h))
            if m_bgr.shape[2] == 3:
                m_bgr = cv2.cvtColor(m_bgr, cv2.COLOR_RGB2BGR)

        f_pp = cv2.cvtColor(cv2.resize(pp_frames[i], (w, h)), cv2.COLOR_RGB2BGR)
        f_de = cv2.cvtColor(cv2.resize(de_frames[i], (w, h)), cv2.COLOR_RGB2BGR)

        cells = [f_in, m_bgr, f_pp, f_de]
        for cell, label in zip(cells, labels):
            ts = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(cell, (4, 4), (12 + ts[0], 12 + ts[1]), (0, 0, 0), -1)
            cv2.putText(cell, label, (8, 8 + ts[1]), font, font_scale, (255, 255, 255), font_thickness)

        top = np.hstack([cells[0], cells[1]])
        bot = np.hstack([cells[2], cells[3]])
        vw.write(np.vstack([top, bot]))

    vw.release()

    _ffmpeg_remux_h264(tmp_path, str(out_path))
    if os.path.exists(str(out_path)) and os.path.getsize(str(out_path)) > 0:
        os.remove(tmp_path)
    else:
        os.rename(tmp_path, str(out_path))
    print(f"  [Comparison] Saved 4-in-1 video: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION C — Frame I/O (read from directory or video file)
# ═══════════════════════════════════════════════════════════════════════════

def _load_frames_rgb(path: str, max_frames: int = -1) -> List[np.ndarray]:
    """Load RGB frames from path (directory or .mp4)."""
    bgr = _load_frames_bgr(path, max_frames)
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr]


def _load_frames_bgr(path: str, max_frames: int = -1) -> List[np.ndarray]:
    """Load BGR frames from path (directory or .mp4)."""
    if os.path.isdir(path):
        files = _read_image_dir(path)
        if max_frames > 0:
            files = files[:max_frames]
        return [cv2.imread(f) for f in files if cv2.imread(f) is not None]

    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1
        if max_frames > 0 and idx >= max_frames:
            break
    cap.release()
    return frames


def _load_mask_frames_gray(path: str, max_frames: int = -1) -> List[np.ndarray]:
    """Load grayscale masks from path (directory or .mp4)."""
    if os.path.isdir(path):
        files = _read_image_dir(path)
        if max_frames > 0:
            files = files[:max_frames]
        masks = []
        for f in files:
            m = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks.append(m)
        return masks

    cap = cv2.VideoCapture(path)
    masks = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        masks.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
        if max_frames > 0 and idx >= max_frames:
            break
    cap.release()
    return masks


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION D — Dataset batch utilities
# ═══════════════════════════════════════════════════════════════════════════

def get_video_list_davis(video_root: str, mask_root: str) -> List[dict]:
    """Build a list of {name, video_path, mask_path, is_sequence} from a DAVIS-like hierarchy."""
    results = []
    for name in sorted(os.listdir(video_root)):
        vp = os.path.join(video_root, name)
        mp = os.path.join(mask_root, name)
        if os.path.isdir(vp) and os.path.isdir(mp):
            results.append({"name": name, "video_path": vp, "mask_path": mp, "is_sequence": True})
    return results


def get_video_list_youtube_vos(video_root: str, mask_root: str) -> List[dict]:
    """Build a list from a YouTube-VOS hierarchy."""
    results = []
    for name in sorted(os.listdir(video_root)):
        vp = os.path.join(video_root, name)
        mp = os.path.join(mask_root, name)
        if os.path.isdir(vp) and os.path.isdir(mp):
            results.append({"name": name, "video_path": vp, "mask_path": mp, "is_sequence": True})
    return results


def get_video_list_custom(video_root: str, mask_root: str) -> List[dict]:
    """Build a list for a custom hierarchy."""
    results = []
    for name in sorted(os.listdir(video_root)):
        vp = os.path.join(video_root, name)
        mp = os.path.join(mask_root, name)
        if os.path.isdir(vp) and os.path.isdir(mp):
            results.append({"name": name, "video_path": vp, "mask_path": mp, "is_sequence": True})
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION E — GPU Memory Management (Model Offloading)
# ═══════════════════════════════════════════════════════════════════════════

def _offload_to_cpu(obj, label=""):
    """Move a model/pipeline to CPU to free GPU memory."""
    if obj is None or isinstance(obj, str):
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(obj, 'pipeline'):        # DiffuEraser
                obj.pipeline.to('cpu')
            elif hasattr(obj, 'pipe'):          # SDInpaintAnchorInpainter
                obj.pipe.to('cpu')
        if label:
            print(f"  [Offload] {label} → CPU")
    except Exception as e:
        if label:
            print(f"  [Offload WARN] {label}: {e}")
    torch.cuda.empty_cache()


def _load_to_gpu(obj, device, label=""):
    """Move a model/pipeline back to GPU."""
    if obj is None or isinstance(obj, str):
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(obj, 'pipeline'):        # DiffuEraser
                obj.pipeline.to(device, torch.float16)
            elif hasattr(obj, 'pipe'):
                obj.pipe.to(device)
        if label:
            print(f"  [Load]    {label} → GPU")
    except Exception as e:
        if label:
            print(f"  [Load WARN] {label}: {e}")


def _gpu_mem_info():
    """Return current GPU memory usage string."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU mem: {alloc:.1f}G alloc / {reserved:.1f}G reserved"
    return "GPU: N/A"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION F — Single-video inference
# ═══════════════════════════════════════════════════════════════════════════

def process_single_video(
    args,
    video_inpainting_sd,
    propainter_model,
    video_path: str,
    mask_path: str,
    out_dir: str,
    name: str = "video",
    prompt: str = "",
    n_prompt: str = "",
    text_guidance_scale: float = 2.0,
) -> Tuple[str, str, List[np.ndarray], float]:
    """
    Run ProPainter + DiffuEraser on a single video input and save outputs to out_dir.

    Returns:
        (priori_path, pred_path, priori_frames, inference_time)
    """
    t0 = time.time()
    device = get_device()

    # ── GPU Offloading: free GPU for ProPainter ──
    if getattr(args, 'offload_models', False):
        _offload_to_cpu(video_inpainting_sd, "DiffuEraser")
        print(f"  [{_gpu_mem_info()}]")

    # 1) Run ProPainter —————————————————————————————————————————————————————
    priori_path = os.path.join(out_dir, "propainter.mp4")

    priori_frames = propainter_model.forward(
        video=video_path,
        mask=mask_path,
        output_path=priori_path,
        resize_ratio=1.0,
        height=-1,
        width=-1,
        video_length=args.video_length,
        mask_dilation=args.mask_dilation_iter,
        ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length,
        subvideo_length=args.subvideo_length,
        raft_iter=20,
        save_fps=24,
        save_frames=False,
        fp16=True,
        return_frames=True,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) Run DiffuEraser ————————————————————————————————————————————————————
    if getattr(args, 'offload_models', False):
        _load_to_gpu(video_inpainting_sd, device, "DiffuEraser")
        print(f"  [{_gpu_mem_info()}]")

    pred_path = os.path.join(out_dir, "diffueraser.mp4")

    use_text_for_main = args.use_text and prompt.strip()
    main_guidance_scale = text_guidance_scale if use_text_for_main else None

    video_inpainting_sd.forward(
        validation_image=video_path,
        validation_mask=mask_path,
        priori=priori_path,
        output_path=pred_path,
        max_img_size=max(args.height, args.width) + 100,
        video_length=args.video_length,
        mask_dilation_iter=args.mask_dilation_iter,
        nframes=22,
        seed=None,
        blended=True,
        prompt=prompt if use_text_for_main else "",
        n_prompt=n_prompt if use_text_for_main else "",
        guidance_scale=main_guidance_scale,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dt = time.time() - t0
    print(f"  Inference time: {dt:.2f}s")

    return priori_path, pred_path, priori_frames, dt


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION G — Argument parser
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Object Removal (OR) Inference — unified ProPainter + DiffuEraser pipeline."
    )

    # -------------------- Input / Output --------------------
    parser.add_argument("--input_video", type=str, help="Path to input video (single-video mode)")
    parser.add_argument("--input_mask", type=str, help="Path to mask video (single-video mode)")
    parser.add_argument("--save_path", type=str, default="results", help="Directory to save all outputs")

    # -------------------- Batch mode (dataset) --------------------
    parser.add_argument("--dataset", type=str, choices=["davis", "youtube-vos", "custom"], help="Dataset name")
    parser.add_argument("--video_root", type=str, help="Root directory of input frames (batch mode)")
    parser.add_argument("--mask_root", type=str, help="Root directory of mask frames (batch mode)")

    # -------------------- Processing parameters --------------------
    parser.add_argument("--video_length", type=int, default=-1, help="Max frames to process per video (-1 for all)")
    parser.add_argument("--height", type=int, default=-1, help="Target height for processing (-1 for original)")
    parser.add_argument("--width", type=int, default=-1, help="Target width for processing (-1 for original)")

    # -------------------- ProPainter settings --------------------
    parser.add_argument("--ref_stride", type=int, default=10, help="ProPainter ref stride")
    parser.add_argument("--neighbor_length", type=int, default=20, help="ProPainter neighbor length")
    parser.add_argument("--subvideo_length", type=int, default=80, help="ProPainter subvideo length")
    parser.add_argument("--mask_dilation_iter", type=int, default=4, help="Mask dilation iterations")

    # -------------------- Model paths --------------------
    parser.add_argument("--base_model_path", type=str, default="weights/stable-diffusion-v1-5")
    parser.add_argument("--vae_path", type=str, default="weights/sd-vae-ft-mse")
    parser.add_argument("--diffueraser_path", type=str, default="weights/diffuEraser")
    parser.add_argument("--propainter_model_dir", type=str, default="weights/propainter")
    parser.add_argument("--pcm_weights_path", type=str, default="weights/PCM_Weights")

    # -------------------- Memory management --------------------
    parser.add_argument("--offload_models", action="store_true",
                        help="Offload inactive models to CPU between stages to reduce GPU memory.")

    # -------------------- Text / Prompt --------------------
    parser.add_argument("--use_text", action="store_true",
                        help="Enable text embedding in main network (requires prompt source)")
    parser.add_argument("--unified_prompt_yaml", type=str, default=None,
                        help="Path to unified YAML containing prompts for all videos (from generate_captions.py)")
    parser.add_argument("--prompt_root", type=str, default=None,
                        help="Directory containing per-video YAML files (e.g. bear.yaml)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Direct text prompt for single-video mode (overrides YAML)")
    parser.add_argument("--n_prompt", type=str, default=None,
                        help="Direct negative prompt for single-video mode (overrides YAML)")
    parser.add_argument("--text_guidance_scale", type=float, default=None,
                        help="CFG guidance scale for text conditioning. "
                             "If not set, uses YAML value or defaults to 2.0.")

    # -------------------- Output artifacts --------------------
    parser.add_argument("--save_comparison", action="store_true", help="Save 4-in-1 comparison video")
    parser.add_argument("--comparison_fps", type=int, default=24, help="FPS for comparison videos")
    parser.add_argument("--summary_out", type=str, default="summary.json", help="Summary JSON filename")

    # -------------------- VBench evaluation --------------------
    VBENCH_DEFAULT_DIMS = [
        "subject_consistency", "background_consistency",
        "temporal_flickering", "motion_smoothness",
        "aesthetic_quality", "imaging_quality",
    ]
    parser.add_argument("--eval_vbench", action="store_true",
                        help="Enable VBench video quality evaluation after inference")
    parser.add_argument("--vbench_dimensions", nargs="+", default=VBENCH_DEFAULT_DIMS,
                        help="VBench dimensions to evaluate (default: 8 dims relevant to inpainting)")
    parser.add_argument("--vbench_output", type=str, default="vbench_results.json",
                        help="Filename for VBench evaluation results")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION H — VBench Evaluation (per-video inline)
# ═══════════════════════════════════════════════════════════════════════════

def init_vbench(args, device):
    """初始化 VBench 子模块（只调用一次）。返回 (dimensions, submodules_dict) 或 None。"""
    import importlib
    from vbench.utils import init_submodules

    dimensions = list(args.vbench_dimensions)

    # overall_consistency 需要 text prompt（ViCLIP），未启用则跳过
    if not args.use_text and "overall_consistency" in dimensions:
        print("  [VBench] Skipping 'overall_consistency' (requires --use_text)")
        dimensions = [d for d in dimensions if d != "overall_consistency"]

    if not dimensions:
        print("  [VBench] No dimensions to evaluate.")
        return None

    print(f"\n[VBench] Initializing {len(dimensions)} dimension(s): {', '.join(dimensions)}")
    submodules_dict = init_submodules(dimensions, local=False, read_frame=False)
    print("[VBench] Models ready.\n")
    return {"dimensions": dimensions, "submodules": submodules_dict}


def evaluate_single_video_vbench(vbench_ctx, name, priori_path, pred_path, save_dir, device):
    """对单个视频的 ProPainter + DiffuEraser 输出运行 VBench，返回 per-video 分数 dict。"""
    import importlib
    import contextlib, io, logging
    from vbench.utils import save_json

    dimensions = vbench_ctx["dimensions"]
    submodules_dict = vbench_ctx["submodules"]

    row_result = {"name": name, "propainter": {}, "diffueraser": {}}

    for method, video_path in [("propainter", priori_path), ("diffueraser", pred_path)]:
        if not video_path or not os.path.exists(video_path):
            continue

        info_list = [{
            "prompt_en": name,
            "dimension": dimensions,
            "video_list": [video_path],
        }]
        info_path = os.path.join(save_dir, f"_vbench_{method}_info.json")
        save_json(info_list, info_path)

        for dim in dimensions:
            try:
                dim_module = importlib.import_module(f"vbench.{dim}")
                compute_fn = getattr(dim_module, f"compute_{dim}")
                # 静默 VBench 内部输出（模型加载、进度条、日志）
                _prev_level = logging.root.level
                logging.disable(logging.CRITICAL)
                with contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    avg_score, _ = compute_fn(info_path, device, submodules_dict[dim])
                logging.disable(_prev_level)
                score = float(avg_score) if not isinstance(avg_score, bool) else (1.0 if avg_score else 0.0)
                row_result[method][dim] = score
            except Exception as e:
                logging.disable(logging.NOTSET)
                print(f"    [VBench] {method}/{dim} error: {e}")
                row_result[method][dim] = -1.0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            os.remove(info_path)
        except OSError:
            pass

    # ── 打印单视频对比表 ──
    valid_dims = [d for d in dimensions
                  if row_result["propainter"].get(d, -1.0) >= 0
                  or row_result["diffueraser"].get(d, -1.0) >= 0]
    if valid_dims:
        print(f"  ┌─ VBench [{name}] {'─' * max(0, 42 - len(name))}┐")
        print(f"  │ {'Dimension':<26s} {'PP':>8s} {'DE':>8s} {'Δ':>8s} │")
        print(f"  │ {'─' * 52} │")
        for dim in valid_dims:
            pp = row_result["propainter"].get(dim, -1.0)
            de = row_result["diffueraser"].get(dim, -1.0)
            pp_s = f"{pp:.4f}" if pp >= 0 else "  ─"
            de_s = f"{de:.4f}" if de >= 0 else "  ─"
            if pp >= 0 and de >= 0:
                d_s = f"{de - pp:+.4f}"
            else:
                d_s = "  ─"
            # 缩写维度名以保持对齐
            short = dim.replace("_consistency", "_con").replace("_smoothness", "_smo") \
                       .replace("_flickering", "_flk").replace("_quality", "_q") \
                       .replace("_degree", "_deg")
            print(f"  │ {short:<26s} {pp_s:>8s} {de_s:>8s} {d_s:>8s} │")
        print(f"  └{'─' * 56}┘")

    return row_result


def print_vbench_summary(args, vbench_all_results, dimensions):
    """汇总全部视频的 VBench 分数，打印对比表格并保存 JSON。"""
    if not vbench_all_results or not dimensions:
        return

    print(f"\n{'=' * 72}")
    print(f"  VBench Summary: ProPainter vs DiffuEraser ({len(vbench_all_results)} video(s))")
    print(f"  {'Dimension':<30s}  {'ProPainter':>12s}  {'DiffuEraser':>12s}  {'Δ (DE-PP)':>10s}")
    print(f"  {'-' * 68}")

    pp_avgs, de_avgs = [], []
    serializable = {"propainter": {}, "diffueraser": {}, "per_video": []}

    for dim in dimensions:
        pp_vals = [r["propainter"].get(dim, -1.0) for r in vbench_all_results if r["propainter"].get(dim, -1.0) >= 0]
        de_vals = [r["diffueraser"].get(dim, -1.0) for r in vbench_all_results if r["diffueraser"].get(dim, -1.0) >= 0]

        pp_mean = sum(pp_vals) / len(pp_vals) if pp_vals else -1.0
        de_mean = sum(de_vals) / len(de_vals) if de_vals else -1.0

        pp_str = f"{pp_mean:.4f}" if pp_mean >= 0 else "N/A"
        de_str = f"{de_mean:.4f}" if de_mean >= 0 else "N/A"

        if pp_mean >= 0 and de_mean >= 0:
            delta_str = f"{de_mean - pp_mean:+.4f}"
            pp_avgs.append(pp_mean)
            de_avgs.append(de_mean)
        else:
            delta_str = "N/A"

        print(f"  {dim:<30s}  {pp_str:>12s}  {de_str:>12s}  {delta_str:>10s}")

        serializable["propainter"][dim] = pp_mean
        serializable["diffueraser"][dim] = de_mean

    # 总平均
    if pp_avgs and de_avgs:
        pp_total = sum(pp_avgs) / len(pp_avgs)
        de_total = sum(de_avgs) / len(de_avgs)
        print(f"  {'-' * 68}")
        print(f"  {'AVERAGE':<30s}  {pp_total:>12.4f}  {de_total:>12.4f}  {de_total - pp_total:>+10.4f}")
        serializable["average"] = {"propainter": pp_total, "diffueraser": de_total}

    print(f"{'=' * 72}\n")

    # per_video 数据
    for r in vbench_all_results:
        serializable["per_video"].append({
            "name": r["name"],
            "propainter": r["propainter"],
            "diffueraser": r["diffueraser"],
        })

    # 保存 JSON
    out_path = os.path.join(args.save_path, args.vbench_output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"  [VBench] Results saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION I — Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Resolve target size
    target_size = None
    if args.height > 0 and args.width > 0:
        w, h = args.width - args.width % 8, args.height - args.height % 8
        target_size = (w, h)
        print(f"[Resolution] Target processing size: {w}×{h}")
    else:
        print("[Resolution] Using original video resolution (no resize)")

    # ------------------------------------------------------------------
    # Load unified YAML (once at startup)
    # ------------------------------------------------------------------
    unified_prompts = {}
    if args.use_text and args.unified_prompt_yaml:
        if os.path.exists(args.unified_prompt_yaml):
            with open(args.unified_prompt_yaml, 'r', encoding='utf-8') as f:
                unified_prompts = yaml.safe_load(f) or {}
            print(f"[Unified YAML] Loaded {len(unified_prompts)} entries from {args.unified_prompt_yaml}")
        else:
            print(f"[WARN] Unified YAML not found: {args.unified_prompt_yaml}")

    # ------------------------------------------------------------------
    # Load models (once)
    # ------------------------------------------------------------------
    device = get_device()
    ckpt = "2-Step"

    print("[Loading DiffuEraser …]")
    video_inpainting_sd = DiffuEraser(
        device,
        args.base_model_path,
        args.vae_path,
        args.diffueraser_path,
        ckpt=ckpt,
        pcm_weights_path=args.pcm_weights_path,
    )

    print("[Loading ProPainter …]")
    propainter_model = Propainter(args.propainter_model_dir, device=device)

    # --- Initial GPU offloading ---
    if args.offload_models:
        print(f"\n[Offload] Enabled. Moving DiffuEraser to CPU (ProPainter stays on GPU).")
        _offload_to_cpu(video_inpainting_sd, "DiffuEraser")
        print(f"[{_gpu_mem_info()}]\n")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    all_rows: List[dict] = []
    total_time = 0.0
    failed = []

    # VBench: 一次性初始化子模块
    vbench_ctx = None
    vbench_all_results: List[dict] = []
    if args.eval_vbench:
        try:
            vbench_ctx = init_vbench(args, device)
        except Exception as e:
            print(f"[VBench WARN] Init failed: {e}. VBench evaluation disabled.")
            vbench_ctx = None

    def _resolve_prompt_for_video(name):
        """Resolve prompt for a specific video.
        Priority: CLI --prompt > unified YAML > per-video YAML > empty.

        Returns: (prompt, n_prompt, text_guidance_scale)
        """
        prompt = ""
        n_prompt = ""
        yaml_scale = 2.0

        # CLI override (global, for single-video or debug)
        if args.prompt:
            prompt = args.prompt
        if args.n_prompt:
            n_prompt = args.n_prompt

        # Unified YAML (per-video entry, flat format)
        if not prompt and name in unified_prompts:
            entry = unified_prompts[name]
            p = entry.get('prompt', '')
            prompt = p[0] if isinstance(p, list) else str(p) if p else ""
            np_ = entry.get('n_prompt', '')
            n_prompt = np_[0] if isinstance(np_, list) else str(np_) if np_ else ""
            yaml_scale = entry.get('text_guidance_scale', 2.0)

        # Per-video YAML fallback
        if not prompt and args.prompt_root:
            yaml_path = os.path.join(args.prompt_root, f"{name}.yaml")
            if os.path.exists(yaml_path):
                prompt, n_prompt, yaml_scale = load_prompt_from_yaml(yaml_path)

        # CLI --text_guidance_scale overrides YAML scale
        final_scale = args.text_guidance_scale if args.text_guidance_scale is not None else yaml_scale

        return prompt, n_prompt, final_scale

    def _process_one(name, video_path, mask_path, out_dir, is_sequence):
        """Process a single video entry."""
        nonlocal total_time

        temp_video_path = video_path
        temp_mask_path  = mask_path

        # Convert frame sequences → temp video (at target resolution)
        if is_sequence:
            temp_video_path = os.path.join(out_dir, "_temp_input.mp4")
            temp_mask_path  = mask_path
            print(f"  Converting frame sequences → temp video …")
            frames_to_video(video_path, temp_video_path, fps=args.comparison_fps, size=target_size)

        # Resolve prompt for this video
        prompt, n_prompt, text_guidance_scale = _resolve_prompt_for_video(name)
        if args.use_text and prompt:
            print(f"  [Text] prompt='{prompt[:60]}{'...' if len(prompt) > 60 else ''}' scale={text_guidance_scale}")

        priori_path, pred_path, priori_frames, dt = process_single_video(
            args,
            video_inpainting_sd,
            propainter_model,
            temp_video_path,
            temp_mask_path,
            out_dir,
            name,
            prompt=prompt,
            n_prompt=n_prompt,
            text_guidance_scale=text_guidance_scale,
        )
        total_time += dt

        # Cleanup temp files
        if is_sequence:
            if os.path.isfile(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass

        torch.cuda.empty_cache()
        return priori_path, pred_path, priori_frames

    # -------- Batch mode --------
    if args.dataset and args.video_root and args.mask_root:
        if args.dataset == "davis":
            video_list = get_video_list_davis(args.video_root, args.mask_root)
        elif args.dataset == "youtube-vos":
            video_list = get_video_list_youtube_vos(args.video_root, args.mask_root)
        else:
            video_list = get_video_list_custom(args.video_root, args.mask_root)

        if not video_list:
            raise RuntimeError("No videos found. Check --video_root / --mask_root.")

        print(f"\n{'=' * 60}")
        print(f"Batch mode: {args.dataset}  |  {len(video_list)} video(s)")
        print(f"{'=' * 60}\n")

        for idx, info in enumerate(video_list):
            name = info["name"]
            print(f"\n[{idx + 1}/{len(video_list)}] {name}")
            print("-" * 40)

            out_dir = os.path.join(args.save_path, name)
            os.makedirs(out_dir, exist_ok=True)

            try:
                priori_path, pred_path, priori_frames = _process_one(
                    name, info["video_path"], info["mask_path"], out_dir, info["is_sequence"],
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                failed.append(name)
                continue

            # Comparison video (4-in-1 from frames)
            if args.save_comparison and priori_frames:
                try:
                    in_frames = _load_frames_rgb(info["video_path"])
                    mask_frames = _load_mask_frames_gray(info["mask_path"])
                    de_frames = _load_frames_rgb(pred_path)

                    comp_4in1_path = os.path.join(out_dir, "comparison_4in1.mp4")
                    create_comparison_video_from_frames(
                        in_frames, mask_frames, priori_frames, de_frames,
                        comp_4in1_path, fps=args.comparison_fps
                    )
                except Exception as e:
                    print(f"  [WARN] Could not create 4-in-1 comparison: {e}")

            all_rows.append({"name": name, "pred_video": pred_path, "priori_video": priori_path})

            # VBench per-video evaluation
            if vbench_ctx:
                try:
                    vb_row = evaluate_single_video_vbench(
                        vbench_ctx, name, priori_path, pred_path, out_dir, device
                    )
                    vbench_all_results.append(vb_row)
                except Exception as e:
                    print(f"  [VBench] {name} failed: {e}")

            print(f"[Progress] {idx + 1}/{len(video_list)} videos completed ({(idx + 1) * 100 // len(video_list)}%)\n")

    # -------- Single video mode --------
    else:
        name = "single_video"
        out_dir = args.save_path

        priori_path, pred_path, priori_frames = _process_one(
            name, args.input_video, args.input_mask, out_dir, is_sequence=False,
        )

        # Comparison video (4-in-1 from frames)
        if args.save_comparison and priori_frames:
            try:
                in_frames = _load_frames_rgb(args.input_video)
                mask_frames = _load_mask_frames_gray(args.input_mask)
                de_frames = _load_frames_rgb(pred_path)

                comp_4in1_path = os.path.join(out_dir, "comparison_4in1.mp4")
                create_comparison_video_from_frames(
                    in_frames, mask_frames, priori_frames, de_frames,
                    comp_4in1_path, fps=args.comparison_fps
                )
            except Exception as e:
                print(f"  [WARN] Could not create 4-in-1 comparison: {e}")

        all_rows.append({"name": name, "pred_video": pred_path, "priori_video": priori_path})

        # VBench per-video evaluation
        if vbench_ctx:
            try:
                vb_row = evaluate_single_video_vbench(
                    vbench_ctx, name, priori_path, pred_path, out_dir, device
                )
                vbench_all_results.append(vb_row)
            except Exception as e:
                print(f"  [VBench] {name} failed: {e}")

    # ------------------------------------------------------------------
    # VBench Summary (final averages)
    # ------------------------------------------------------------------
    if vbench_ctx and vbench_all_results:
        try:
            print_vbench_summary(args, vbench_all_results, vbench_ctx["dimensions"])
        except Exception as e:
            print(f"\n[VBench ERROR] Summary failed: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = {"mode": "inference_only", "count": len(all_rows), "items": all_rows}

    summary_path = os.path.join(args.save_path, args.summary_out)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  DONE — {summary['count']} video(s) processed")
    if failed:
        print(f"  Failed: {failed}")
    print(f"  Total inference time: {total_time:.1f}s")
    print(f"  Summary → {summary_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()