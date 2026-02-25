# -*- coding: utf-8 -*-
"""
run_BR.py — BR (Background Restoration) 主程序，对齐 OR Pipeline + 无损评测

Goal: ProPainter inference + DiffuEraser evaluation with hard binary mask compositing:
      comp = pred * mask + gt * (1 - mask)

Usage:
CUDA_VISIBLE_DEVICES=0 python run_BR.py \
  --dataset davis --video_root /path/to/JPEGImages_432_240/ \
  --mask_root /path/to/test_masks/ --gt_root /path/to/JPEGImages_432_240/ \
  --save_path results_davis --input_size 432x240 --compute_metrics

If --input_size is not specified, original video resolution will be used.
"""

import os
import sys

# Must set environment variables BEFORE importing transformers/diffusers
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import logging

# Suppress all warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers.models.modeling_utils").setLevel(logging.ERROR)

import argparse
import yaml
from pathlib import Path
from time import time
import shutil
import tempfile
import cv2
import numpy as np
import torch

from propainter.inference import Propainter
from diffueraser.diffueraser import DiffuEraser


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# ---------------------------------------------------------------------------
# GPU offloading helpers (aligned with run_OR.py)
# ---------------------------------------------------------------------------
def _offload_to_cpu(model, label="Model"):
    """Move model to CPU to free GPU memory."""
    if model is None:
        return
    if hasattr(model, 'pipeline'):
        model.pipeline.to("cpu")
    elif hasattr(model, 'to'):
        model.to("cpu")
    elif hasattr(model, 'pipe'):
        model.pipe.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  [Offload] {label} → CPU")

def _load_to_gpu(model, device, label="Model"):
    """Move model back to GPU for inference."""
    if model is None:
        return
    if hasattr(model, 'pipeline'):
        model.pipeline.to(device, torch.float16)
    elif hasattr(model, 'to'):
        model.to(device)
    elif hasattr(model, 'pipe'):
        model.pipe.to(device, torch.float16)
    print(f"  [Load]    {label} → GPU")

def _gpu_mem_info():
    if not torch.cuda.is_available():
        return "CPU mode"
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved() / 1e9
    return f"GPU mem: {a:.1f}G alloc / {r:.1f}G reserved"

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_prompt_from_yaml(yaml_path: str):
    """从 YAML 加载 prompt 和 n_prompt 配置。
    
    YAML 格式:
        prompt:
          - "a sunlit park path with green grass"
        n_prompt:
          - "blurry, artifacts"
        text_guidance_scale: 2.0
    """
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

def parse_input_size(size_str):
    """Parse input size string like '432x240' or '432,240' into (width, height)."""
    if size_str is None:
        return None, None
    size_str = size_str.strip()
    for sep in ['x', 'X', ',', '*']:
        if sep in size_str:
            parts = size_str.split(sep)
            if len(parts) == 2:
                try:
                    w, h = int(parts[0].strip()), int(parts[1].strip())
                    return w, h
                except ValueError:
                    pass
    raise ValueError(f"Invalid input_size format: '{size_str}'. Use 'WxH' (e.g., '432x240')")

def list_video_names(video_root: Path):
    return [p.name for p in sorted(video_root.iterdir()) 
            if p.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in p.iterdir())]

def load_rgb_frames(frames_dir: Path, max_frames: int = -1):
    files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if max_frames and max_frames > 0: files = files[:max_frames]
    frames = [cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB).astype(np.uint8) 
              for fp in files if cv2.imread(str(fp)) is not None]
    if not frames: raise ValueError(f"No frames found in {frames_dir}")
    return frames

def load_gray_masks(masks_dir: Path, max_frames: int = -1):
    files = sorted([p for p in masks_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if max_frames and max_frames > 0: files = files[:max_frames]
    masks = [cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE).astype(np.uint8) 
             for fp in files if cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE) is not None]
    if not masks: raise ValueError(f"No masks found in {masks_dir}")
    return masks

def resize_frames(frames, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    """Resize a list of frames to target width and height."""
    if target_w is None or target_h is None:
        return frames
    return [cv2.resize(f, (target_w, target_h), interpolation=interpolation) for f in frames]

def resize_masks(masks, target_w, target_h):
    """Resize a list of masks to target width and height using NEAREST interpolation."""
    if target_w is None or target_h is None:
        return masks
    return [cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST) for m in masks]

def save_frames_to_dir(frames, out_dir: Path, prefix="frame", ext=".png"):
    """Save frames to a directory for model input."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        out_path = out_dir / f"{prefix}_{i:05d}{ext}"
        cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return out_dir

def save_masks_to_dir(masks, out_dir: Path, prefix="mask", ext=".png", invert=False):
    """Save masks to a directory for model input.
    
    Args:
        invert: If True, invert the mask (0->255, 255->0) before saving.
                Use this when source mask has BLACK=hole but model expects WHITE=hole.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, mask in enumerate(masks):
        out_path = out_dir / f"{prefix}_{i:05d}{ext}"
        save_mask = 255 - mask if invert else mask
        cv2.imwrite(str(out_path), save_mask)
    return out_dir

def normalize_length(seq, n):
    return seq[:n] if len(seq) >= n else seq + [seq[-1]] * (n - len(seq))

def ensure_same_hw(rgb_frames, gt_frames, masks):
    n = min(len(rgb_frames), len(gt_frames), len(masks))
    rgb_frames, gt_frames, masks = rgb_frames[:n], gt_frames[:n], masks[:n]
    H, W = gt_frames[0].shape[:2]
    for i in range(n):
        if gt_frames[i].shape[:2] != (H, W): raise ValueError("GT frames have inconsistent sizes")
        if rgb_frames[i].shape[:2] != (H, W): raise ValueError(f"Input frame {i} size mismatch")
        if masks[i].shape[:2] != (H, W): raise ValueError(f"Mask {i} size mismatch")
    return rgb_frames, gt_frames, masks

def composite_with_gt(pred_frames, gt_frames, masks, mask_inverse=False):
    """comp = pred * mask + gt * (1-mask), where mask==1 is HOLE
    
    Hard binary mask — no blur, no feathering. Best for PSNR/SSIM.
    
    Args:
        mask_inverse: If True, treat BLACK (0) as hole. Default False (WHITE as hole).
    """
    n = min(len(pred_frames), len(gt_frames), len(masks))
    out, masks01 = [], []
    for i in range(n):
        if mask_inverse:
            m01 = (masks[i] == 0).astype(np.uint8)
        else:
            m01 = (masks[i] > 0).astype(np.uint8)
        m3 = np.repeat(m01[:, :, None], 3, axis=2)
        out.append((pred_frames[i].astype(np.uint8) * m3 + gt_frames[i].astype(np.uint8) * (1 - m3)).astype(np.uint8))
        masks01.append(m01)
    return out, masks01

def save_mp4(frames_rgb, out_mp4, fps=24, codec='mp4v'):
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*codec), fps, (W, H))
    for fr in frames_rgb: vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()

# =========================
# Output Formatting
# =========================
def fmt(x, w=10, prec=4):
    if x is None: return f"{'N/A':>{w}}"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)): return f"{'inf':>{w}}"
    return f"{x:>{w}.{prec}f}"

def print_table_header():
    print(f"{'Method':<12} | {'PSNR':>10} | {'SSIM':>10} | {'LPIPS':>10} | {'Ewarp':>10} | {'AS':>10} | {'IS':>10} | {'VFID':>10}")
    print("-" * 100)

def print_table_row(name, psnr, ssim, lpips, ewarp, a_s, i_s, vfid):
    print(f"{name:<12} | {fmt(psnr)} | {fmt(ssim)} | {fmt(lpips)} | {fmt(ewarp)} | {fmt(a_s)} | {fmt(i_s)} | {fmt(vfid)}")


def _ffmpeg_remux_h264(src_mp4, dst_mp4):
    """Re-encode mp4v → H.264 via ffmpeg for universal player compatibility."""
    import subprocess, shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return  # ffmpeg not available, keep mp4v file
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src_mp4),
             "-c:v", "libx264", "-crf", "18", "-preset", "fast",
             "-pix_fmt", "yuv420p", "-an", str(dst_mp4)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=120,
        )
    except Exception:
        pass  # keep original file on failure


def create_comparison_video_from_frames(in_frames, mask_frames, pp_frames, de_frames,
                                        output_path, fps=24):
    """Create 2x2 comparison video from frame lists (RGB numpy arrays).

    Layout:  Input      | Mask
             ProPainter | DiffuEraser

    Writes with cv2 (mp4v), then re-encodes to H.264 via ffmpeg for Windows compatibility.
    """
    n = min(len(in_frames), len(mask_frames), len(pp_frames), len(de_frames))
    if n == 0:
        return

    h, w = in_frames[0].shape[:2]
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write with mp4v first (always works)
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

    # Re-encode to H.264 for Windows compatibility
    _ffmpeg_remux_h264(tmp_path, str(out_path))
    if os.path.exists(str(out_path)) and os.path.getsize(str(out_path)) > 0:
        os.remove(tmp_path)
    else:
        # ffmpeg failed, keep mp4v file
        os.rename(tmp_path, str(out_path))
    print(f"  [Comparison] Saved 4-in-1 video: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--video_root', type=str, required=True)
    parser.add_argument('--mask_root', type=str, required=True)
    parser.add_argument('--gt_root', type=str, required=True)
    parser.add_argument('--prompt_root', type=str, default=None,
                        help="Root directory for per-video prompt YAML files (e.g. prompt_cache)")
    parser.add_argument('--unified_prompt_yaml', type=str, default=None,
                        help="Path to a unified YAML file containing all video prompts (from generate_captions.py)")
    parser.add_argument('--use_text', action='store_true',
                        help="Enable text guidance using prompts from YAML files")
    parser.add_argument('--text_guidance_scale', type=float, default=None,
                        help="Classifier-free guidance scale for text (overrides YAML value)")

    parser.add_argument('--save_path', type=str, default='results_davis')
    parser.add_argument('--input_size', type=str, default=None,
                        help="Input image size as 'WxH' (e.g., '432x240'). If not specified, use original resolution.")
    parser.add_argument('--height', type=int, default=-1, help="[Deprecated] Use --input_size instead")
    parser.add_argument('--width', type=int, default=-1, help="[Deprecated] Use --input_size instead")
    parser.add_argument('--video_length', type=int, default=-1)
    parser.add_argument('--ref_stride', type=int, default=10)
    parser.add_argument('--neighbor_length', type=int, default=20)
    parser.add_argument('--subvideo_length', type=int, default=80)
    parser.add_argument('--mask_dilation_iter', type=int, default=0)
    parser.add_argument('--mask_inverse', action='store_true',
                        help="Invert mask: use when BLACK=hole (e.g., RORD dataset). Default assumes WHITE=hole (DAVIS).")
    parser.add_argument('--save_comparison', action='store_true')
    parser.add_argument('--compute_metrics', action='store_true')
    parser.add_argument('--no_metrics', action='store_true',
                        help="Disable metrics computation (overrides --compute_metrics)")
    parser.add_argument('--base_model_path', type=str, default='/home/hj/DiffuEraser1/weights/stable-diffusion-v1-5')
    parser.add_argument('--vae_path', type=str, default='/home/hj/DiffuEraser1/weights/sd-vae-ft-mse')
    parser.add_argument('--diffueraser_path', type=str, default='/home/hj/DiffuEraser1/weights/diffuEraser')
    parser.add_argument('--propainter_model_dir', type=str, default='/home/hj/DiffuEraser1/weights/propainter')
    parser.add_argument('--pcm_weights_path', type=str, default='/home/hj/DiffuEraser1/weights/PCM_Weights')
    parser.add_argument('--i3d_model_path', type=str, default='/home/hj/DiffuEraser1/weights/i3d_rgb_imagenet.pt')
    parser.add_argument('--raft_model_path', type=str, default='/home/hj/DiffuEraser1/weights/propainter/raft-things.pth')

    # ========== Anchor Frame Strategy ==========

    parser.add_argument('--skip_diffueraser', action='store_true',
                        help="BR only: skip DiffuEraser and report ProPainter only.")

    # ========== GPU Offloading ==========
    parser.add_argument('--offload_models', action='store_true',
                        help="Offload DiffuEraser to CPU when not in use (saves VRAM)")

    args = parser.parse_args()

    # Handle --no_metrics override
    if args.no_metrics:
        args.compute_metrics = False

    # Parse input size
    if args.input_size:
        target_w, target_h = parse_input_size(args.input_size)
    elif args.width > 0 and args.height > 0:
        target_w, target_h = args.width, args.height
    else:
        target_w, target_h = None, None  # Use original resolution

    video_root, mask_root, gt_root = Path(args.video_root), Path(args.mask_root), Path(args.gt_root)
    save_root = Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    names = list_video_names(video_root)
    if not names: raise ValueError(f"No videos found under: {video_root}")

    print(f"\n{'='*60}\nAligned Lossless Evaluation\n{'='*60}")
    print(f"Videos: {video_root} ({len(names)} found)\nMasks:  {mask_root}\nGT:     {gt_root}")
    if target_w and target_h:
        print(f"Input Size: {target_w}x{target_h}")
    else:
        print(f"Input Size: Original resolution (auto)")
    print(f"Mask Convention: {'BLACK=hole (RORD)' if args.mask_inverse else 'WHITE=hole (DAVIS)'}")

    if args.offload_models:
        print(f"GPU Offloading: Enabled")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Metrics (lazy) ----
    metrics = None
    if args.compute_metrics:
        from metrics import MetricsCalculator
        metrics = MetricsCalculator(device=device, i3d_model_path=args.i3d_model_path,
                                    propainter_model_dir=args.propainter_model_dir,
                                    raft_model_path=args.raft_model_path)

    # ---- Load models once (reuse across videos) ----
    print("[Loading ProPainter …]")
    pp = Propainter(args.propainter_model_dir, device)

    de = None
    if not args.skip_diffueraser:
        print("[Loading DiffuEraser …]")
        _prev_level = logging.root.manager.disable
        logging.disable(logging.WARNING)
        de = DiffuEraser(device, args.base_model_path, args.vae_path, args.diffueraser_path,
                         pcm_weights_path=args.pcm_weights_path)
        logging.disable(_prev_level)



    if args.compute_metrics:
        print("[Metrics enabled]")
    else:
        print("[Metrics disabled]")

    # ---- GPU Offloading: initial state ----
    if args.offload_models:
        print(f"\n[Offload] Enabled. Moving DiffuEraser to CPU (ProPainter stays on GPU).")
        _offload_to_cpu(de, "DiffuEraser")
        print(f"  [{_gpu_mem_info()}]")

    # Accumulators
    pp_ori_acts, pp_comp_acts, de_ori_acts, de_comp_acts = [], [], [], []
    acc_pp = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'ewarp': 0.0, 'as': 0.0, 'is': 0.0, 'n': 0}
    acc_de = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'ewarp': 0.0, 'as': 0.0, 'is': 0.0, 'n': 0}

    # ---- Pre-load unified prompt YAML if provided ----
    unified_prompts = {}
    if args.use_text and args.unified_prompt_yaml:
        if os.path.exists(args.unified_prompt_yaml):
            with open(args.unified_prompt_yaml, 'r', encoding='utf-8') as f:
                unified_prompts = yaml.safe_load(f) or {}
            print(f"[Unified YAML] Loaded {len(unified_prompts)} entries from {args.unified_prompt_yaml}")
        else:
            print(f"[WARN] Unified YAML not found: {args.unified_prompt_yaml}")

    print(f"\n{'='*60}\nBatch mode: {args.dataset}  |  {len(names)} video(s)\n{'='*60}")

    for vid_i, name in enumerate(names, 1):
        print(f"\n\n[{vid_i}/{len(names)}] {name}")
        print("-" * 40)

        vdir, mdir, gdir = video_root / name, mask_root / name, gt_root / name
        if not all(d.exists() for d in [vdir, mdir, gdir]):
            print(f"  Skip: missing {'video' if not vdir.exists() else 'mask' if not mdir.exists() else 'GT'} dir")
            continue

        # Load frames
        in_frames = load_rgb_frames(vdir, args.video_length)
        gt_frames = load_rgb_frames(gdir, args.video_length)
        masks = load_gray_masks(mdir, args.video_length)

        # ---- Load Prompt (Text Guidance) ----
        prompt, n_prompt, text_guidance_scale = "", "", 7.5
        if args.use_text:
            # Priority: unified YAML > per-video YAML
            loaded_from = None
            if name in unified_prompts:
                entry = unified_prompts[name]
                p = entry.get('prompt', '')
                prompt = p[0] if isinstance(p, list) else str(p) if p else ""
                np_ = entry.get('n_prompt', '')
                n_prompt = np_[0] if isinstance(np_, list) else str(np_) if np_ else ""
                yaml_scale = entry.get('text_guidance_scale', 2.0)
                loaded_from = "unified YAML"
            elif args.prompt_root:
                yaml_path = os.path.join(args.prompt_root, f"{name}.yaml")
                if os.path.exists(yaml_path):
                    prompt, n_prompt, yaml_scale = load_prompt_from_yaml(yaml_path)
                    loaded_from = os.path.basename(yaml_path)
                else:
                    print(f"  [Text Guidance] WARN: YAML not found at {yaml_path}")
            
            if loaded_from:
                # Prioritize CLI argument if provided
                if args.text_guidance_scale is not None:
                    text_guidance_scale = args.text_guidance_scale
                else:
                    text_guidance_scale = yaml_scale
                
                print(f"  [Text Guidance] Loaded prompt from {loaded_from}")
                print(f"    Content: {prompt[:60]}...")
                print(f"    Scale:   {text_guidance_scale} (Source: {'CLI' if args.text_guidance_scale is not None else 'YAML'})")
            elif not args.prompt_root and not unified_prompts:
                print(f"  [Text Guidance] Enabled but no prompt source specified. Using empty prompt.")
                if args.text_guidance_scale is not None:
                    text_guidance_scale = args.text_guidance_scale
        
        orig_h, orig_w = in_frames[0].shape[:2]
        
        # Temp directories for resized frames
        temp_dir = None
        model_vdir, model_mdir = vdir, mdir
        need_temp_dir = (target_w and target_h) or args.mask_inverse
        
        # Resize if input_size is specified
        if target_w and target_h:
            in_frames = resize_frames(in_frames, target_w, target_h)
            gt_frames = resize_frames(gt_frames, target_w, target_h)
            masks = resize_masks(masks, target_w, target_h)
        
        # Create temp directory if needed
        if need_temp_dir:
            temp_dir = Path(tempfile.mkdtemp(prefix=f"diffueraser_{name}_"))
            model_vdir = save_frames_to_dir(in_frames, temp_dir / "frames")
            model_mdir = save_masks_to_dir(masks, temp_dir / "masks", invert=args.mask_inverse)
        
        n = min(len(in_frames), len(gt_frames), len(masks))
        in_frames, gt_frames, masks = in_frames[:n], gt_frames[:n], normalize_length(masks[:n], n)
        in_frames, gt_frames, masks = ensure_same_hw(in_frames, gt_frames, masks)
        
        proc_h, proc_w = in_frames[0].shape[:2]

        # ---- GPU Offloading: ensure correct state ----
        if args.offload_models:
            _offload_to_cpu(de, "DiffuEraser")
            print(f"  [{_gpu_mem_info()}]")

        # ---- ProPainter ----
        print("  ProPainter...", end=" ", flush=True)
        t0 = time()
        pp_frames = pp.forward(
            video=str(model_vdir), mask=str(model_mdir), output_path=str(save_root / name / "propainter.mp4"),
            resize_ratio=1.0, video_length=args.video_length, 
            height=-1, width=-1,
            mask_dilation=args.mask_dilation_iter, ref_stride=args.ref_stride,
            neighbor_length=args.neighbor_length, subvideo_length=args.subvideo_length,
            raft_iter=20, save_fps=24, save_frames=False, fp16=True, return_frames=True)
        print(f"{time()-t0:.1f}s")
        pp_comp, masks01 = composite_with_gt(pp_frames, gt_frames, masks, mask_inverse=args.mask_inverse)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



        # ---- DiffuEraser ----
        if de is not None:
            if args.offload_models:
                _load_to_gpu(de, device, "DiffuEraser")
                print(f"  [{_gpu_mem_info()}]")

            print("  DiffuEraser...", end=" ", flush=True)
            t1 = time()
            de_frames = de.forward(
                validation_image=str(model_vdir), validation_mask=str(model_mdir), priori="__unused__",
                output_path=str(save_root / name / "diffueraser.mp4"),
                max_img_size=max(proc_w, proc_h) + 100,
                video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter,
                nframes=22, seed=None, blended=False, priori_frames=pp_frames, return_frames=True,
                guidance_scale=text_guidance_scale, prompt=prompt, n_prompt=n_prompt)
            print(f"{time()-t1:.1f}s  Inference time: {time()-t1:.2f}s")
            de_comp, _ = composite_with_gt(de_frames, gt_frames, masks, mask_inverse=args.mask_inverse)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if args.offload_models:
                _offload_to_cpu(de, "DiffuEraser")
        else:
            de_comp = pp_comp
        
        # Cleanup temp directory
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)

        # ---- Output Artifacts ----
        out_dir = save_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.save_comparison:
            save_mp4(pp_comp, out_dir / "propainter_comp.mp4")
            if de is not None:
                save_mp4(de_comp, out_dir / "diffueraser_comp.mp4")

        # 4-in-1 comparison video
        create_comparison_video_from_frames(
            in_frames, masks, pp_comp, de_comp,
            str(out_dir / "comparison_4in1.mp4"), fps=24
        )

        # ---- Metrics ----
        if metrics:
            print("  Computing metrics...")
            pp_res = metrics.compute_video_metrics(pp_comp, gt_frames, masks=masks, compute_vfid=True)
            if 'ori_i3d_act' in pp_res:
                pp_ori_acts.append(pp_res['ori_i3d_act'])
                pp_comp_acts.append(pp_res['comp_i3d_act'])
            acc_pp['n'] += 1
            for k in ['psnr', 'ssim', 'lpips', 'as', 'is']:
                acc_pp[k] += float(pp_res.get(f'{k}_mean', 0.0))
            if pp_res.get('ewarp', -1.0) >= 0:
                acc_pp['ewarp'] += pp_res['ewarp']

            if de is not None:
                de_res = metrics.compute_video_metrics(de_comp, gt_frames, masks=masks, compute_vfid=True)
                if 'ori_i3d_act' in de_res:
                    de_ori_acts.append(de_res['ori_i3d_act'])
                    de_comp_acts.append(de_res['comp_i3d_act'])
                acc_de['n'] += 1
                for k in ['psnr', 'ssim', 'lpips', 'as', 'is']:
                    acc_de[k] += float(de_res.get(f'{k}_mean', 0.0))
                if de_res.get('ewarp', -1.0) >= 0:
                    acc_de['ewarp'] += de_res['ewarp']

            # Print current video results
            print(f"\n  Metrics for {name}:")
            print_table_header()
            print_table_row("ProPainter", pp_res.get('psnr_mean'), pp_res.get('ssim_mean'),
                          pp_res.get('lpips_mean'), pp_res.get('ewarp'), pp_res.get('as_mean'), pp_res.get('is_mean'), None)
            if de is not None:
                print_table_row("DiffuEraser", de_res.get('psnr_mean'), de_res.get('ssim_mean'),
                              de_res.get('lpips_mean'), de_res.get('ewarp'), de_res.get('as_mean'), de_res.get('is_mean'), None)

            # Running average
            avg = lambda acc, k: acc[k] / max(1, acc['n'])
            pp_vfid = metrics.compute_final_vfid(pp_ori_acts, pp_comp_acts) if len(pp_ori_acts) >= 2 else None
            pp_vfid = None if pp_vfid and pp_vfid < 0 else pp_vfid
            de_vfid = None
            if de is not None and len(de_ori_acts) >= 2:
                de_vfid = metrics.compute_final_vfid(de_ori_acts, de_comp_acts)
                de_vfid = None if de_vfid and de_vfid < 0 else de_vfid

            print(f"\n  Running Average ({acc_pp['n']} videos):")
            print_table_header()
            print_table_row("ProPainter", avg(acc_pp,'psnr'), avg(acc_pp,'ssim'), avg(acc_pp,'lpips'),
                          avg(acc_pp,'ewarp'), avg(acc_pp,'as'), avg(acc_pp,'is'), pp_vfid)
            if de is not None:
                print_table_row("DiffuEraser", avg(acc_de,'psnr'), avg(acc_de,'ssim'), avg(acc_de,'lpips'),
                              avg(acc_de,'ewarp'), avg(acc_de,'as'), avg(acc_de,'is'), de_vfid)
        else:
            print(f"  [INFO] {name}: saved (metrics disabled)")

        print(f"[Progress] {vid_i}/{len(names)} videos completed ({100*vid_i//len(names)}%)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final VFID
    if metrics:
        print(f"\n{'='*60}\nFinal Dataset VFID\n{'='*60}")
        print(f"ProPainter:  {metrics.compute_final_vfid(pp_ori_acts, pp_comp_acts):.6f}")
        if not args.skip_diffueraser and len(de_ori_acts) >= 2:
            print(f"DiffuEraser: {metrics.compute_final_vfid(de_ori_acts, de_comp_acts):.6f}")
        elif args.skip_diffueraser:
            print("DiffuEraser: (skipped)")

    print("\nDone.")

if __name__ == '__main__':
    main()