# -*- coding: utf-8 -*-
"""
compare_all.py  ──  GT vs Experiment single-inference comparison

Supports BOTH BR and OR datasets with the same code.
Uses BR's zero-compression approach: memory frames, return_frames, priori_frames.

CLI flags control blending and mask dilation:
  --blended           Enable GaussianBlur feathered mask blending
  --mask_dilation_iter Mask dilation iterations (0 = no dilation)

GT is OPTIONAL:
  - With --gt_root: computes pixel metrics (PSNR/SSIM/LPIPS/Ewarp/AS/IS/VFID) + side-by-side
  - Without --gt_root: only VBench (if --eval)

Only ONE DiffuEraser inference per video (using the experiment's gs/prompt/blend/dilation).
"""

import os
import sys
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["DIFFUSERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("libs.unet_motion_model").setLevel(logging.ERROR)

import argparse
import json
import csv
import textwrap
import shutil
import tempfile
import gc
from pathlib import Path
from time import time

import cv2
import numpy as np
import torch
import yaml

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INFERENCE_DIR))

# VBench
VBENCH_ROOT = Path("/home/hj/VBench")
if str(VBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VBENCH_ROOT))

# ALL cases use BR's modules (supports return_frames + priori_frames)
from propainter.inference import Propainter
from diffueraser.diffueraser import DiffuEraser


# =====================================================================
#  Constants & Utilities
# =====================================================================

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_video_names(video_root, mask_root=None):
    """List video names. If mask_root given, return intersection."""
    vr = Path(video_root)
    v_names = {p.name for p in vr.iterdir() if p.is_dir()}
    if mask_root:
        mr = Path(mask_root)
        m_names = {p.name for p in mr.iterdir() if p.is_dir()}
        v_names = v_names & m_names
    return sorted(v_names)


def load_rgb_frames(frames_dir, max_frames=-1):
    frames_dir = Path(frames_dir)
    files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    return [cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB).astype(np.uint8)
            for fp in files if cv2.imread(str(fp)) is not None]


def load_gray_masks(masks_dir, max_frames=-1):
    masks_dir = Path(masks_dir)
    files = sorted([p for p in masks_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if max_frames and max_frames > 0:
        files = files[:max_frames]
    return [cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            for fp in files if cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE) is not None]


def save_frames_to_dir(frames, out_dir, prefix="frame", ext=".png"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(out_dir / f"{prefix}_{i:05d}{ext}"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    return out_dir


def save_masks_to_dir(masks, out_dir, prefix="mask", ext=".png"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(masks):
        cv2.imwrite(str(out_dir / f"{prefix}_{i:05d}{ext}"), m)
    return out_dir


def composite_with_gt(pred_frames, gt_frames, masks):
    """comp = pred * mask + gt * (1-mask), where mask==1 is HOLE (WHITE)."""
    n = min(len(pred_frames), len(gt_frames), len(masks))
    out, masks01 = [], []
    for i in range(n):
        m01 = (masks[i] > 0).astype(np.uint8)
        m3 = np.repeat(m01[:, :, None], 3, axis=2)
        out.append((pred_frames[i].astype(np.uint8) * m3 +
                     gt_frames[i].astype(np.uint8) * (1 - m3)).astype(np.uint8))
        masks01.append(m01)
    return out, masks01


def save_mp4(frames_rgb, out_mp4, fps=24):
    p = Path(out_mp4)
    p.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for fr in frames_rgb:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    _ffmpeg_remux(str(p))


def _ffmpeg_remux(path):
    import subprocess, shutil as _sh
    ffmpeg = _sh.which("ffmpeg")
    if not ffmpeg:
        return
    tmp = path + ".h264.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", path, "-c:v", "libx264", "-crf", "18",
             "-preset", "fast", "-pix_fmt", "yuv420p", "-an", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
    except Exception:
        pass


def fmt(x, w=10, prec=4):
    if x is None:
        return f"{'N/A':>{w}}"
    return f"{x:>{w}.{prec}f}"


# =====================================================================
#  Metrics (auto-enabled when GT is available + --eval)
# =====================================================================

def get_metrics_calc(args, device, has_gt):
    """Load metrics calculator if GT exists and --eval is on."""
    if not args.eval or not has_gt:
        return None
    try:
        from metrics import MetricsCalculator
        i3d = args.i3d_model_path if args.i3d_model_path and os.path.exists(args.i3d_model_path) else None
        raft = args.raft_model_path if args.raft_model_path and os.path.exists(args.raft_model_path) else None
        mc = MetricsCalculator(device=device, i3d_model_path=i3d, raft_model_path=raft)
        return mc
    except Exception as e1:
        # Fallback: try without i3d/raft args
        try:
            from metrics import MetricsCalculator
            mc = MetricsCalculator(device=device)
            return mc
        except Exception as e2:
            print(f"[WARN] Metrics init failed: {e1} / {e2}")
            return None


# =====================================================================
#  VBench (optional)
# =====================================================================

def init_vbench(args, device):
    from vbench.utils import init_submodules
    dims = list(args.vbench_dimensions)
    if not dims:
        return None
    print(f"\n[VBench] Initializing {len(dims)} dimension(s)")
    submodules_dict = init_submodules(dims, local=False, read_frame=False)
    print("[VBench] Ready.\n")
    return {"dimensions": dims, "submodules": submodules_dict}


def evaluate_vbench_on_video(vbench_ctx, video_path, name, save_dir, device):
    """Run VBench on a single video, return scores dict."""
    import importlib, contextlib, io
    from vbench.utils import save_json

    dims = vbench_ctx["dimensions"]
    submodules = vbench_ctx["submodules"]
    scores = {}

    if not video_path or not os.path.exists(video_path):
        return scores

    info_list = [{"prompt_en": name, "dimension": dims, "video_list": [str(video_path)]}]
    info_path = os.path.join(save_dir, f"_vbench_{os.path.basename(video_path)}.json")
    save_json(info_list, info_path)

    for dim in dims:
        try:
            dim_module = importlib.import_module(f"vbench.{dim}")
            compute_fn = getattr(dim_module, f"compute_{dim}")
            _prev = logging.root.level
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                avg, _ = compute_fn(info_path, device, submodules[dim])
            logging.disable(_prev)
            scores[dim] = float(avg) if not isinstance(avg, bool) else (1.0 if avg else 0.0)
        except Exception as e:
            logging.disable(logging.NOTSET)
            print(f"    [VBench] {dim}: {e}")
            scores[dim] = -1.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        os.remove(info_path)
    except OSError:
        pass
    return scores


# =====================================================================
#  Score items
# =====================================================================

_PIXEL_SCORE_ITEMS = [
    ("PSNR+",  "psnr_mean"),
    ("SSIM+",  "ssim_mean"),
    ("LPIPS-", "lpips_mean"),
    ("Ewarp-", "ewarp"),
    ("AS+",    "as_mean"),
    ("IS+",    "is_mean"),
    ("VFID-",  "vfid"),
]

VBENCH_DIMS = [
    "subject_consistency", "background_consistency",
    "temporal_flickering", "motion_smoothness",
    "aesthetic_quality", "imaging_quality",
]


# =====================================================================
#  Single-Video Inference (single run, NOT BL vs TG)
# =====================================================================

def run_one_video(name, vdir, mdir, gdir, args, pp_model, de_model,
                  prompt, n_prompt, text_guidance_scale,
                  metrics_calc, vbench_ctx, device, out_dir):
    """Run single DiffuEraser inference. Output: comparison.mp4 (GT vs Exp) + comparison.json."""

    result = {"name": name, "metrics": {}, "vbench": {}, "caption": prompt}

    # Load frames
    in_frames = load_rgb_frames(vdir, args.video_length)
    masks = load_gray_masks(mdir, args.video_length)
    n = min(len(in_frames), len(masks))
    in_frames, masks = in_frames[:n], masks[:n]

    gt_frames = None
    has_gt = gdir is not None and Path(gdir).exists()
    if has_gt:
        gt_frames = load_rgb_frames(gdir, args.video_length)
        n = min(n, len(gt_frames))
        in_frames, masks, gt_frames = in_frames[:n], masks[:n], gt_frames[:n]

    # Resize if --height/--width specified
    tgt_h, tgt_w = getattr(args, 'height', -1), getattr(args, 'width', -1)
    if tgt_h > 0 and tgt_w > 0:
        orig_h, orig_w = in_frames[0].shape[:2]
        tgt_h = tgt_h - tgt_h % 8
        tgt_w = tgt_w - tgt_w % 8
        if (orig_h, orig_w) != (tgt_h, tgt_w):
            print(f"  Resize: {orig_w}x{orig_h} -> {tgt_w}x{tgt_h}")
            in_frames = [cv2.resize(f, (tgt_w, tgt_h)) for f in in_frames]
            masks = [cv2.resize(m, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST) for m in masks]
            if gt_frames:
                gt_frames = [cv2.resize(f, (tgt_w, tgt_h)) for f in gt_frames]

    proc_h, proc_w = in_frames[0].shape[:2]

    # Temp dir
    temp_dir = Path(tempfile.mkdtemp(prefix=f"cmpAll_{name}_"))
    model_vdir = save_frames_to_dir(in_frames, temp_dir / "frames")
    model_mdir = save_masks_to_dir(masks, temp_dir / "masks")

    # ── ProPainter ──
    print(f"  ProPainter...", end=" ", flush=True)
    t0 = time()
    pp_frames = pp_model.forward(
        video=str(model_vdir), mask=str(model_mdir),
        output_path=str(temp_dir / "_pp.mp4"),
        resize_ratio=1.0, video_length=args.video_length,
        height=-1, width=-1,
        mask_dilation=args.mask_dilation_iter, ref_stride=args.ref_stride,
        neighbor_length=args.neighbor_length, subvideo_length=args.subvideo_length,
        raft_iter=20, save_fps=24, save_frames=False, fp16=True, return_frames=True)
    print(f"{time()-t0:.1f}s")
    torch.cuda.empty_cache()

    # ── DiffuEraser (single inference with experiment config) ──
    print(f"  DiffuEraser [gs={text_guidance_scale}, blend={args.blended}, "
          f"dilation={args.mask_dilation_iter}]...", end=" ", flush=True)
    t1 = time()
    exp_frames = de_model.forward(
        validation_image=str(model_vdir), validation_mask=str(model_mdir),
        priori="__unused__",
        output_path=str(temp_dir / "_exp.mp4"),
        max_img_size=min(max(proc_w, proc_h) + 100, 1820),
        video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter,
        nframes=22, seed=None,
        blended=args.blended, priori_frames=pp_frames, return_frames=True,
        guidance_scale=text_guidance_scale,
        prompt=prompt, n_prompt=n_prompt)
    print(f"{time()-t1:.1f}s")
    torch.cuda.empty_cache()

    # ── Composite with GT (if available) ──
    if has_gt:
        exp_comp, masks01 = composite_with_gt(exp_frames, gt_frames, masks)
    else:
        exp_comp = exp_frames

    # ── Evaluation (controlled by single --eval flag) ──
    if args.eval:
        # Pixel Metrics (auto-enabled when GT is available)
        if metrics_calc and has_gt:
            print(f"  📊 Pixel Metrics (GT available)...")
            exp_res = metrics_calc.compute_video_metrics(exp_comp, gt_frames, masks=masks, compute_vfid=True)

            result["metrics"] = {
                k: float(v) for k, v in exp_res.items()
                if isinstance(v, (int, float)) and not k.endswith('_act')
            }

            # I3D activations for VFID
            if 'ori_i3d_act' in exp_res:
                result["_ori_act"] = exp_res['ori_i3d_act']
                result["_comp_act"] = exp_res['comp_i3d_act']

        # VBench
        if vbench_ctx:
            print(f"  📊 VBench evaluation...")
            exp_tmp = str(temp_dir / "_exp_vbench.mp4")
            save_mp4(exp_comp, exp_tmp)
            exp_vb = evaluate_vbench_on_video(vbench_ctx, exp_tmp, f"{name}_exp", str(temp_dir), device)
            result["vbench"] = exp_vb

        # ── Pretty-print per-video scores ──
        _print_per_video_scores(result, has_gt)

    # ── Comparison Video (GT vs Experiment, no scores overlay) ──
    if has_gt and gt_frames:
        n_out = min(len(gt_frames), len(exp_comp))
        _create_side_by_side(
            name, gt_frames[:n_out], exp_comp[:n_out],
            str(out_dir / "comparison.mp4"), fps=24,
            tag_line=f"gs={text_guidance_scale} blend={args.blended} dil={args.mask_dilation_iter}"
        )
    else:
        # No GT: just save experiment output as standalone video
        save_mp4(exp_comp, str(out_dir / "experiment.mp4"))
        print(f"  [Save] {name} -> {out_dir / 'experiment.mp4'}")

    # ── Save per-video JSON ──
    saveable = {
        "name": name, "caption": prompt,
        "blended": args.blended, "mask_dilation_iter": args.mask_dilation_iter,
        "text_guidance_scale": text_guidance_scale,
        "metrics": result.get("metrics", {}),
        "vbench": result.get("vbench", {}),
    }
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(saveable, f, ensure_ascii=False, indent=2)

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    gc.collect()
    torch.cuda.empty_cache()

    return result


def _print_per_video_scores(result, has_gt):
    """Pretty-print per-video evaluation scores."""
    m = {}
    m.update(result.get("metrics", {}))
    m.update(result.get("vbench", {}))

    if not m:
        return

    items = []
    if has_gt:
        for label, key in _PIXEL_SCORE_ITEMS:
            if key in m:
                items.append((label, key))
    for dim in VBENCH_DIMS:
        if dim in m:
            short = dim.replace("_consistency", "_con").replace("_smoothness", "_smo") \
                       .replace("_flickering", "_flk").replace("_quality", "_q")
            items.append((short, dim))

    if not items:
        return

    col_w = 12
    hdr = f"  {'Metric':<20s} {'Score':>{col_w}s}"
    sep_line = f"  {'─' * (20 + col_w + 3)}"

    print(sep_line)
    print(hdr)
    print(sep_line)

    for label, key in items:
        v = m.get(key)
        vs = f"{v:.4f}" if isinstance(v, (int, float)) and v >= 0 else "N/A"
        print(f"  {label:<20s} {vs:>{col_w}s}")

    print(sep_line)


# =====================================================================
#  Side-by-Side Comparison Video (GT vs Experiment, clean)
# =====================================================================

def _create_side_by_side(name, gt_frames, exp_frames,
                         output_path, fps=24, tag_line=""):
    """Create clean side-by-side comparison: GT (left) vs Experiment (right).
    No scores, no prompt text overlay. Only simple labels."""
    n = min(len(gt_frames), len(exp_frames))
    if n == 0:
        return

    h_orig, w_orig = gt_frames[0].shape[:2]
    cell_w = min(w_orig, 640)
    cell_h = min(h_orig, 360)
    cell_w -= cell_w % 2
    cell_h -= cell_h % 2
    canvas_w = cell_w * 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.38, cell_w / 750)
    sfs = fs * 0.8
    sft = max(1, int(sfs * 1.5))
    lh = int(20 * fs + 6)

    # Label bar height
    label_h = lh + 4
    total_h = label_h + cell_h
    total_h += total_h % 2

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tmp = output_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_w, total_h))

    for i in range(n):
        canvas = np.zeros((total_h, canvas_w, 3), dtype=np.uint8)

        # Label bar
        canvas[:label_h, :cell_w] = (60, 40, 40)
        canvas[:label_h, cell_w:] = (40, 60, 40)
        cv2.putText(canvas, "GT", (8, lh - 2), font, sfs, (200, 200, 255), sft)
        cv2.putText(canvas, f"Experiment | {name}", (cell_w + 8, lh - 2),
                    font, sfs * 0.85, (200, 255, 200), sft)

        # Video frames (RGB -> BGR for cv2.VideoWriter)
        yv = label_h
        canvas[yv:yv + cell_h, :cell_w] = cv2.resize(
            cv2.cvtColor(gt_frames[i], cv2.COLOR_RGB2BGR), (cell_w, cell_h))
        canvas[yv:yv + cell_h, cell_w:] = cv2.resize(
            cv2.cvtColor(exp_frames[i], cv2.COLOR_RGB2BGR), (cell_w, cell_h))

        # Center divider
        cv2.line(canvas, (cell_w, 0), (cell_w, total_h), (100, 100, 100), 1)

        vw.write(canvas)

    vw.release()
    _ffmpeg_remux(tmp)
    if os.path.exists(tmp):
        os.replace(tmp, output_path)
    print(f"  [Compare] {name} -> {output_path}")


# =====================================================================
#  Summary
# =====================================================================

def print_and_save_summary(all_results, args, output_dir):
    has_metrics = any(r.get("metrics") for r in all_results)
    has_vbench = any(r.get("vbench") for r in all_results)

    print(f"\n{'=' * 80}")
    print(f"  Experiment Summary  |  blend={args.blended}  dilation={args.mask_dilation_iter}  gs={args.text_guidance_scale}")
    print(f"  {len(all_results)} videos  |  GT={'yes' if args.gt_root else 'no'}")
    print(f"{'=' * 80}")

    if has_metrics:
        keys = ["psnr_mean", "ssim_mean", "lpips_mean", "ewarp", "as_mean", "is_mean"]
        short = ["PSNR", "SSIM", "LPIPS", "Ewarp", "AS", "IS"]
        print(f"  {'Video':<16s}", end="")
        for s in short:
            print(f" {s:>8s}", end="")
        print()
        print(f"  {'-' * (20 + 9 * len(short))}")
        for r in all_results:
            m = r.get("metrics", {})
            print(f"  {r['name']:<16s}", end="")
            for k in keys:
                v = m.get(k)
                print(f" {fmt(v, 8, 3)}", end="")
            print()

    if has_vbench:
        dims = VBENCH_DIMS
        print(f"\n  VBench:")
        print(f"  {'Video':<16s}", end="")
        for d in dims:
            short = d.replace("_consistency", "_con").replace("_smoothness", "_smo") \
                     .replace("_flickering", "_flk").replace("_quality", "_q")[:10]
            print(f" {short:>10s}", end="")
        print()
        for r in all_results:
            vb = r.get("vbench", {})
            print(f"  {r['name']:<16s}", end="")
            for d in dims:
                v = vb.get(d)
                print(f" {fmt(v, 10, 4)}", end="")
            print()

    print(f"{'=' * 80}\n")

    # JSON
    serializable = {
        "config": {"blended": args.blended, "mask_dilation_iter": args.mask_dilation_iter,
                   "ckpt": args.ckpt,
                   "text_guidance_scale": args.text_guidance_scale, "has_gt": bool(args.gt_root)},
        "num_videos": len(all_results),
        "per_video": [],
    }
    for r in all_results:
        serializable["per_video"].append({
            "name": r["name"], "caption": r.get("caption", ""),
            "metrics": r.get("metrics", {}),
            "vbench": r.get("vbench", {}),
        })

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"  [JSON] {output_dir / 'summary.json'}")

    # CSV
    with open(output_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["video", "blended", "dilation", "guidance_scale"]
        metric_keys = ["psnr_mean", "ssim_mean", "lpips_mean", "ewarp", "as_mean", "is_mean"]
        header += metric_keys + VBENCH_DIMS
        w.writerow(header)
        for r in all_results:
            row = [r["name"], args.blended, args.mask_dilation_iter, args.text_guidance_scale]
            m = r.get("metrics", {})
            for k in metric_keys:
                row.append(m.get(k, ""))
            vb = r.get("vbench", {})
            for d in VBENCH_DIMS:
                row.append(vb.get(d, ""))
            w.writerow(row)
    print(f"  [CSV]  {output_dir / 'summary.csv'}")


# =====================================================================
#  Argparse
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="GT vs Experiment single-inference comparison (supports both BR/OR datasets)."
    )

    # Data
    p.add_argument("--dataset", type=str, default="davis")
    p.add_argument("--video_root", type=str, required=True)
    p.add_argument("--mask_root", type=str, required=True)
    p.add_argument("--gt_root", type=str, default=None,
                   help="GT root (optional). If provided, pixel metrics are computed.")
    p.add_argument("--caption_yaml", type=str, required=True)
    p.add_argument("--text_guidance_scale", type=float, default=3.5)
    p.add_argument("--ckpt", type=str, default="Normal CFG 4-Step",
                   help="PCM checkpoint key: '2-Step','4-Step','Normal CFG 4-Step', etc.")

    # Processing
    p.add_argument("--video_length", type=int, default=100)
    p.add_argument("--height", type=int, default=-1,
                   help="Resize input height (-1 = no resize). Use for Full-Res OR data.")
    p.add_argument("--width", type=int, default=-1,
                   help="Resize input width (-1 = no resize). Use for Full-Res OR data.")
    p.add_argument("--ref_stride", type=int, default=3)
    p.add_argument("--neighbor_length", type=int, default=25)
    p.add_argument("--subvideo_length", type=int, default=80)

    # Key experiment knobs
    p.add_argument("--mask_dilation_iter", type=int, default=0,
                   help="Mask dilation iterations (0=no dilation, 4-8 for OR-style)")
    p.add_argument("--blended", action="store_true",
                   help="Enable GaussianBlur feathered mask blending (OR-style)")

    # Model paths
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--vae_path", type=str, required=True)
    p.add_argument("--diffueraser_path", type=str, required=True)
    p.add_argument("--propainter_model_dir", type=str, required=True)
    p.add_argument("--pcm_weights_path", type=str, required=True)
    p.add_argument("--i3d_model_path", type=str, default="")
    p.add_argument("--raft_model_path", type=str, default="")

    # Evaluation (single flag: --eval)
    VBENCH_DEFAULT = [
        "subject_consistency", "background_consistency",
        "temporal_flickering", "motion_smoothness",
        "aesthetic_quality", "imaging_quality",
    ]
    p.add_argument("--eval", action="store_true",
                   help="Enable evaluation. With --gt_root: VBench + pixel metrics. Without: VBench only.")
    p.add_argument("--vbench_dimensions", nargs="+", default=VBENCH_DEFAULT)

    # Output
    p.add_argument("--output_dir", type=str, default="comparison_all")
    p.add_argument("--max_videos", type=int, default=10)
    p.add_argument("--fps", type=int, default=24)

    return p.parse_args()


# =====================================================================
#  Main
# =====================================================================

def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Caption YAML
    caption_dict = {}
    if os.path.exists(args.caption_yaml):
        with open(args.caption_yaml, "r", encoding="utf-8") as f:
            caption_dict = yaml.safe_load(f) or {}
        print(f"[Captions] Loaded {len(caption_dict)} entries from {args.caption_yaml}")
    else:
        print(f"[WARN] Caption YAML not found: {args.caption_yaml}")
        print(f"[WARN] Videos will have NO prompt text!")

    # List videos
    names = list_video_names(args.video_root, args.mask_root)
    if not names:
        raise RuntimeError("No videos found")
    if args.max_videos > 0:
        names = names[:args.max_videos]

    has_gt = args.gt_root and os.path.isdir(args.gt_root)

    # Determine eval mode
    eval_mode = "OFF"
    if args.eval:
        eval_mode = "VBench + Pixel Metrics (GT available)" if has_gt else "VBench only (no GT)"

    print(f"\n{'═' * 60}")
    print(f"  GT vs Experiment  (single inference)")
    print(f"  {'─' * 56}")
    print(f"  Videos       : {len(names)}")
    print(f"  GT           : {'✅ ' + args.gt_root if has_gt else '❌ (no --gt_root)'}")
    print(f"  Blended      : {'✅' if args.blended else '❌'}")
    print(f"  Mask Dilation : {args.mask_dilation_iter}")
    print(f"  CMF ckpt     : {args.ckpt}")
    print(f"  Text Scale   : {args.text_guidance_scale}")
    print(f"  Evaluation   : {eval_mode}")
    print(f"{'═' * 60}\n")

    # Load models
    print("[Loading ProPainter ...]")
    pp_model = Propainter(args.propainter_model_dir, device=device)

    print("[Loading DiffuEraser ...]")
    _prev = logging.root.manager.disable
    logging.disable(logging.WARNING)
    de_model = DiffuEraser(device, args.base_model_path, args.vae_path,
                           args.diffueraser_path, ckpt=args.ckpt,
                           pcm_weights_path=args.pcm_weights_path)
    logging.disable(_prev)

    # Metrics (auto-enabled when --eval + GT)
    metrics_calc = get_metrics_calc(args, device, has_gt)

    # VBench (auto-enabled when --eval)
    vbench_ctx = None
    if args.eval:
        try:
            vbench_ctx = init_vbench(args, device)
        except Exception as e:
            print(f"[VBench WARN] Init failed: {e}")

    # Process
    all_results = []
    for idx, name in enumerate(names, 1):
        print(f"\n[{idx}/{len(names)}] {name}")
        print("-" * 40)

        vdir = Path(args.video_root) / name
        mdir = Path(args.mask_root) / name
        gdir = Path(args.gt_root) / name if has_gt else None

        if not vdir.exists() or not mdir.exists():
            print(f"  SKIP: missing dir")
            continue

        # Resolve prompt
        prompt, n_prompt = "", ""
        if name in caption_dict:
            entry = caption_dict[name]
            p = entry.get("prompt", "")
            prompt = p[0] if isinstance(p, list) else str(p) if p else ""
            np_ = entry.get("n_prompt", "")
            n_prompt = np_[0] if isinstance(np_, list) else str(np_) if np_ else ""
            trunc = prompt[:60] + ("..." if len(prompt) > 60 else "")
            print(f"  Caption: {trunc}")

        vid_out = output_dir / name
        vid_out.mkdir(parents=True, exist_ok=True)

        try:
            result = run_one_video(
                name, str(vdir), str(mdir), str(gdir) if gdir else None,
                args, pp_model, de_model,
                prompt, n_prompt, args.text_guidance_scale,
                metrics_calc, vbench_ctx, device, vid_out
            )
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Clean up empty output dir if no results were saved
            if vid_out.exists() and not any(vid_out.iterdir()):
                vid_out.rmdir()
                print(f"  [Cleanup] Removed empty dir: {vid_out}")

        print(f"[Progress] {idx}/{len(names)} ({100 * idx // len(names)}%)")
        torch.cuda.empty_cache()

    # Summary
    if all_results:
        print_and_save_summary(all_results, args, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  DONE - {len(all_results)} videos")
    print(f"  Output: {output_dir}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
