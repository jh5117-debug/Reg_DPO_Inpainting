#!/usr/bin/env python
# coding=utf-8
"""
DPO Stage 1 Training — DiffuEraser DPO Finetune

训练目标: UNet2D + BrushNet (可训练)
冻结: VAE, text_encoder, ref_model (BrushNet_ref + UNet2D_ref)
损失函数: Diffusion-DPO loss

基于 train_DiffuEraser_stage1.py 改造，核心变化:
1. 使用 DPODataset 替代 FinetuneDataset
2. 加载冻结的 ref_model (BrushNet_ref + UNet2D_ref)
3. pos/neg 共享 noise + timestep，concat 后一次前向
4. DPO loss 替代 MSE loss
5. 最多保存 2 个权重 (best + last)
"""

import argparse
import contextlib
import gc
import warnings
import logging
import math
import os
import sys
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import imageio
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange, repeat
from packaging import version
from tqdm.auto import tqdm
import ast

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import transformers
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

# 需要从项目根目录 import
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffueraser.pipeline_diffueraser_stage1 import StableDiffusionDiffuEraserPipelineStageOne
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import UNetMotionModel
from DPO_finetune.dataset.dpo_dataset import DPODataset
from dataset.file_client import FileClient
from dataset.img_util import imfrombytes


if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


# ============================================================
# DPO Loss + Reg-DPO Diagnostics
# ============================================================
def compute_dpo_loss(model_pred, ref_pred, noise, beta_dpo=2500.0):
    """
    Diffusion-DPO loss + Reg-DPO 论文要求的诊断指标。

    model_pred: [2*B*F, 4, H, W] — policy 的 ε prediction (pos+neg concat)
    ref_pred:   [2*B*F, 4, H, W] — ref 的 ε prediction
    noise:      [B*F, 4, H, W]   — 共享 noise (target ε)

    Returns:
        loss:         DPO loss (标量)
        diagnostics:  dict 包含所有诊断指标
    """
    target = noise.repeat(2, 1, 1, 1)  # repeat 匹配 pos+neg

    # ‖ε - ε_θ‖² per-sample
    model_losses = (model_pred.float() - target.float()).pow(2).mean(dim=[1, 2, 3])
    model_losses_w, model_losses_l = model_losses.chunk(2)
    model_diff = model_losses_w - model_losses_l

    # ‖ε - ε_ref‖² (常量)
    with torch.no_grad():
        ref_losses = (ref_pred.float() - target.float()).pow(2).mean(dim=[1, 2, 3])
        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
        ref_diff = ref_losses_w - ref_losses_l

    scale_term = -0.5 * beta_dpo
    inside_term = scale_term * (model_diff - ref_diff)

    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
    loss = (-1.0 * F.logsigmoid(inside_term)).mean()

    # === Reg-DPO 诊断指标 ===
    # Win Gap = model_loss_w - ref_loss_w (理想为负: policy 比 ref 在正样本上更好)
    win_gap = (model_losses_w - ref_losses_w).mean()
    # Lose Gap = model_loss_l - ref_loss_l (理想为正: policy 不去改善负样本)
    lose_gap = (model_losses_l - ref_losses_l).mean()

    # Reward Margin: ref model 本身对 pos/neg 的区分度 (理想为负: ref 在正样本上更好)
    reward_margin = (ref_losses_w - ref_losses_l).mean()

    # Sigma Term: sigmoid(inside_term) 均值，检测饱和 (接近 1.0 = 梯度消失)
    sigma_term = torch.sigmoid(inside_term).mean()

    # KL Divergence: policy 偏离 ref 的程度 (太大 = catastrophic forgetting)
    all_model_losses = model_losses.mean()
    all_ref_losses = ref_losses.mean()
    kl_div = 0.5 * (all_model_losses - all_ref_losses)

    diagnostics = {
        "dpo_loss": loss.detach().item(),
        "implicit_acc": implicit_acc.detach().item(),
        "mse_w": model_losses_w.mean().detach().item(),
        "mse_l": model_losses_l.mean().detach().item(),
        "ref_mse_w": ref_losses_w.mean().detach().item(),
        "ref_mse_l": ref_losses_l.mean().detach().item(),
        "win_gap": win_gap.detach().item(),
        "lose_gap": lose_gap.detach().item(),
        "reward_margin": reward_margin.detach().item(),
        "sigma_term": sigma_term.detach().item(),
        "kl_divergence": kl_div.detach().item(),
    }

    return loss, diagnostics


def compute_dpo_grad_norm(loss, params_to_optimize):
    """
    计算 DPO loss 对可训练参数的梯度 L2 范数 (DGR 诊断)。
    需要在 backward 之后、optimizer.step 之前调用。
    """
    total_norm = 0.0
    for p in params_to_optimize:
        if p.grad is not None:
            total_norm += p.grad.data.float().norm(2).item() ** 2
    return total_norm ** 0.5


def format_dpo_diagnostics(step, diag, grad_norm=None, extra=None):
    """
    格式化 DPO 诊断指标为美观的 ASCII 表格，每 300 步输出。
    """
    sep = "═" * 72
    thin = "─" * 72
    lines = [
        "",
        f"  {sep}",
        f"  ║  DPO Diagnostics @ Step {step:>6d}".ljust(73) + "║",
        f"  {sep}",
        f"  ║  {'Metric':<20s}  {'Value':>12s}  {'Ideal':>12s}  {'Status':>10s}  ║",
        f"  ║  {thin[:68]}  ║",
    ]

    def status_icon(val, ideal_dir, threshold=0):
        if ideal_dir == "neg":
            return "✅" if val < threshold else "⚠️"
        elif ideal_dir == "pos":
            return "✅" if val > threshold else "⚠️"
        elif ideal_dir == "mid":
            return "✅" if 0.5 < val < 0.95 else "⚠️"
        return "—"

    rows = [
        ("L_dpo",         diag["dpo_loss"],     "↓ decreasing", status_icon(diag["dpo_loss"], "neg", 0.693)),
        ("Implicit Acc",  diag["implicit_acc"], "0.5~0.9",      status_icon(diag["implicit_acc"], "mid")),
        ("Win Gap",       diag["win_gap"],      "< 0 (neg)",    status_icon(diag["win_gap"], "neg")),
        ("Lose Gap",      diag["lose_gap"],     "> 0 (pos)",    status_icon(diag["lose_gap"], "pos")),
        ("Reward Margin",  diag["reward_margin"],"< 0 (neg)",    status_icon(diag["reward_margin"], "neg")),
        ("Sigma Term",    diag["sigma_term"],   "0.5~0.9",      status_icon(diag["sigma_term"], "mid")),
        ("KL Divergence", diag["kl_divergence"], "small",        "—"),
        ("MSE (win)",     diag["mse_w"],        "↓",            "—"),
        ("MSE (lose)",    diag["mse_l"],        "↑ or stable",  "—"),
        ("Ref MSE (win)", diag["ref_mse_w"],    "baseline",     "—"),
        ("Ref MSE (lose)",diag["ref_mse_l"],    "baseline",     "—"),
    ]

    if grad_norm is not None:
        rows.append(("DGR (grad norm)", grad_norm, "> 0 (alive)", status_icon(grad_norm, "pos", 1e-6)))
        if diag.get("grad_norm_ratio") is not None:
            rows.append(("Grad Norm Ratio", diag["grad_norm_ratio"], "> 0.01", status_icon(diag["grad_norm_ratio"], "pos", 0.01)))

    for name, val, ideal, st in rows:
        lines.append(f"  ║  {name:<20s}  {val:>12.6f}  {ideal:>12s}  {st:>10s}  ║")

    if extra:
        lines.append(f"  ║  {thin[:68]}  ║")
        for k, v in extra.items():
            lines.append(f"  ║  {k:<20s}  {v:>12.4f}  {'':>12s}  {'':>10s}  ║")

    lines.append(f"  {sep}")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Utility Functions
# ============================================================
def save_videos_grid(video, path: str, fps=8):
    outputs = [np.array(img) for img in video]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def print_model_info(models_dict, logger):
    logger.info("====== Model Parameters ======")
    for name, model in models_dict.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        status = "🔥 trainable" if trainable > 0 else "❄️ frozen"
        logger.info(f"  {name:15s}: total={total/1e6:>7.1f}M  trainable={trainable/1e6:>7.1f}M  frozen={frozen/1e6:>7.1f}M  [{status}]")
    logger.info("==============================")


def format_metrics_table(step, metrics_dict, n_videos):
    headers = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    col_w = max(10, max(len(h) for h in headers) + 2)
    n_cols = len(headers)
    table_w = (col_w + 1) * n_cols + 1

    lines = []
    title = f" Validation @ Step {step} ({n_videos} videos) "
    lines.append(f"\n{'=' * table_w}")
    lines.append(title.center(table_w))
    lines.append('=' * table_w)
    lines.append('|'.join(h.center(col_w) for h in [''] + headers))
    lines.append('-' * table_w)
    lines.append('|'.join(v.center(col_w) for v in ['Avg'] + [f'{v:.4f}' for v in values]))
    lines.append('=' * table_w)
    return '\n'.join(lines)


# ============================================================
# Validation (Stage 1: PSNR + SSIM only)
# ============================================================
def log_validation(
    vae, text_encoder, tokenizer, unet, brushnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation...")

    from inference.metrics import compute_psnr, compute_ssim

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*was not found in config.*")
        pipeline = StableDiffusionDiffuEraserPipelineStageOne.from_pretrained(
            args.base_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet),
            brushnet=accelerator.unwrap_model(brushnet),
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    val_data_dir = args.val_data_dir
    images_root = os.path.join(val_data_dir, "JPEGImages_432_240")
    masks_root = os.path.join(val_data_dir, "test_masks")

    if not os.path.isdir(images_root):
        logger.warning(f"Validation image dir not found: {images_root}, skipping.")
        del pipeline; gc.collect(); torch.cuda.empty_cache()
        return None, None

    video_dirs = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
    logger.info(f"Found {len(video_dirs)} validation videos in {images_root}")

    all_psnr, all_ssim = [], []
    file_client = FileClient('disk')

    for video_name in video_dirs:
        video_image_dir = os.path.join(images_root, video_name)
        video_mask_dir = os.path.join(masks_root, video_name)

        if not os.path.isdir(video_mask_dir):
            continue

        frame_list = sorted(os.listdir(video_image_dir))
        selected_index = list(range(len(frame_list)))[:args.nframes]

        frames, masks, masked_images = [], [], []
        for idx in selected_index:
            frame_path = os.path.join(video_image_dir, frame_list[idx])
            img_bytes = file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            frames.append(img)

            mask_path = os.path.join(video_mask_dir, str(idx).zfill(5) + '.png')
            if not os.path.exists(mask_path):
                break
            mask = Image.open(mask_path).convert('L')
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            masked_image = np.array(img) * (1 - (np.array(mask)[:, :, np.newaxis].astype(np.float32) / 255))
            masked_images.append(Image.fromarray(masked_image.astype(np.uint8)))

        if len(frames) != len(selected_index) or len(masks) != len(selected_index):
            continue

        try:
            with inference_ctx:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
                    images = pipeline(
                        num_frames=args.nframes, prompt="clean background",
                        images=masked_images, masks=masks,
                        num_inference_steps=50, generator=generator,
                        guidance_scale=0.0,
                    ).frames
        except Exception as e:
            logger.warning(f"Inference failed for {video_name}: {e}")
            continue

        pred_np = [np.array(img, dtype=np.uint8) for img in images]
        gt_np = [
            np.array(img.resize((pred_np[0].shape[1], pred_np[0].shape[0]), Image.BILINEAR), dtype=np.uint8)
            for img in frames[:len(pred_np)]
        ]

        v_psnr = [compute_psnr(gt, pred) for gt, pred in zip(gt_np, pred_np)]
        v_ssim = [compute_ssim(gt, pred) for gt, pred in zip(gt_np, pred_np)]
        all_psnr.append(float(np.mean(v_psnr)))
        all_ssim.append(float(np.mean(v_ssim)))

    avg_psnr, avg_ssim = None, None
    if all_psnr and all_ssim:
        avg_psnr = float(np.mean(all_psnr))
        avg_ssim = float(np.mean(all_ssim))
        table = format_metrics_table(step, {'PSNR': avg_psnr, 'SSIM': avg_ssim}, len(all_psnr))
        logger.info(table)

        if is_wandb_available() and accelerator.is_main_process:
            wandb.log({"val/psnr_mean": avg_psnr, "val/ssim_mean": avg_ssim}, step=step)
    else:
        logger.warning("No valid validation results collected.")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return avg_psnr, avg_ssim


# ============================================================
# Helper
# ============================================================
def import_model_class_from_model_name_or_path(base_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        base_model_name_or_path, subfolder="text_encoder", revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def ensure_list(input_variable):
    if isinstance(input_variable, list):
        return input_variable
    elif isinstance(input_variable, str):
        try:
            parsed_list = ast.literal_eval(input_variable)
            if isinstance(parsed_list, list):
                return parsed_list
            else:
                raise ValueError(f"The input string didn't evaluate to a list: {input_variable}")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input string: {input_variable}") from e
    else:
        raise TypeError(f"Input must be a list or string representing a list, got {type(input_variable)}")


# ============================================================
# Parse Args
# ============================================================
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="DiffuEraser DPO Stage 1 Training")
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="dpo-finetune-stage1")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs-dpo-stage1")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0)

    # Validation
    parser.add_argument("--validation_prompt", type=str, default=["clean background", "clean background"])
    parser.add_argument("--validation_image", type=str, default=["data_val/JPEGImages_432_240/bear", "data_val/JPEGImages_432_240/boat"])
    parser.add_argument("--validation_mask", type=str, default=["data_val/test_masks/bear", "data_val/test_masks/boat"])
    parser.add_argument("--val_data_dir", type=str, default="data_val")
    parser.add_argument("--validation_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=300,
                        help="每隔多少步输出详细 DPO 诊断日志")

    # W&B
    parser.add_argument("--tracker_project_name", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # ===== DPO 特有参数 =====
    parser.add_argument("--dpo_data_root", type=str, default="data/DPO_Finetune_data",
                        help="DPO 偏好对数据集根目录")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Ref model 权重路径 (SFT 后的 DiffuEraser 权重，含 unet_main/ 和 brushnet/)")
    parser.add_argument("--beta_dpo", type=float, default=2500.0,
                        help="DPO 温度系数 beta")
    parser.add_argument("--davis_oversample", type=int, default=10,
                        help="DAVIS 视频过采样倍数")
    parser.add_argument("--chunk_aligned", action="store_true",
                        help="是否根据 meta.json chunk 边界对齐采样")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.validation_image = ensure_list(args.validation_image)
    args.validation_mask = ensure_list(args.validation_mask)
    args.validation_prompt = ensure_list(args.validation_prompt)

    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8.")

    return args


# ============================================================
# Collate
# ============================================================
def collate_fn(examples):
    pixel_values_pos = torch.stack([e["pixel_values_pos"] for e in examples]).float()
    pixel_values_neg = torch.stack([e["pixel_values_neg"] for e in examples]).float()
    conditioning_pixel_values = torch.stack([e["conditioning_pixel_values"] for e in examples]).float()
    masks = torch.stack([e["masks"] for e in examples]).float()
    input_ids = torch.stack([e["input_ids"] for e in examples])

    return {
        "pixel_values_pos": pixel_values_pos,
        "pixel_values_neg": pixel_values_neg,
        "conditioning_pixel_values": conditioning_pixel_values,
        "masks": masks,
        "input_ids": input_ids,
    }


# ============================================================
# Main
# ============================================================
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # ===== WandB 初始化提前: 确保后续任何报错都能在 WandB 中可见 =====
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        for key in ["validation_prompt", "validation_image", "validation_mask"]:
            tracker_config.pop(key, None)

        init_kwargs = {}
        if args.report_to == "wandb":
            init_kwargs["wandb"] = {"name": f"dpo-stage1-{args.max_train_steps or 'auto'}steps"}
            if args.wandb_entity:
                init_kwargs["wandb"]["entity"] = args.wandb_entity

        try:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)
            logger.info("WandB tracker initialized successfully (early init).")
        except Exception as e:
            logger.warning(f"Failed to init WandB tracker: {e}. Continuing without tracking.")

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.base_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False,
        )

    text_encoder_cls = import_model_class_from_model_name_or_path(args.base_model_name_or_path, args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.base_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    # ===== 辅助函数: 从 UNetMotionModel 提取 2D 权重到 UNet2DConditionModel =====
    def _extract_2d_from_motion(motion_unet, base_model_path, revision=None, variant=None):
        """
        SFT 权重的 config.json 声明为 UNetMotionModel，不能直接用 UNet2DConditionModel 加载。
        必须先创建空的 UNet2D (从 SD base model 获取正确 config)，然后逐模块拷贝 2D 权重。
        """
        unet_2d = UNet2DConditionModel.from_pretrained(
            base_model_path, subfolder="unet", revision=revision, variant=variant
        )
        # 从 UNetMotionModel 拷贝 2D 权重
        unet_2d.conv_in.load_state_dict(motion_unet.conv_in.state_dict())
        unet_2d.time_proj.load_state_dict(motion_unet.time_proj.state_dict())
        unet_2d.time_embedding.load_state_dict(motion_unet.time_embedding.state_dict())

        for i, down_block in enumerate(motion_unet.down_blocks):
            unet_2d.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
            if hasattr(unet_2d.down_blocks[i], "attentions") and hasattr(down_block, "attentions"):
                unet_2d.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
            if unet_2d.down_blocks[i].downsamplers and down_block.downsamplers:
                unet_2d.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())

        for i, up_block in enumerate(motion_unet.up_blocks):
            unet_2d.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
            if hasattr(unet_2d.up_blocks[i], "attentions") and hasattr(up_block, "attentions"):
                unet_2d.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
            if unet_2d.up_blocks[i].upsamplers and up_block.upsamplers:
                unet_2d.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())

        unet_2d.mid_block.resnets.load_state_dict(motion_unet.mid_block.resnets.state_dict())
        unet_2d.mid_block.attentions.load_state_dict(motion_unet.mid_block.attentions.state_dict())

        if motion_unet.conv_norm_out is not None:
            unet_2d.conv_norm_out.load_state_dict(motion_unet.conv_norm_out.state_dict())
        if hasattr(motion_unet, 'conv_act') and motion_unet.conv_act is not None:
            unet_2d.conv_act.load_state_dict(motion_unet.conv_act.state_dict())
        unet_2d.conv_out.load_state_dict(motion_unet.conv_out.state_dict())
        return unet_2d

    # ===== 检测 ref_model_path 的 unet config 类型 =====
    ref_unet_config_path = os.path.join(args.ref_model_path, "unet_main", "config.json")
    is_motion_model = False
    if os.path.exists(ref_unet_config_path):
        import json
        with open(ref_unet_config_path) as f:
            unet_cfg = json.load(f)
        is_motion_model = (unet_cfg.get("_class_name", "") == "UNetMotionModel")
        logger.info(f"Ref unet config _class_name: {unet_cfg.get('_class_name')}, is_motion_model={is_motion_model}")

    # ===== Policy model: 从 ref_model_path 初始化 (可训练) =====
    if is_motion_model:
        logger.info(f"Loading policy UNet via UNetMotionModel from {args.ref_model_path} (extracting 2D weights)")
        _motion_unet = UNetMotionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")
        unet_main = _extract_2d_from_motion(_motion_unet, args.base_model_name_or_path, args.revision, args.variant)
        del _motion_unet
        logger.info("Successfully extracted policy 2D UNet weights from UNetMotionModel")
    else:
        logger.info(f"Loading policy UNet2D directly from {args.ref_model_path}")
        unet_main = UNet2DConditionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")

    logger.info(f"Loading policy BrushNet from {args.ref_model_path}")
    brushnet = BrushNetModel.from_pretrained(args.ref_model_path, subfolder="brushnet")

    # ===== Ref model: 从同一路径加载，冻结 =====
    if is_motion_model:
        logger.info(f"Loading ref UNet via UNetMotionModel from {args.ref_model_path} (extracting 2D weights)")
        _motion_unet_ref = UNetMotionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")
        unet_ref = _extract_2d_from_motion(_motion_unet_ref, args.base_model_name_or_path, args.revision, args.variant)
        del _motion_unet_ref
        logger.info("Successfully extracted ref 2D UNet weights from UNetMotionModel")
    else:
        logger.info(f"Loading ref UNet2D directly from {args.ref_model_path}")
        unet_ref = UNet2DConditionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")

    unet_ref.requires_grad_(False)
    unet_ref.eval()

    logger.info(f"Loading ref BrushNet from {args.ref_model_path}")
    brushnet_ref = BrushNetModel.from_pretrained(args.ref_model_path, subfolder="brushnet")
    brushnet_ref.requires_grad_(False)
    brushnet_ref.eval()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_main.train()
    brushnet.train()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet_main.enable_xformers_memory_efficient_attention()
            brushnet.enable_xformers_memory_efficient_attention()
            unet_ref.enable_xformers_memory_efficient_attention()
            brushnet_ref.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.gradient_checkpointing:
        unet_main.enable_gradient_checkpointing()
        brushnet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet_main.parameters())) + \
                         list(filter(lambda p: p.requires_grad, brushnet.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = DPODataset(args, tokenizer, dpo_data_root=args.dpo_data_root)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn,
        batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers,
    )
    train_dataset_len = len(train_dataset)
    train_dataloader_len = train_dataset_len // args.train_batch_size

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    unet_main, brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet_main, brushnet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet_ref.to(accelerator.device, dtype=weight_dtype)
    brushnet_ref.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # WandB init 已在前面提前完成

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running DPO Stage 1 Training *****")
    logger.info(f"  Num examples = {train_dataset_len}")
    logger.info(f"  Num batches each epoch (per GPU) = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Beta DPO = {args.beta_dpo}")
    print_model_info({
        'unet_main (policy)': unet_main, 'brushnet (policy)': brushnet,
        'unet_ref (frozen)': unet_ref, 'brushnet_ref (frozen)': brushnet_ref,
        'vae': vae, 'text_encoder': text_encoder,
    }, logger)

    warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*was not found in config.*")
    global_step = 0
    first_epoch = 0

    # 最佳权重追踪
    best_composite_score = -float('inf')
    initial_grad_norm = None  # DGR 归一化基准

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path), map_location="cpu")
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet_main, brushnet):
                torch.cuda.empty_cache()
                gc.collect()

                # === VAE Encode ===
                pos_latents = vae.encode(
                    rearrange(batch["pixel_values_pos"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                neg_latents = vae.encode(
                    rearrange(batch["pixel_values_neg"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor

                # BrushNet conditioning: GT masked image + mask
                n_batch = batch["conditioning_pixel_values"].shape[0]
                cond_latents = vae.encode(
                    rearrange(batch["conditioning_pixel_values"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                ).latent_dist.sample() * vae.config.scaling_factor
                cond_latents = rearrange(cond_latents, "(b f) c h w -> b f c h w", b=n_batch)

                masks = torch.nn.functional.interpolate(
                    batch["masks"].to(dtype=weight_dtype),
                    size=(1, pos_latents.shape[-2], pos_latents.shape[-1])
                )

                # VAE encode 完毕，释放原始像素 tensor 节省显存
                del batch["pixel_values_pos"], batch["pixel_values_neg"], batch["conditioning_pixel_values"]

                brushnet_cond = rearrange(
                    torch.concat([cond_latents, masks], 2),
                    "b f c h w -> (b f) c h w"
                )  # [(b f), 5, h, w]

                # === Shared noise + timestep ===
                noise = torch.randn_like(pos_latents)
                bsz = pos_latents.shape[0] // args.nframes
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=pos_latents.device
                ).long()
                # 展开 timesteps 到 per-frame 维度，匹配 latent shape (bsz*nframes, ...)
                timesteps_expanded = timesteps.repeat_interleave(args.nframes, dim=0)

                # 加噪: pos 和 neg 使用相同的 noise 但不同的 latent
                noisy_pos = noise_scheduler.add_noise(pos_latents, noise, timesteps_expanded)
                noisy_neg = noise_scheduler.add_noise(neg_latents, noise, timesteps_expanded)

                # Concat pos + neg: batch dim 翻倍
                noisy_all = torch.cat([noisy_pos, noisy_neg], dim=0)  # [2*b*f, 4, h, w]
                brushnet_cond_all = torch.cat([brushnet_cond, brushnet_cond], dim=0)
                timesteps_all = timesteps_expanded.repeat(2)  # [2*b*f]

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                encoder_hidden_states_expanded = rearrange(
                    repeat(encoder_hidden_states, "b c d -> b t c d", t=args.nframes),
                    'b t c d -> (b t) c d'
                )
                encoder_hidden_states_all = torch.cat([encoder_hidden_states_expanded, encoder_hidden_states_expanded], dim=0)

                # === Policy forward ===
                down_samples, mid_sample, up_samples = brushnet(
                    noisy_all, timesteps_all,
                    encoder_hidden_states=encoder_hidden_states_all,
                    brushnet_cond=brushnet_cond_all,
                    return_dict=False,
                )

                torch.cuda.empty_cache()
                gc.collect()

                model_pred = unet_main(
                    noisy_all, timesteps_all,
                    encoder_hidden_states=encoder_hidden_states_all,
                    down_block_add_samples=[s.to(dtype=weight_dtype) for s in down_samples],
                    mid_block_add_sample=mid_sample.to(dtype=weight_dtype),
                    up_block_add_samples=[s.to(dtype=weight_dtype) for s in up_samples],
                    return_dict=True,
                ).sample

                # Policy BrushNet 输出已被 UNet 消费，立即释放
                del down_samples, mid_sample, up_samples
                torch.cuda.empty_cache()
                gc.collect()

                # === Ref forward (no_grad) ===
                with torch.no_grad():
                    ref_down, ref_mid, ref_up = brushnet_ref(
                        noisy_all, timesteps_all,
                        encoder_hidden_states=encoder_hidden_states_all,
                        brushnet_cond=brushnet_cond_all,
                        return_dict=False,
                    )
                    ref_pred = unet_ref(
                        noisy_all, timesteps_all,
                        encoder_hidden_states=encoder_hidden_states_all,
                        down_block_add_samples=[s.to(dtype=weight_dtype) for s in ref_down],
                        mid_block_add_sample=ref_mid.to(dtype=weight_dtype),
                        up_block_add_samples=[s.to(dtype=weight_dtype) for s in ref_up],
                        return_dict=True,
                    ).sample

                # Ref BrushNet 输出已被消费，立即释放
                del ref_down, ref_mid, ref_up

                # === DPO Loss ===
                loss, diagnostics = compute_dpo_loss(
                    model_pred, ref_pred, noise, beta_dpo=args.beta_dpo
                )

                torch.cuda.empty_cache()
                gc.collect()

                accelerator.backward(loss)

                # DGR: 计算梯度范数（检测梯度消失）
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = compute_dpo_grad_norm(loss, params_to_optimize)
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    # === 每 300 步: 详细诊断日志 ===
                    if global_step % args.logging_steps == 0 or global_step == 1:
                        diag_table = format_dpo_diagnostics(
                            global_step, diagnostics, grad_norm=grad_norm
                        )
                        logger.info(diag_table)

                    # Checkpoint 保存
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for removing in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, removing))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Validation + 权重保存 (最多 best + last)
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        avg_psnr, avg_ssim = log_validation(
                            vae, text_encoder, tokenizer, unet_main, brushnet,
                            args, accelerator, weight_dtype, global_step,
                        )

                        if avg_psnr is not None and avg_ssim is not None:
                            composite = 0.5 * (avg_psnr / 50.0) + 0.5 * avg_ssim

                            if composite > best_composite_score:
                                best_composite_score = composite
                                try:
                                    best_dir = os.path.join(args.output_dir, "best_weights")
                                    os.makedirs(os.path.join(best_dir, "unet_main"), exist_ok=True)
                                    os.makedirs(os.path.join(best_dir, "brushnet"), exist_ok=True)
                                    unwrap_model(unet_main).save_pretrained(os.path.join(best_dir, "unet_main"))
                                    unwrap_model(brushnet).save_pretrained(os.path.join(best_dir, "brushnet"))
                                    logger.info(f"New best weights saved (composite={composite:.4f})")

                                    if is_wandb_available():
                                        artifact = wandb.Artifact(
                                            "dpo-stage1-best", type="model",
                                            metadata={"step": global_step, "psnr": avg_psnr, "ssim": avg_ssim, "composite": composite}
                                        )
                                        artifact.add_dir(best_dir)
                                        wandb.log_artifact(artifact)
                                except Exception as e:
                                    logger.warning(f"Failed to save best weights: {e}")

            # === WandB + progress bar logging (每步) ===
            logs = {
                "dpo_loss": diagnostics["dpo_loss"],
                "implicit_acc": diagnostics["implicit_acc"],
                "mse_w": diagnostics["mse_w"],
                "mse_l": diagnostics["mse_l"],
                "win_gap": diagnostics["win_gap"],
                "lose_gap": diagnostics["lose_gap"],
                "reward_margin": diagnostics["reward_margin"],
                "sigma_term": diagnostics["sigma_term"],
                "kl_divergence": diagnostics["kl_divergence"],
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if grad_norm is not None:
                logs["dgr_grad_norm"] = grad_norm
                if initial_grad_norm is None:
                    initial_grad_norm = grad_norm
                if initial_grad_norm > 0:
                    ratio = grad_norm / initial_grad_norm
                    logs["grad_norm_ratio"] = ratio
                    diagnostics["grad_norm_ratio"] = ratio
            progress_bar.set_postfix(**{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in list(logs.items())[:6]})
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 保存 last 权重
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        last_dir = os.path.join(args.output_dir, "last_weights")
        os.makedirs(os.path.join(last_dir, "unet_main"), exist_ok=True)
        os.makedirs(os.path.join(last_dir, "brushnet"), exist_ok=True)
        accelerator.unwrap_model(unet_main).save_pretrained(os.path.join(last_dir, "unet_main"))
        accelerator.unwrap_model(brushnet).save_pretrained(os.path.join(last_dir, "brushnet"))
        logger.info(f"Last weights saved to {last_dir}")

        if is_wandb_available():
            try:
                artifact = wandb.Artifact(
                    "dpo-stage1-last", type="model",
                    metadata={"step": global_step}
                )
                artifact.add_dir(last_dir)
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to upload last weights: {e}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Training crashed!\n{tb}")
        if is_wandb_available() and wandb.run is not None:
            wandb.alert(
                title="DPO Stage 1 Crashed",
                text=f"```\n{tb}\n```",
                level=wandb.AlertLevel.ERROR,
            )
            wandb.finish(exit_code=1)
        raise
