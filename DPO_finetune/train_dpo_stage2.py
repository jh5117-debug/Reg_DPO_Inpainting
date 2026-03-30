#!/usr/bin/env python
# coding=utf-8
"""
DPO Stage 2 Training — DiffuEraser DPO Finetune

训练目标: MotionModule (可训练)
冻结: VAE, text_encoder, UNet2D, BrushNet, ref_model 全部
损失函数: Diffusion-DPO loss

基于 train_DiffuEraser_stage2.py 改造，核心变化:
1. 使用 DPODataset 替代 FinetuneDataset
2. UNet2D + BrushNet 来自 DPO Stage 1，冻结
3. ref_model 来自 SFT 权重 (含 MotionModule)，冻结
4. 仅 MotionModule 参数可训练
5. DPO loss 替代 MSE loss
6. 验证 PSNR + SSIM + Ewarp + TC
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from diffueraser.pipeline_diffueraser import StableDiffusionDiffuEraserPipeline
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import UNetMotionModel, MotionAdapter
from DPO_finetune.dataset.dpo_dataset import DPODataset
from DPO_finetune.train_dpo_stage1 import (
    compute_dpo_loss,
    compute_dpo_grad_norm,
    gather_dpo_diagnostics,
    format_dpo_diagnostics,
    print_model_info,
    save_wandb_run_info,
    setup_process_console_capture,
    sync_console_logs_to_wandb,
)
from dataset.file_client import FileClient
from dataset.img_util import imfrombytes

if is_wandb_available():
    import wandb

check_min_version("0.27.0.dev0")
logger = get_logger(__name__)


# ============================================================
# Validation (Stage 2: PSNR + SSIM + Ewarp + TC)
# ============================================================
def log_validation(
    vae, text_encoder, tokenizer, unet_main, brushnet, args, accelerator, weight_dtype, step
):
    logger.info("Running Stage 2 validation...")

    from inference.metrics import compute_psnr, compute_ssim

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
        warnings.filterwarnings("ignore", message=".*was not found in config.*")
        pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            args.base_model_name_or_path,
            vae=accelerator.unwrap_model(vae),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            unet=accelerator.unwrap_model(unet_main),
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
    inference_ctx = torch.autocast("cuda")

    val_data_dir = args.val_data_dir
    images_root = os.path.join(val_data_dir, "JPEGImages_432_240")
    masks_root = os.path.join(val_data_dir, "test_masks")

    if not os.path.isdir(images_root):
        logger.warning(f"Validation image dir not found: {images_root}, skipping.")
        del pipeline; gc.collect(); torch.cuda.empty_cache()
        return {}

    video_dirs = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
    logger.info(f"Found {len(video_dirs)} validation videos")

    all_psnr, all_ssim, all_ewarp, all_tc = [], [], [], []
    file_client = FileClient('disk')

    # Ewarp metric
    ewarp_metric = None
    try:
        from inference.metrics import MetricsCalculator
        raft_path = getattr(args, "raft_model_path", os.path.join(PROJECT_ROOT, "weights", "propainter", "raft-things.pth"))
        i3d_path = getattr(args, "i3d_model_path", os.path.join(PROJECT_ROOT, "weights", "i3d_rgb_imagenet.pt"))
        if os.path.exists(raft_path):
            ewarp_metric = MetricsCalculator(
                device=accelerator.device,
                raft_model_path=raft_path,
                i3d_model_path=i3d_path,
            )
    except Exception as e:
        logger.warning(f"Failed to init ewarp metric: {e}")

    # TC metric
    tc_metric = None
    try:
        from inference.metrics import TemporalConsistencyMetric
        clip_path = getattr(args, "clip_model_path", os.path.join(PROJECT_ROOT, "weights", "open_clip", "ViT-H-14"))
        tc_metric = TemporalConsistencyMetric(device=accelerator.device, model_path=clip_path)
    except Exception as e:
        logger.warning(f"Failed to init TC metric: {e}")

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

        if ewarp_metric is not None:
            try:
                ewarp = ewarp_metric.calc_ewarp(pred_np, gt_np)
                all_ewarp.append(float(ewarp))
            except Exception as e:
                logger.warning(f"Ewarp failed for {video_name}: {e}")

        if tc_metric is not None:
            try:
                tc = tc_metric.compute(pred_np)
                all_tc.append(float(tc))
            except Exception as e:
                logger.warning(f"TC failed for {video_name}: {e}")

    results = {}
    if all_psnr:
        results["psnr"] = float(np.mean(all_psnr))
        results["ssim"] = float(np.mean(all_ssim))
    if all_ewarp:
        results["ewarp"] = float(np.mean(all_ewarp))
    if all_tc:
        results["tc"] = float(np.mean(all_tc))

    if results and accelerator.is_main_process:
        log_str = " | ".join([f"{k}: {v:.4f}" for k, v in results.items()])
        logger.info(f"[Validation @ Step {step}] {log_str} ({len(all_psnr)} videos)")

        if is_wandb_available():
            wandb.log({f"val/{k}": v for k, v in results.items()}, step=step)

    del pipeline
    if ewarp_metric:
        del ewarp_metric
    if tc_metric:
        del tc_metric
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ============================================================
# Best Score Computation
# ============================================================
def compute_composite_score(results, history):
    """综合 PSNR/SSIM/Ewarp/TC，使用简单 min-max 归一化"""
    psnr = results.get("psnr")
    ssim = results.get("ssim")
    ewarp = results.get("ewarp")
    tc = results.get("tc")

    if psnr is None or ssim is None:
        return None

    # 简化: 使用固定参考范围
    psnr_norm = min(psnr / 50.0, 1.0)
    ssim_norm = ssim

    score = 0.3 * psnr_norm + 0.2 * ssim_norm

    if ewarp is not None:
        ewarp_norm = min(ewarp / 10.0, 1.0)
        score += 0.3 * (1.0 - ewarp_norm)
    else:
        score += 0.3 * 0.5  # 缺省值

    if tc is not None:
        score += 0.2 * tc
    else:
        score += 0.2 * 0.5

    return score


# ============================================================
# Helpers
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
                raise ValueError(f"Didn't eval to list: {input_variable}")
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid input: {input_variable}") from e
    else:
        raise TypeError(f"Expected list or str, got {type(input_variable)}")


# ============================================================
# Parse Args
# ============================================================
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="DiffuEraser DPO Stage 2 Training")
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--pretrained_dpo_stage1", type=str, default=None,
                        help="DPO Stage 1 best/last 权重路径 (含 unet_main/ brushnet/)")
    parser.add_argument("--baseline_unet_path", type=str, default=None,
                        help="包含 MotionModule 的 DiffuEraser baseline 权重路径")
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--motion_adapter_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="dpo-finetune-stage2")
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
    parser.add_argument("--logging_dir", type=str, default="logs-dpo-stage2")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--proportion_empty_prompts", type=float, default=0)

    # Validation
    parser.add_argument("--validation_prompt", type=str, default=["clean background"])
    parser.add_argument("--validation_image", type=str, default=["data_val/JPEGImages_432_240/bear"])
    parser.add_argument("--validation_mask", type=str, default=["data_val/test_masks/bear"])
    parser.add_argument("--val_data_dir", type=str, default="data_val")
    parser.add_argument("--validation_steps", type=int, default=300)
    parser.add_argument("--logging_steps", type=int, default=300,
                        help="每隔多少步输出详细 DPO 诊断日志")

    # W&B
    parser.add_argument("--tracker_project_name", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # ===== DPO 特有参数 =====
    parser.add_argument("--dpo_data_root", type=str, default="data/DPO_Finetune_data")
    parser.add_argument("--ref_model_path", type=str, default=None,
                        help="Ref model 权重路径 (SFT 后的完整 DiffuEraser 权重)")
    parser.add_argument("--beta_dpo", type=float, default=500.0,
                        help="DPO 温度系数 beta (推荐 500~1000，过大导致 sigmoid 饱和)")
    parser.add_argument("--davis_oversample", type=int, default=10)
    parser.add_argument("--chunk_aligned", action="store_true")

    # Metric model paths
    parser.add_argument("--raft_model_path", type=str, default=None)
    parser.add_argument("--i3d_model_path", type=str, default=None)
    parser.add_argument("--clip_model_path", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.validation_image = ensure_list(args.validation_image)
    args.validation_mask = ensure_list(args.validation_mask)
    args.validation_prompt = ensure_list(args.validation_prompt)

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
    setup_process_console_capture(args.output_dir)
    accelerator.wait_for_everyone()

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
        os.makedirs(args.output_dir, exist_ok=True)

    # ===== WandB 初始化提前: 确保后续任何报错都能在 WandB 中可见 =====
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        for key in ["validation_prompt", "validation_image", "validation_mask"]:
            tracker_config.pop(key, None)

        init_kwargs = {}
        if args.report_to == "wandb":
            init_kwargs["wandb"] = {"name": f"dpo-stage2-{args.max_train_steps or 'auto'}steps"}
            if args.wandb_entity:
                init_kwargs["wandb"]["entity"] = args.wandb_entity

        try:
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs=init_kwargs)
            logger.info("WandB tracker initialized successfully (early init).")
            save_wandb_run_info(args.output_dir, args)
            sync_console_logs_to_wandb(args.output_dir, policy="live")
        except Exception as e:
            logger.error(f"Failed to init WandB tracker: {e}")
            raise RuntimeError(
                "WandB tracker initialization failed before training started. "
                "Aborting to avoid running without a visible W&B run."
            ) from e

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    else:
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

    # ===== Policy UNetMotionModel =====
    # 从 DiffuEraser baseline 加载 MotionModule (含完整 3D UNet)
    # 然后用 DPO Stage 1 的 2D 权重覆盖
    logger.info(f"Loading baseline UNetMotionModel from {args.baseline_unet_path}")
    unet_main = UNetMotionModel.from_pretrained(args.baseline_unet_path, subfolder="unet_main")

    # 覆盖 2D 权重 (来自 DPO Stage 1)
    logger.info(f"Overriding 2D weights from DPO Stage 1: {args.pretrained_dpo_stage1}")
    stage1_unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_dpo_stage1, subfolder="unet_main", variant=args.variant,
    )

    # 逐模块拷贝 2D 权重
    unet_main.conv_in.load_state_dict(stage1_unet.conv_in.state_dict())
    unet_main.time_proj.load_state_dict(stage1_unet.time_proj.state_dict())
    unet_main.time_embedding.load_state_dict(stage1_unet.time_embedding.state_dict())

    for i, down_block in enumerate(stage1_unet.down_blocks):
        unet_main.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
        if hasattr(unet_main.down_blocks[i], "attentions") and hasattr(down_block, "attentions"):
            unet_main.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
        if unet_main.down_blocks[i].downsamplers and down_block.downsamplers:
            unet_main.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())

    for i, up_block in enumerate(stage1_unet.up_blocks):
        unet_main.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
        if hasattr(unet_main.up_blocks[i], "attentions") and hasattr(up_block, "attentions"):
            unet_main.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
        if unet_main.up_blocks[i].upsamplers and up_block.upsamplers:
            unet_main.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())

    unet_main.mid_block.resnets.load_state_dict(stage1_unet.mid_block.resnets.state_dict())
    unet_main.mid_block.attentions.load_state_dict(stage1_unet.mid_block.attentions.state_dict())

    if stage1_unet.conv_norm_out is not None:
        unet_main.conv_norm_out.load_state_dict(stage1_unet.conv_norm_out.state_dict())
    if hasattr(stage1_unet, 'conv_act') and stage1_unet.conv_act is not None:
        unet_main.conv_act.load_state_dict(stage1_unet.conv_act.state_dict())
    unet_main.conv_out.load_state_dict(stage1_unet.conv_out.state_dict())
    del stage1_unet
    logger.info("Successfully loaded baseline MotionModule + DPO Stage 1 2D weights")

    # 显式设置训练态：from_pretrained() 默认 eval mode
    # 不设 train() 会导致 gradient_checkpointing 失效 + temporal dropout 关闭
    unet_main.train()

    # Policy BrushNet (来自 DPO Stage 1，冻结)
    brushnet = BrushNetModel.from_pretrained(args.pretrained_dpo_stage1, subfolder="brushnet")

    # 冻结 2D params + BrushNet，只训练 MotionModule
    vae.requires_grad_(False)
    unet_main.freeze_unet2d_params()  # 只冻结 2D，保留 temporal layers 可训练
    text_encoder.requires_grad_(False)
    brushnet.requires_grad_(False)

    # ===== Ref model (完整 UNetMotionModel + BrushNet，冻结) =====
    logger.info(f"Loading ref UNetMotionModel from {args.ref_model_path}")
    unet_ref = UNetMotionModel.from_pretrained(args.ref_model_path, subfolder="unet_main")
    unet_ref.requires_grad_(False)
    unet_ref.eval()

    logger.info(f"Loading ref BrushNet from {args.ref_model_path}")
    brushnet_ref = BrushNetModel.from_pretrained(args.ref_model_path, subfolder="brushnet")
    brushnet_ref.requires_grad_(False)
    brushnet_ref.eval()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet_main.enable_xformers_memory_efficient_attention()
            brushnet.enable_xformers_memory_efficient_attention()
            unet_ref.enable_xformers_memory_efficient_attention()
            brushnet_ref.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available.")

    if args.gradient_checkpointing:
        unet_main.enable_gradient_checkpointing()

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

    # 只优化 MotionModule 参数
    motion_params = [p for p in unet_main.parameters() if p.requires_grad]
    logger.info(f"Trainable params (MotionModule): {sum(p.numel() for p in motion_params)/1e6:.1f}M")

    optimizer = optimizer_class(
        motion_params,
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
    logger.info("***** Running DPO Stage 2 Training *****")
    logger.info(f"  Num examples = {train_dataset_len}")
    logger.info(f"  Num batches each epoch (per GPU) = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Beta DPO = {args.beta_dpo}")
    print_model_info({
        'unet_main (policy-MM)': unet_main, 'brushnet (frozen)': brushnet,
        'unet_ref (frozen)': unet_ref, 'brushnet_ref (frozen)': brushnet_ref,
        'vae': vae, 'text_encoder': text_encoder,
    }, logger)

    warnings.filterwarnings("ignore", message=".*scale.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*was not found in config.*")
    global_step = 0
    first_epoch = 0
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
            accelerator.print("No checkpoint found, starting fresh.")
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
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
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
                )

                # === Shared noise + timestep ===
                noise = torch.randn_like(pos_latents)
                bsz = pos_latents.shape[0] // args.nframes
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=pos_latents.device
                ).long()
                timesteps_expanded = timesteps.repeat_interleave(args.nframes, dim=0)

                noisy_pos = noise_scheduler.add_noise(
                    pos_latents, noise, timesteps_expanded
                )
                noisy_neg = noise_scheduler.add_noise(
                    neg_latents, noise, timesteps_expanded
                )

                noisy_all = torch.cat([noisy_pos, noisy_neg], dim=0)
                brushnet_cond_all = torch.cat([brushnet_cond, brushnet_cond], dim=0)
                # BrushNet (2D) 需要 per-frame timesteps: (2*bsz*nframes,)
                timesteps_all_2d = timesteps_expanded.repeat(2)
                # UNetMotionModel 内部自己 repeat_interleave(num_frames)，只需要 per-batch: (2*bsz,)
                timesteps_all_motion = timesteps.repeat(2)

                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]
                encoder_hidden_states_expanded = rearrange(
                    repeat(encoder_hidden_states, "b c d -> b t c d", t=args.nframes),
                    'b t c d -> (b t) c d'
                )
                encoder_hidden_states_all = torch.cat(
                    [encoder_hidden_states_expanded, encoder_hidden_states_expanded], dim=0
                )

                # === Policy forward ===
                # BrushNet forward (2D encoder, 冻结)
                down_samples, mid_sample, up_samples = brushnet(
                    noisy_all, timesteps_all_2d,
                    encoder_hidden_states=encoder_hidden_states_all,
                    brushnet_cond=brushnet_cond_all,
                    return_dict=False,
                )

                torch.cuda.empty_cache()
                gc.collect()

                # UNetMotionModel forward (MotionModule 可训练)
                # DPO concat 后 noisy_all batch 翻倍，encoder_hidden_states 也需要翻倍
                encoder_hidden_states_motion = encoder_hidden_states.repeat(2, 1, 1)
                model_pred = unet_main(
                    noisy_all, timesteps_all_motion,
                    encoder_hidden_states=encoder_hidden_states_motion,
                    down_block_add_samples=[s.to(dtype=weight_dtype) for s in down_samples],
                    mid_block_add_sample=mid_sample.to(dtype=weight_dtype),
                    up_block_add_samples=[s.to(dtype=weight_dtype) for s in up_samples],
                    return_dict=True,
                    num_frames=args.nframes,
                ).sample

                # Policy BrushNet 输出已被 UNet 消费，立即释放
                del down_samples, mid_sample, up_samples
                torch.cuda.empty_cache()
                gc.collect()

                # === Ref forward (no_grad) ===
                with torch.no_grad():
                    ref_down, ref_mid, ref_up = brushnet_ref(
                        noisy_all, timesteps_all_2d,
                        encoder_hidden_states=encoder_hidden_states_all,
                        brushnet_cond=brushnet_cond_all,
                        return_dict=False,
                    )
                    ref_pred = unet_ref(
                        noisy_all, timesteps_all_motion,
                        encoder_hidden_states=encoder_hidden_states_motion,
                        down_block_add_samples=[s.to(dtype=weight_dtype) for s in ref_down],
                        mid_block_add_sample=ref_mid.to(dtype=weight_dtype),
                        up_block_add_samples=[s.to(dtype=weight_dtype) for s in ref_up],
                        return_dict=True,
                        num_frames=args.nframes,
                    ).sample

                # Ref BrushNet 输出已被消费，立即释放
                del ref_down, ref_mid, ref_up

                # === DPO Loss ===
                loss, diagnostics = compute_dpo_loss(
                    model_pred, ref_pred, noise, beta_dpo=args.beta_dpo
                )
                # 跨卡 gather: 全局 implicit_acc / inside_term 统计
                diagnostics = gather_dpo_diagnostics(diagnostics, accelerator)

                torch.cuda.empty_cache()
                gc.collect()

                accelerator.backward(loss)

                # DGR: 计算梯度范数（检测梯度消失）
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = compute_dpo_grad_norm(loss, motion_params)
                    accelerator.clip_grad_norm_(motion_params, args.max_grad_norm)

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

                    # Validation + 权重保存
                    if args.validation_prompt is not None and not os.environ.get("SKIP_VALIDATION") and \
                       global_step % args.validation_steps == 0:
                        results = log_validation(
                            vae, text_encoder, tokenizer, unet_main, brushnet,
                            args, accelerator, weight_dtype, global_step,
                        )

                        if results:
                            composite = compute_composite_score(results, None)
                            if composite is not None and composite > best_composite_score:
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
                                            "dpo-stage2-best", type="model",
                                            metadata={"step": global_step, "composite": composite, **results}
                                        )
                                        artifact.add_dir(best_dir)
                                        wandb.log_artifact(artifact)
                                except Exception as e:
                                    logger.warning(f"Failed to save best weights: {e}")

            # === WandB + progress bar logging (每步) ===
            scope_prefix = "global/" if diagnostics.get("_scope") == "global" else "rank0/"
            logs = {
                "rank0/dpo_loss": diagnostics["dpo_loss"],
                f"{scope_prefix}implicit_acc": diagnostics["implicit_acc"],
                "rank0/mse_w": diagnostics["mse_w"],
                "rank0/mse_l": diagnostics["mse_l"],
                "rank0/win_gap": diagnostics["win_gap"],
                "rank0/lose_gap": diagnostics["lose_gap"],
                "rank0/reward_margin": diagnostics["reward_margin"],
                "rank0/sigma_term": diagnostics["sigma_term"],
                "rank0/kl_divergence": diagnostics["kl_divergence"],
                f"{scope_prefix}inside_term_mean": diagnostics.get("inside_term_mean", 0),
                f"{scope_prefix}inside_term_min": diagnostics.get("inside_term_min", 0),
                f"{scope_prefix}inside_term_max": diagnostics.get("inside_term_max", 0),
                f"{scope_prefix}loser_dominant_ratio": diagnostics.get("loser_degrade_ratio", 0),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if grad_norm is not None:
                logs["rank0/dgr_grad_norm"] = grad_norm
                if initial_grad_norm is None:
                    initial_grad_norm = grad_norm
                if initial_grad_norm > 0:
                    ratio = grad_norm / initial_grad_norm
                    logs["rank0/grad_norm_ratio"] = ratio
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
                artifact = wandb.Artifact("dpo-stage2-last", type="model", metadata={"step": global_step})
                artifact.add_dir(last_dir)
                wandb.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"Failed to upload last weights: {e}")
        sync_console_logs_to_wandb(args.output_dir, policy="now")

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
            try:
                sync_console_logs_to_wandb(args.output_dir, policy="now")
                wandb.alert(
                    title="DPO Stage 2 Crashed",
                    text=f"```\n{tb}\n```",
                    level=wandb.AlertLevel.ERROR,
                )
                wandb.finish(exit_code=1)
            except Exception:
                pass
        raise
