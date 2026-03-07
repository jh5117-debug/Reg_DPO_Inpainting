#!/usr/bin/env python
# coding=utf-8
"""
convert_checkpoint.py — 统一权重转换脚本

将 accelerator.save_state() 保存的 checkpoint 转换为
  unet_main/config.json + model.safetensors
  brushnet/config.json + model.safetensors

支持 Stage 1 (UNet2DConditionModel) 和 Stage 2 (UNetMotionModel)。

Usage:
  python convert_checkpoint.py \
    --stage 1 \
    --checkpoint_dir finetune-stage1/checkpoint-50000 \
    --base_model_path weights/stable-diffusion-v1-5 \
    --brushnet_path weights/diffuEraser \
    --output_dir converted_weights/finetuned-stage1

  python convert_checkpoint.py \
    --stage 2 \
    --checkpoint_dir finetune-stage2/checkpoint-50000 \
    --base_model_path weights/stable-diffusion-v1-5 \
    --brushnet_path weights/diffuEraser \
    --motion_adapter_path weights/animatediff-motion-adapter-v1-5-2 \
    --pretrained_stage1 converted_weights/finetuned-stage1 \
    --output_dir converted_weights/finetuned-stage2
"""

import argparse
import os
import torch
from accelerate import Accelerator
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel


def convert_stage1(args):
    """加载 Stage 1 checkpoint 并导出 UNet2D + BrushNet 权重。"""
    print(f"[Stage 1] Loading checkpoint from: {args.checkpoint_dir}")

    unet_main = UNet2DConditionModel.from_pretrained(
        args.base_model_path, subfolder="unet"
    )
    brushnet = BrushNetModel.from_pretrained(args.brushnet_path)

    accelerator = Accelerator()
    unet_main, brushnet = accelerator.prepare(unet_main, brushnet)
    accelerator.load_state(args.checkpoint_dir)

    unet_main = accelerator.unwrap_model(unet_main)
    brushnet = accelerator.unwrap_model(brushnet)

    unet_out = os.path.join(args.output_dir, "unet_main")
    brushnet_out = os.path.join(args.output_dir, "brushnet")
    os.makedirs(unet_out, exist_ok=True)
    os.makedirs(brushnet_out, exist_ok=True)

    unet_main.save_pretrained(unet_out)
    brushnet.save_pretrained(brushnet_out)
    print(f"[Stage 1] Saved: {unet_out}, {brushnet_out}")


def convert_stage2(args):
    """加载 Stage 2 checkpoint 并导出 UNetMotionModel + BrushNet 权重。"""
    from libs.unet_motion_model import MotionAdapter, UNetMotionModel

    print(f"[Stage 2] Loading checkpoint from: {args.checkpoint_dir}")

    if not args.pretrained_stage1:
        raise ValueError("Stage 2 转换需要 --pretrained_stage1 参数")
    if not args.motion_adapter_path:
        raise ValueError("Stage 2 转换需要 --motion_adapter_path 参数")

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_stage1, subfolder="unet_main"
    )
    brushnet = BrushNetModel.from_pretrained(
        args.pretrained_stage1, subfolder="brushnet"
    )
    motion_adapter = MotionAdapter.from_pretrained(args.motion_adapter_path)
    unet_main = UNetMotionModel.from_unet2d(unet, motion_adapter)

    accelerator = Accelerator()
    unet_main, brushnet = accelerator.prepare(unet_main, brushnet)
    accelerator.load_state(args.checkpoint_dir)

    unet_main = accelerator.unwrap_model(unet_main)
    brushnet = accelerator.unwrap_model(brushnet)

    unet_out = os.path.join(args.output_dir, "unet_main")
    brushnet_out = os.path.join(args.output_dir, "brushnet")
    os.makedirs(unet_out, exist_ok=True)
    os.makedirs(brushnet_out, exist_ok=True)

    unet_main.save_pretrained(unet_out)
    brushnet.save_pretrained(brushnet_out)
    print(f"[Stage 2] Saved: {unet_out}, {brushnet_out}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert accelerator checkpoint to inference-ready weights."
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2],
        help="Training stage (1 or 2).",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to the accelerator checkpoint directory (e.g., finetune-stage1/checkpoint-50000).",
    )
    parser.add_argument(
        "--base_model_path", type=str, required=True,
        help="Path to base SD1.5 model directory.",
    )
    parser.add_argument(
        "--brushnet_path", type=str, required=True,
        help="Path to pretrained BrushNet directory.",
    )
    parser.add_argument(
        "--motion_adapter_path", type=str, default=None,
        help="Path to MotionAdapter weights (required for Stage 2).",
    )
    parser.add_argument(
        "--pretrained_stage1", type=str, default=None,
        help="Path to converted Stage 1 weights (required for Stage 2).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for converted weights.",
    )
    args = parser.parse_args()

    if args.stage == 1:
        convert_stage1(args)
    else:
        convert_stage2(args)

    print("Done!")


if __name__ == "__main__":
    main()
