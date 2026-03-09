#!/usr/bin/env python
# coding=utf-8
"""
run_train_stage1.py — Stage 1 训练入口

自动检测项目根目录（scripts/ 的父目录），无需硬编码路径。
训练完成后权重自动转换 (内嵌在 train_DiffuEraser_stage1.py 末尾)。

Usage:
    python scripts/run_train_stage1.py                # 单卡
    python scripts/run_train_stage1.py --num_gpus 8   # 8 卡
"""

import argparse
import os
import subprocess
import sys


def get_project_root():
    """自动检测项目根目录 = 本脚本所在目录 (scripts/) 的上一级。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_paths(project_root):
    """从项目根目录推导所有路径。"""
    return {
        "work_dir": project_root,
        "weights": os.path.join(project_root, "weights"),
        "davis": os.path.join(project_root, "dataset", "DAVIS"),
        "ytbv": os.path.join(project_root, "dataset", "YTBV"),
        "eval_davis": os.path.join(project_root, "data", "eval", "DAVIS"),
    }


def build_stage1_cmd(paths, args):
    """组装 Stage 1 训练命令。"""
    eval_davis = paths["eval_davis"]
    val_images = [
        os.path.join(eval_davis, "JPEGImages", "480p", "bear"),
        os.path.join(eval_davis, "JPEGImages", "480p", "boat"),
    ]
    val_masks = [
        os.path.join(eval_davis, "Annotations", "480p", "bear"),
        os.path.join(eval_davis, "Annotations", "480p", "boat"),
    ]
    val_prompts = ["clean background", "clean background"]

    cmd = [
        "accelerate", "launch",
        "--num_processes", str(args.num_gpus),
        "--mixed_precision", args.mixed_precision,
        "train_DiffuEraser_stage1.py",
        "--base_model_name_or_path", os.path.join(paths["weights"], "stable-diffusion-v1-5"),
        "--brushnet_model_name_or_path", os.path.join(paths["weights"], "diffuEraser"),
        "--vae_path", os.path.join(paths["weights"], "sd-vae-ft-mse"),
        "--output_dir", os.path.join(paths["work_dir"], "finetune-stage1"),
        "--logging_dir", "logs-finetune-stage1",
        "--davis_root", paths["davis"],
        "--ytvos_root", paths["ytbv"],
        "--resolution", "512",
        "--nframes", str(args.nframes),
        "--train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", args.lr_scheduler,
        "--lr_warmup_steps", str(args.lr_warmup_steps),
        "--max_train_steps", str(args.max_train_steps),
        "--checkpointing_steps", str(args.checkpointing_steps),
        "--validation_steps", str(args.validation_steps),
        "--seed", str(args.seed),
        "--report_to", "wandb",
        "--tracker_project_name", args.wandb_project,
        "--enable_xformers_memory_efficient_attention",
        "--gradient_checkpointing",
        "--mixed_precision", args.mixed_precision,
        "--set_grads_to_none",
        "--resume_from_checkpoint", "latest",
        "--validation_image", str(val_images),
        "--validation_mask", str(val_masks),
        "--validation_prompt", str(val_prompts),
    ]

    if args.checkpoints_total_limit:
        cmd.extend(["--checkpoints_total_limit", str(args.checkpoints_total_limit)])

    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])

    return cmd


def run_stage1(args=None):
    """运行 Stage 1 训练。"""
    if args is None:
        args = parse_args()

    project_root = get_project_root()
    paths = get_project_paths(project_root)
    cmd = build_stage1_cmd(paths, args)

    print("=" * 60)
    print("  DiffuEraser Stage 1 Training")
    print("=" * 60)
    print(f"  Project Root: {project_root}")
    print(f"  GPUs:         {args.num_gpus}")
    print(f"  Max Steps:    {args.max_train_steps}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  LR:           {args.learning_rate}")
    print("=" * 60)
    print(f"\n  Command:\n  {' '.join(cmd[:6])} \\\n    " + " \\\n    ".join(cmd[6:]))
    print()

    result = subprocess.run(cmd, cwd=paths["work_dir"])
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Training Entry")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--validation_steps", type=int, default=2000)
    parser.add_argument("--nframes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--wandb_project", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_stage1())
