#!/usr/bin/env python
# coding=utf-8
"""
run_train_stage1.py — Stage 1 训练入口

自动检测项目根目录（scripts/ 的父目录），无需硬编码路径。
数据和权重路径可通过命令行参数覆盖。

Usage:
    python scripts/run_train_stage1.py                          # 默认路径
    python scripts/run_train_stage1.py --data_dir /path/to/data # 自定义数据路径
"""

import argparse
import os
import subprocess
import sys


def get_project_root():
    """自动检测项目根目录 = 本脚本所在目录 (scripts/) 的上一级。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_stage1_cmd(project_root, args):
    """组装 Stage 1 训练命令。"""
    data_dir = args.data_dir or os.path.join(project_root, "data")
    weights_dir = args.weights_dir or os.path.join(project_root, "weights")

    davis_root = os.path.join(data_dir, "DAVIS")
    ytbv_root = os.path.join(data_dir, "YTBV")

    # Validation uses a separate eval dataset (auto-discovered from val_data_dir)
    eval_dir = os.path.join(project_root, "data_val")

    cmd = [
        "accelerate", "launch",
        "--num_processes", str(args.num_gpus),
        "--mixed_precision", args.mixed_precision,
        "train_DiffuEraser_stage1.py",
        "--base_model_name_or_path", os.path.join(weights_dir, "stable-diffusion-v1-5"),
        "--brushnet_model_name_or_path", os.path.join(weights_dir, "diffuEraser", "brushnet"),
        "--baseline_unet_path", args.baseline_unet_path or os.path.join(weights_dir, "diffuEraser"),
        "--vae_path", os.path.join(weights_dir, "sd-vae-ft-mse"),
        "--output_dir", os.path.join(project_root, "finetune-stage1"),
        "--logging_dir", "logs-finetune-stage1",
        "--davis_root", davis_root,
        "--ytvos_root", ytbv_root,
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
        "--val_data_dir", eval_dir,
        "--seed", str(args.seed),
        "--report_to", "wandb",
        "--tracker_project_name", args.wandb_project,
        "--enable_xformers_memory_efficient_attention",
        "--gradient_checkpointing",
        "--mixed_precision", args.mixed_precision,
        "--set_grads_to_none",
        "--resume_from_checkpoint", "latest",
    ]

    if args.checkpoints_total_limit:
        cmd.extend(["--checkpoints_total_limit", str(args.checkpoints_total_limit)])

    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])

    return cmd, data_dir, weights_dir


def run_stage1(args=None):
    """运行 Stage 1 训练。"""
    if args is None:
        args = parse_args()

    project_root = get_project_root()
    cmd, data_dir, weights_dir = build_stage1_cmd(project_root, args)

    print("=" * 60)
    print("  DiffuEraser Stage 1 Training")
    print("=" * 60)
    print(f"  Project Root: {project_root}")
    print(f"  Data Dir:     {data_dir}")
    print(f"  Weights Dir:  {weights_dir}")
    print(f"  GPUs:         {args.num_gpus}")
    print(f"  Max Steps:    {args.max_train_steps}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  LR:           {args.learning_rate}")
    print("=" * 60)
    print(f"\n  Command:\n  {' '.join(cmd[:6])} \\\n    " + " \\\n    ".join(cmd[6:]))
    print()

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Training Entry")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录 (含 DAVIS/ YTBV/)。默认: <project_root>/data/")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="权重目录。默认: <project_root>/weights/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=26000)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--validation_steps", type=int, default=2000)
    parser.add_argument("--nframes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--wandb_project", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--baseline_unet_path", type=str, default=None,
                        help="DiffuEraser baseline weights dir. Default: <weights_dir>/diffuEraser")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run_stage1())
