#!/usr/bin/env python
# coding=utf-8
"""
run_train_stage1.py — Stage 1 训练入口

从 $PROJECT_HOME 环境变量推导所有路径，调用 accelerate launch 启动训练。
训练完成后权重自动转换 (内嵌在 train_DiffuEraser_stage1.py 末尾)。

Usage:
    python run_train_stage1.py                # 单卡
    python run_train_stage1.py --num_gpus 8   # 8 卡
"""

import argparse
import os
import subprocess
import sys


def get_project_paths():
    """从 PROJECT_HOME 环境变量推导项目路径。"""
    project_home = os.environ.get("PROJECT_HOME")
    if not project_home:
        raise EnvironmentError(
            "请设置 PROJECT_HOME 环境变量，例如:\n"
            '  export PROJECT_HOME="/sc-projects/sc-proj-cc09-repair/hongyou"'
        )

    work_dir = os.path.join(project_home, "dev", "DiffuEraser_finetune")
    return {
        "work_dir": work_dir,
        "weights": os.path.join(work_dir, "weights"),
        "davis": os.path.join(work_dir, "dataset", "DAVIS"),
        "ytbv": os.path.join(work_dir, "dataset", "YTBV"),
        "eval_davis": os.path.join(work_dir, "data", "eval", "DAVIS"),
    }


def build_stage1_cmd(paths, args):
    """组装 Stage 1 训练命令。"""
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
        "--validation_image", str(["bear", "boat"]),
        "--validation_mask", str(["bear", "boat"]),
        "--validation_prompt", str(["clean background", "clean background"]),
    ]

    # 替换 validation 路径为基于 eval_davis 的完整路径
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

    # 去掉末尾的占位符 validation 参数，用正确的替代
    cmd = [c for c in cmd if not any(
        c.startswith(x) for x in ["['bear'", "['clean"]
    )]
    # 移除对应的 flag (已经在列表中)
    new_cmd = []
    skip_next = False
    for i, c in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if c in ("--validation_image", "--validation_mask", "--validation_prompt"):
            skip_next = True
            continue
        new_cmd.append(c)
    cmd = new_cmd

    cmd.extend(["--validation_image", str(val_images)])
    cmd.extend(["--validation_mask", str(val_masks)])
    cmd.extend(["--validation_prompt", str(val_prompts)])

    if args.checkpoints_total_limit:
        cmd.extend(["--checkpoints_total_limit", str(args.checkpoints_total_limit)])

    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])

    return cmd


def run_stage1(args=None):
    """运行 Stage 1 训练。"""
    if args is None:
        args = parse_args()

    paths = get_project_paths()
    cmd = build_stage1_cmd(paths, args)

    print("=" * 60)
    print("  DiffuEraser Stage 1 Training")
    print("=" * 60)
    print(f"  Work Dir:   {paths['work_dir']}")
    print(f"  GPUs:       {args.num_gpus}")
    print(f"  Max Steps:  {args.max_train_steps}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  LR:         {args.learning_rate}")
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
