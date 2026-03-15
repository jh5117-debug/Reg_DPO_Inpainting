#!/usr/bin/env python
# coding=utf-8
"""
run_train_all.py — 一键 Stage 1 + Stage 2 训练

顺序执行 Stage 1 和 Stage 2 训练。Stage 1 完成后自动衔接 Stage 2。
两个阶段的权重转换均在各自的训练脚本末尾自动执行。

Usage:
    python run_train_all.py                # 单卡
    python run_train_all.py --num_gpus 8   # 8 卡
"""

import argparse
import sys

from run_train_stage1 import run_stage1
from run_train_stage2 import run_stage2


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-click Stage 1 + Stage 2 Training"
    )

    # 通用参数
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录 (含 DAVIS/ YTBV/)。默认: <project_root>/data/")
    parser.add_argument("--weights_dir", type=str, default=None,
                        help="权重目录。默认: <project_root>/weights/")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Stage 1
    parser.add_argument("--s1_batch_size", type=int, default=1)
    parser.add_argument("--s1_gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--s1_learning_rate", type=float, default=5e-6)
    parser.add_argument("--s1_lr_scheduler", type=str, default="constant")
    parser.add_argument("--s1_lr_warmup_steps", type=int, default=500)
    parser.add_argument("--s1_max_train_steps", type=int, default=26000)
    parser.add_argument("--s1_checkpointing_steps", type=int, default=2000)
    parser.add_argument("--s1_checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--s1_validation_steps", type=int, default=2000)
    parser.add_argument("--s1_nframes", type=int, default=10)

    # Stage 2
    parser.add_argument("--s2_batch_size", type=int, default=1)
    parser.add_argument("--s2_gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--s2_learning_rate", type=float, default=5e-6)
    parser.add_argument("--s2_lr_scheduler", type=str, default="constant")
    parser.add_argument("--s2_lr_warmup_steps", type=int, default=500)
    parser.add_argument("--s2_max_train_steps", type=int, default=26000)
    parser.add_argument("--s2_checkpointing_steps", type=int, default=2000)
    parser.add_argument("--s2_checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--s2_validation_steps", type=int, default=2000)
    parser.add_argument("--s2_nframes", type=int, default=22)

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Construct Stage 1 args ---
    class Stage1Args:
        pass

    s1_args = Stage1Args()
    s1_args.num_gpus = args.num_gpus
    s1_args.batch_size = args.s1_batch_size
    s1_args.gradient_accumulation_steps = args.s1_gradient_accumulation_steps
    s1_args.learning_rate = args.s1_learning_rate
    s1_args.lr_scheduler = args.s1_lr_scheduler
    s1_args.lr_warmup_steps = args.s1_lr_warmup_steps
    s1_args.max_train_steps = args.s1_max_train_steps
    s1_args.checkpointing_steps = args.s1_checkpointing_steps
    s1_args.checkpoints_total_limit = args.s1_checkpoints_total_limit
    s1_args.validation_steps = args.s1_validation_steps
    s1_args.nframes = args.s1_nframes
    s1_args.seed = args.seed
    s1_args.mixed_precision = args.mixed_precision
    s1_args.wandb_project = args.wandb_project
    s1_args.wandb_entity = args.wandb_entity
    s1_args.data_dir = args.data_dir
    s1_args.weights_dir = args.weights_dir

    print("=" * 60)
    print("  DiffuEraser Full Pipeline: Stage 1 + Stage 2")
    print("=" * 60)

    # --- Stage 1 ---
    print("\n" + "=" * 60)
    print("  [1/2]  Starting Stage 1 Training ...")
    print("=" * 60 + "\n")

    ret = run_stage1(s1_args)
    if ret != 0:
        print(f"\n  ❌ Stage 1 failed with return code {ret}. Aborting.")
        return ret

    print("\n  ✅ Stage 1 completed successfully!\n")

    # --- Construct Stage 2 args ---
    class Stage2Args:
        pass

    s2_args = Stage2Args()
    s2_args.num_gpus = args.num_gpus
    s2_args.batch_size = args.s2_batch_size
    s2_args.gradient_accumulation_steps = args.s2_gradient_accumulation_steps
    s2_args.learning_rate = args.s2_learning_rate
    s2_args.lr_scheduler = args.s2_lr_scheduler
    s2_args.lr_warmup_steps = args.s2_lr_warmup_steps
    s2_args.max_train_steps = args.s2_max_train_steps
    s2_args.checkpointing_steps = args.s2_checkpointing_steps
    s2_args.checkpoints_total_limit = args.s2_checkpoints_total_limit
    s2_args.validation_steps = args.s2_validation_steps
    s2_args.nframes = args.s2_nframes
    s2_args.seed = args.seed
    s2_args.mixed_precision = args.mixed_precision
    s2_args.wandb_project = args.wandb_project
    s2_args.wandb_entity = args.wandb_entity
    s2_args.data_dir = args.data_dir
    s2_args.weights_dir = args.weights_dir
    s2_args.pretrained_stage1 = None  # 自动从 finetune-stage1/converted_weights/ 获取

    # --- Stage 2 ---
    print("\n" + "=" * 60)
    print("  [2/2]  Starting Stage 2 Training ...")
    print("=" * 60 + "\n")

    ret = run_stage2(s2_args)
    if ret != 0:
        print(f"\n  ❌ Stage 2 failed with return code {ret}.")
        return ret

    print("\n  ✅ Stage 2 completed successfully!")
    print("  ✅ Full pipeline finished!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
