#!/usr/bin/env python
# coding=utf-8
"""
run_dpo_stage2.py — DPO Stage 2 训练入口

前置条件：DPO Stage 1 训练已完成。
"""

import argparse
import os
import subprocess
import sys


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_cmd(project_root, args):
    weights_dir = args.weights_dir or os.path.join(project_root, "weights")
    dpo_data_root = args.dpo_data_root or os.path.join(project_root, "data", "DPO_Finetune_data")

    pretrained_dpo_stage1 = args.pretrained_dpo_stage1 or os.path.join(
        project_root, "dpo-finetune-stage1", "best_weights"
    )
    ref_model_path = args.ref_model_path or os.path.join(
        project_root, "finetune-stage2", "converted_weights_step34000"
    )
    baseline_unet_path = args.baseline_unet_path or os.path.join(weights_dir, "diffuEraser")
    eval_dir = args.val_data_dir or os.path.join(project_root, "data_val")

    cmd = [
        "accelerate", "launch",
        "--num_processes", str(args.num_gpus),
        "--mixed_precision", args.mixed_precision,
        "DPO_finetune/train_dpo_stage2.py",
        "--base_model_name_or_path", os.path.join(weights_dir, "stable-diffusion-v1-5"),
        "--pretrained_dpo_stage1", pretrained_dpo_stage1,
        "--baseline_unet_path", baseline_unet_path,
        "--ref_model_path", ref_model_path,
        "--vae_path", os.path.join(weights_dir, "sd-vae-ft-mse"),
        "--dpo_data_root", dpo_data_root,
        "--output_dir", args.output_dir or os.path.join(project_root, "dpo-finetune-stage2"),
        "--logging_dir", "logs-dpo-stage2",
        "--val_data_dir", eval_dir,
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
        "--beta_dpo", str(args.beta_dpo),
        "--davis_oversample", str(args.davis_oversample),
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
    if args.chunk_aligned:
        cmd.append("--chunk_aligned")

    # Metric model paths
    raft_path = os.path.join(weights_dir, "propainter", "raft-things.pth")
    i3d_path = os.path.join(weights_dir, "i3d_rgb_imagenet.pt")
    clip_path = os.path.join(weights_dir, "open_clip", "ViT-H-14")

    if os.path.exists(raft_path):
        cmd.extend(["--raft_model_path", raft_path])
    if os.path.exists(i3d_path):
        cmd.extend(["--i3d_model_path", i3d_path])
    if os.path.isdir(clip_path):
        cmd.extend(["--clip_model_path", clip_path])

    return cmd, dpo_data_root, pretrained_dpo_stage1, ref_model_path


def run(args=None):
    if args is None:
        args = parse_args()
    project_root = get_project_root()
    cmd, dpo_data_root, dpo_s1, ref_model = build_cmd(project_root, args)

    print("=" * 60)
    print("  DiffuEraser DPO Stage 2 Training")
    print("=" * 60)
    print(f"  Project Root:       {project_root}")
    print(f"  DPO Data Root:      {dpo_data_root}")
    print(f"  DPO Stage 1:        {dpo_s1}")
    print(f"  Ref Model:          {ref_model}")
    print(f"  GPUs:               {args.num_gpus}")
    print(f"  Max Steps:          {args.max_train_steps}")
    print(f"  Beta DPO:           {args.beta_dpo}")
    print(f"  LR:                 {args.learning_rate}")
    print("=" * 60)
    print(f"\n  Command:\n  {' '.join(cmd[:6])} \\\n    " + " \\\n    ".join(cmd[6:]))
    print()

    if not os.path.exists(dpo_s1):
        print(f"\n  ⚠️  DPO Stage 1 weights not found at: {dpo_s1}")
        print("  请先运行 DPO Stage 1 训练，或指定 --pretrained_dpo_stage1 参数。")
        return 1

    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Stage 2 Training Entry")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--weights_dir", type=str, default=None)
    parser.add_argument("--dpo_data_root", type=str, default=None)
    parser.add_argument("--pretrained_dpo_stage1", type=str, default=None)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--baseline_unet_path", type=str, default=None)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=30000)
    parser.add_argument("--checkpointing_steps", type=int, default=2000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--validation_steps", type=int, default=2000)
    parser.add_argument("--nframes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--wandb_project", type=str, default="DPO_Diffueraser")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--beta_dpo", type=float, default=2500.0)
    parser.add_argument("--davis_oversample", type=int, default=10)
    parser.add_argument("--chunk_aligned", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run())
