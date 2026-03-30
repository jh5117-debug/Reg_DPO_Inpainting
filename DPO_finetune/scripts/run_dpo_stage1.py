#!/usr/bin/env python
# coding=utf-8
"""
run_dpo_stage1.py — DPO Stage 1 训练入口

自动检测项目根目录（DPO_finetune/scripts/ 的祖父目录），无需硬编码路径。
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def get_project_root():
    """项目根目录 = DPO_finetune/scripts/ 的上两级。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_cmd(project_root, args):
    weights_dir = args.weights_dir or os.path.join(project_root, "weights")
    dpo_data_root = args.dpo_data_root or os.path.join(project_root, "data", "DPO_Finetune_data")
    ref_model_path = args.ref_model_path or os.path.join(
        project_root, "finetune-stage2", "converted_weights_step34000"
    )
    eval_dir = args.val_data_dir or os.path.join(project_root, "data_val")

    cmd = [
        "accelerate", "launch",
        "--num_processes", str(args.num_gpus),
        "--mixed_precision", args.mixed_precision,
        "DPO_finetune/train_dpo_stage1.py",
        "--base_model_name_or_path", os.path.join(weights_dir, "stable-diffusion-v1-5"),
        "--vae_path", os.path.join(weights_dir, "sd-vae-ft-mse"),
        "--ref_model_path", ref_model_path,
        "--dpo_data_root", dpo_data_root,
        "--output_dir", args.output_dir or os.path.join(project_root, "dpo-finetune-stage1"),
        "--logging_dir", "logs-dpo-stage1",
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
        "--set_grads_to_none",
        "--resume_from_checkpoint", "latest",
    ]

    if args.checkpoints_total_limit:
        cmd.extend(["--checkpoints_total_limit", str(args.checkpoints_total_limit)])
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    if args.chunk_aligned:
        cmd.append("--chunk_aligned")

    return cmd, dpo_data_root, ref_model_path


def get_output_dir(project_root, args):
    return args.output_dir or os.path.join(project_root, "dpo-finetune-stage1")


def find_slurm_log_path(project_root):
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    candidate_dirs = [
        Path(project_root) / "logs",
        Path(project_root) / "log",
        Path(project_root),
    ]

    if slurm_job_id:
        for log_dir in candidate_dirs:
            exact_path = log_dir / f"dpo-stage1-{slurm_job_id}.out"
            if exact_path.exists():
                return exact_path

    candidates = []
    for log_dir in candidate_dirs:
        if not log_dir.exists():
            continue
        candidates.extend(log_dir.glob("dpo-stage1-*.out"))

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def upload_full_crash_log_to_wandb(project_root, output_dir, returncode):
    run_info_path = Path(output_dir) / "wandb_run_info.json"
    if not run_info_path.exists():
        print(f"[launcher] W&B run info not found at {run_info_path}, skip crash log upload.", file=sys.stderr)
        return

    try:
        import wandb
    except Exception as e:
        print(f"[launcher] Failed to import wandb for crash upload: {e}", file=sys.stderr)
        return

    try:
        with open(run_info_path) as f:
            run_info = json.load(f)

        run = wandb.init(
            project=run_info["project"],
            entity=run_info.get("entity"),
            id=run_info["id"],
            resume="allow",
            name=run_info.get("name"),
        )

        slurm_log_path = find_slurm_log_path(project_root)
        if slurm_log_path is None or not slurm_log_path.exists():
            wandb.alert(
                title="DPO Stage 1 Launcher Failed",
                text=f"Launcher exited with code {returncode}, but no Slurm log file was found.",
                level=wandb.AlertLevel.ERROR,
            )
            wandb.finish(exit_code=returncode)
            return

        full_text = slurm_log_path.read_text(errors="replace")
        tail_chars = 3500
        full_log_name = "slurm_full_crash_output.out"
        full_log_copy = Path(run.dir) / full_log_name
        shutil.copyfile(slurm_log_path, full_log_copy)
        wandb.save(str(full_log_copy), policy="now")

        summary_name = "launcher_crash_summary.txt"
        summary_copy = Path(run.dir) / summary_name
        with open(summary_copy, "w") as f:
            f.write(f"returncode: {returncode}\n")
            f.write(f"slurm_log_path: {slurm_log_path}\n")
            f.write(f"wandb_run_url: {run_info.get('url')}\n")
            f.write("\n===== LOG TAIL =====\n")
            f.write(full_text[-tail_chars:])
            f.write("\n")
        wandb.save(str(summary_copy), policy="now")

        if getattr(wandb, "run", None) is not None:
            wandb.run.summary["launcher_returncode"] = returncode
            wandb.run.summary["slurm_full_log_uploaded"] = True
            wandb.run.summary["slurm_log_path"] = str(slurm_log_path)
            wandb.run.summary["slurm_full_log_file"] = full_log_name

        wandb.alert(
            title="DPO Stage 1 Launcher Failed",
            text=(
                f"Launcher exited with code {returncode}.\n"
                f"Full Slurm log uploaded as `{full_log_name}`.\n\n"
                f"Tail of the full log:\n```\n{full_text[-tail_chars:]}\n```"
            ),
            level=wandb.AlertLevel.ERROR,
        )
        wandb.finish(exit_code=returncode)
    except Exception:
        print("[launcher] Failed to upload full crash log to W&B.", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def run(args=None):
    if args is None:
        args = parse_args()
    project_root = get_project_root()
    output_dir = get_output_dir(project_root, args)
    cmd, dpo_data_root, ref_model_path = build_cmd(project_root, args)

    print("=" * 60)
    print("  DiffuEraser DPO Stage 1 Training")
    print("=" * 60)
    print(f"  Project Root:    {project_root}")
    print(f"  DPO Data Root:   {dpo_data_root}")
    print(f"  Ref Model:       {ref_model_path}")
    print(f"  GPUs:            {args.num_gpus}")
    print(f"  Max Steps:       {args.max_train_steps}")
    print(f"  Beta DPO:        {args.beta_dpo}")
    print(f"  LR:              {args.learning_rate}")
    print("=" * 60)
    print(f"\n  Command:\n  {' '.join(cmd[:6])} \\\n    " + " \\\n    ".join(cmd[6:]))
    print()

    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        upload_full_crash_log_to_wandb(project_root, output_dir, result.returncode)
    return result.returncode


def parse_args():
    parser = argparse.ArgumentParser(description="DPO Stage 1 Training Entry")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--weights_dir", type=str, default=None)
    parser.add_argument("--dpo_data_root", type=str, default=None)
    parser.add_argument("--ref_model_path", type=str, default=None)
    parser.add_argument("--val_data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=20000)
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
