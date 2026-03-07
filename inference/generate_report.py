# -*- coding: utf-8 -*-
"""
generate_report.py — 12 实验结果汇总报告生成器

用法:
    python generate_report.py dir1 dir2 ... dir12

从每个实验目录的 summary.json 中读取数据，
输出 experiment_report.md，包含 OR/BR 分类表格 + 实验总结。
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime


PIXEL_KEYS = ["psnr_mean", "ssim_mean", "lpips_mean", "ewarp", "as_mean", "is_mean"]
PIXEL_LABELS = {"psnr_mean": "PSNR↑", "ssim_mean": "SSIM↑", "lpips_mean": "LPIPS↓",
                "ewarp": "Ewarp↓", "as_mean": "AS↑", "is_mean": "IS↑"}
VBENCH_DIMS = [
    "subject_consistency", "background_consistency",
    "temporal_flickering", "motion_smoothness",
    "aesthetic_quality", "imaging_quality",
]
VBENCH_LABELS = {
    "subject_consistency": "Subj_Con↑", "background_consistency": "BG_Con↑",
    "temporal_flickering": "Temp_Flk↑", "motion_smoothness": "Mot_Smo↑",
    "aesthetic_quality": "Aesth_Q↑", "imaging_quality": "Img_Q↑",
}


def _is_dual_mode(per_video):
    """Check if per_video entries use baseline_/text_ prefixed keys."""
    if not per_video:
        return False
    sample = per_video[0]
    return any(k.startswith(("baseline_", "text_")) for k in sample)


def _get_vbench(entry, mode_prefix=None):
    """Get vbench dict from a per_video entry, with fallback."""
    if mode_prefix:
        d = entry.get(f"{mode_prefix}_vbench")
        if d:
            return d
    return entry.get("vbench", {})


def _get_metrics(entry, mode_prefix=None):
    """Get metrics dict from a per_video entry, with fallback."""
    if mode_prefix:
        d = entry.get(f"{mode_prefix}_metrics")
        if d:
            return d
    return entry.get("metrics", {})


def load_summary(exp_dir):
    """Load summary.json from experiment directory."""
    p = Path(exp_dir) / "summary.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def avg_metric(per_video, mode_prefix, key):
    """Compute average of a metric across all videos.
    mode_prefix: 'baseline', 'text', or None (single-mode fallback).
    """
    vals = []
    for v in per_video:
        val = _get_metrics(v, mode_prefix).get(key)
        if val is None:
            val = _get_vbench(v, mode_prefix).get(key)
        if val is not None and isinstance(val, (int, float)) and val >= 0:
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def fmt(v, prec=4):
    if v is None:
        return "N/A"
    return f"{v:.{prec}f}"


def parse_exp_name(name):
    """Parse experiment dir name to extract metadata."""
    parts = name.split("_")
    info = {"name": name, "cfg_type": "", "steps": "", "dataset": "",
            "blend": "", "dilation": "", "gs": ""}

    if name.startswith("smallcfg"):
        info["cfg_type"] = "smallcfg"
    elif name.startswith("normalcfg"):
        info["cfg_type"] = "normalcfg"

    for p in parts:
        if p.endswith("step"):
            info["steps"] = p
        if p in ("OR", "BR"):
            info["dataset"] = p

    if "noblend" in name:
        info["blend"] = "No"
        info["dilation"] = "0"
    elif "blend" in name:
        info["blend"] = "Yes"
    if "dil8" in name:
        info["dilation"] = "8"
    if "nodil" in name:
        info["dilation"] = "0"

    for p in parts:
        if p.startswith("gs"):
            info["gs"] = p[2:]

    return info


def generate_detailed_table(experiments, dataset_filter, has_gt):
    """Generate markdown table for one dataset type (OR or BR)."""
    filtered = [(name, data) for name, data in experiments if parse_exp_name(name)["dataset"] == dataset_filter]
    if not filtered:
        return ""

    lines = []

    # Detect if data uses dual mode (baseline/text) or single mode
    dual_mode = any(_is_dual_mode(data.get("per_video", [])) for _, data in filtered)
    if dual_mode:
        modes = [("baseline", "BL"), ("text", "TG")]
    else:
        modes = [(None, None)]

    # Sort by VBench average descending (best first)
    def _vbench_avg(nd):
        pv = nd[1].get("per_video", [])
        vs = [avg_metric(pv, modes[0][0], d) for d in VBENCH_DIMS]
        vs = [x for x in vs if x is not None]
        return sum(vs) / len(vs) if vs else 0
    filtered.sort(key=_vbench_avg, reverse=True)

    # VBench table
    lines.append(f"### VBench Scores ({dataset_filter})")
    lines.append("")
    header = "| Experiment |"
    if dual_mode:
        header += " Mode |"
    for dim in VBENCH_DIMS:
        header += f" {VBENCH_LABELS[dim]} |"
    header += " **Avg** |"
    lines.append(header)

    n_cols = len(VBENCH_DIMS) + (3 if dual_mode else 2)
    sep = "|" + "|".join(["---"] * n_cols) + "|"
    lines.append(sep)

    for name, data in filtered:
        info = parse_exp_name(name)
        per_video = data.get("per_video", [])
        short = f"{info['cfg_type']}_{info['steps']}_{'blend' if info['blend']=='Yes' else 'noblend'}_gs{info['gs']}"

        for mode_prefix, mode_label in modes:
            row = f"| {short} |"
            if dual_mode:
                row += f" {mode_label} |"
            vals = []
            for dim in VBENCH_DIMS:
                v = avg_metric(per_video, mode_prefix, dim)
                row += f" {fmt(v)} |"
                if v is not None:
                    vals.append(v)
            avg_val = sum(vals) / len(vals) if vals else None
            row += f" **{fmt(avg_val)}** |"
            lines.append(row)

    lines.append("")

    # Pixel metrics table (BR only)
    if has_gt:
        lines.append(f"### Pixel Metrics ({dataset_filter}, GT available)")
        lines.append("")
        header = "| Experiment |"
        if dual_mode:
            header += " Mode |"
        for pk in PIXEL_KEYS:
            header += f" {PIXEL_LABELS[pk]} |"
        lines.append(header)

        n_cols = len(PIXEL_KEYS) + (2 if dual_mode else 1)
        sep = "|" + "|".join(["---"] * n_cols) + "|"
        lines.append(sep)

        # Sort pixel metrics by PSNR descending
        px_sorted = sorted(filtered, key=lambda nd: avg_metric(nd[1].get("per_video", []), modes[0][0], "psnr_mean") or 0, reverse=True)
        for name, data in px_sorted:
            info = parse_exp_name(name)
            per_video = data.get("per_video", [])
            short = f"{info['cfg_type']}_{info['steps']}_{'blend' if info['blend']=='Yes' else 'noblend'}_gs{info['gs']}"

            for mode_prefix, mode_label in modes:
                row = f"| {short} |"
                if dual_mode:
                    row += f" {mode_label} |"
                for pk in PIXEL_KEYS:
                    v = avg_metric(per_video, mode_prefix, pk)
                    row += f" {fmt(v)} |"
                lines.append(row)

        lines.append("")

    return "\n".join(lines)


def generate_cross_exp_comparison(experiments):
    """Generate cross-experiment comparison summary."""
    lines = []
    dual_mode = any(_is_dual_mode(data.get("per_video", [])) for _, data in experiments)

    lines.append("## Cross-Experiment Comparison (VBench Average)")
    lines.append("")
    if dual_mode:
        lines.append("Average VBench score across all videos, **Text-Guided (TG)** mode only:")
    else:
        lines.append("Average VBench score across all videos:")
    lines.append("")

    header = "| Experiment | Dataset | Blend | Dil | Steps | GS |"
    for dim in VBENCH_DIMS:
        header += f" {VBENCH_LABELS[dim]} |"
    header += " **Avg** |"
    lines.append(header)

    n_cols = len(VBENCH_DIMS) + 7
    sep = "|" + "|".join(["---"] * n_cols) + "|"
    lines.append(sep)

    mode_prefix = "text" if dual_mode else None

    # Sort by VBench average descending
    def _cross_avg(nd):
        pv = nd[1].get("per_video", [])
        vs = [avg_metric(pv, mode_prefix, d) for d in VBENCH_DIMS]
        vs = [x for x in vs if x is not None]
        return sum(vs) / len(vs) if vs else 0
    sorted_exps = sorted(experiments, key=_cross_avg, reverse=True)

    for name, data in sorted_exps:
        info = parse_exp_name(name)
        per_video = data.get("per_video", [])
        short = f"{info['cfg_type']}_{info['steps']}"

        row = f"| {short} | {info['dataset']} | {info['blend']} | {info['dilation']} | {info['steps']} | {info['gs']} |"
        vals = []
        for dim in VBENCH_DIMS:
            v = avg_metric(per_video, mode_prefix, dim)
            row += f" {fmt(v)} |"
            if v is not None:
                vals.append(v)
        avg_val = sum(vals) / len(vals) if vals else None
        row += f" **{fmt(avg_val)}** |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def generate_delta_table(experiments):
    """Generate TG - BL delta table. Only applicable for dual-mode data."""
    # Skip entirely when data is single-mode (no baseline/text distinction)
    dual_mode = any(_is_dual_mode(data.get("per_video", [])) for _, data in experiments)
    if not dual_mode:
        return ""

    lines = []
    lines.append("## Prompt Effect Analysis (TG − BL Delta)")
    lines.append("")
    lines.append("Positive delta = prompt improved the metric.")
    lines.append("")

    header = "| Experiment | Dataset |"
    for dim in VBENCH_DIMS:
        header += f" Δ{VBENCH_LABELS[dim]} |"
    lines.append(header)

    n_cols = len(VBENCH_DIMS) + 2
    sep = "|" + "|".join(["---"] * n_cols) + "|"
    lines.append(sep)

    for name, data in experiments:
        info = parse_exp_name(name)
        per_video = data.get("per_video", [])
        short = f"{info['cfg_type']}_{info['steps']}_{'blend' if info['blend']=='Yes' else 'noblend'}_gs{info['gs']}"

        row = f"| {short} | {info['dataset']} |"
        for dim in VBENCH_DIMS:
            bl_v = avg_metric(per_video, "baseline", dim)
            tg_v = avg_metric(per_video, "text", dim)
            if bl_v is not None and tg_v is not None:
                d = tg_v - bl_v
                row += f" {d:+.4f} |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_report.py exp_dir1 exp_dir2 ...")
        sys.exit(1)

    exp_dirs = sys.argv[1:]
    experiments = []

    for d in exp_dirs:
        data = load_summary(d)
        if data is None:
            print(f"[WARN] summary.json not found in {d}, skipping.")
            continue
        name = os.path.basename(d.rstrip("/"))
        experiments.append((name, data))

    if not experiments:
        print("[ERROR] No valid experiment data found.")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiment(s).")

    report = []
    report.append(f"# 12-Experiment Comparison Report")
    report.append(f"")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"")
    report.append(f"## Experiment Configuration")
    report.append(f"")
    report.append(f"| # | Directory | ckpt | GS | Dataset | Blend | Dilation | Videos |")
    report.append(f"|---|-----------|------|----|---------|-------|----------|--------|")

    for i, (name, data) in enumerate(experiments, 1):
        cfg = data.get("config", {})
        info = parse_exp_name(name)
        n = data.get("num_videos", 0)
        report.append(f"| {i} | `{name}` | {cfg.get('ckpt', '?')} | {cfg.get('text_guidance_scale', '?')} | {info['dataset']} | {info['blend']} | {info['dilation']} | {n} |")

    report.append("")

    # OR section
    or_exps = [(n, d) for n, d in experiments if parse_exp_name(n)["dataset"] == "OR"]
    if or_exps:
        report.append("---")
        report.append("")
        report.append("## Object Removal (OR) Results")
        report.append("")
        report.append(generate_detailed_table(experiments, "OR", has_gt=False))

    # BR section
    br_exps = [(n, d) for n, d in experiments if parse_exp_name(n)["dataset"] == "BR"]
    if br_exps:
        report.append("---")
        report.append("")
        report.append("## Background Restoration (BR) Results")
        report.append("")
        report.append(generate_detailed_table(experiments, "BR", has_gt=True))

    # Cross-experiment comparison
    report.append("---")
    report.append("")
    report.append(generate_cross_exp_comparison(experiments))

    # Delta analysis
    report.append("---")
    report.append("")
    report.append(generate_delta_table(experiments))

    # Summary
    report.append("---")
    report.append("")
    report.append("## Experiment Summary")
    report.append("")
    report.append("### Key Findings")
    report.append("")
    report.append("| Comparison | Question |")
    report.append("|-----------|----------|")
    report.append("| smallcfg_2step vs smallcfg_4step | Does increasing steps improve quality? |")
    report.append("| smallcfg_4step vs normalcfg_4step | Does the CFG-trained LoRA produce better text-guided results? |")
    report.append("| noblend vs blend+dil8 | Does mask blending improve visual quality? |")
    report.append("| BL vs TG (all configs) | Does text guidance consistently help across ckpts? |")
    report.append("")
    report.append("> Review the tables above to answer these questions based on your metric priorities.")
    report.append("")

    out_path = "experiment_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Report saved to: {out_path}")
    print(f"Total experiments: {len(experiments)}")


if __name__ == "__main__":
    main()
