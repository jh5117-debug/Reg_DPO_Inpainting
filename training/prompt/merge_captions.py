#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 DAVIS 和 YTVOS 的 caption YAML 文件为统一的训练 caption 文件。

Usage:
    python merge_captions.py \
        --davis_yaml captions/all_captions_BR.yaml \
        --ytvos_yaml captions/all_captions_ytvos.yaml \
        --output captions/all_captions_merged.yaml
"""

import argparse
import yaml
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge DAVIS and YTVOS captions")
    parser.add_argument("--davis_yaml", type=str, required=True,
                        help="Path to DAVIS captions YAML")
    parser.add_argument("--ytvos_yaml", type=str, required=True,
                        help="Path to YTVOS captions YAML")
    parser.add_argument("--output", type=str, required=True,
                        help="Output merged YAML path")
    args = parser.parse_args()

    merged = {}

    # Load DAVIS captions
    try:
        with open(args.davis_yaml, "r", encoding="utf-8") as f:
            davis_data = yaml.safe_load(f) or {}
        for k, v in davis_data.items():
            merged[f"davis_{k}"] = v
        print(f"Loaded {len(davis_data)} DAVIS captions")
    except FileNotFoundError:
        print(f"Warning: {args.davis_yaml} not found, skipping DAVIS")

    # Load YTVOS captions
    try:
        with open(args.ytvos_yaml, "r", encoding="utf-8") as f:
            ytvos_data = yaml.safe_load(f) or {}
        for k, v in ytvos_data.items():
            merged[f"ytvos_{k}"] = v
        print(f"Loaded {len(ytvos_data)} YTVOS captions")
    except FileNotFoundError:
        print(f"Warning: {args.ytvos_yaml} not found, skipping YTVOS")

    # Write merged
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# Merged captions (DAVIS + YTVOS)\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n")
        f.write(f"# Total entries: {len(merged)}\n\n")
        yaml.dump(merged, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"Merged {len(merged)} entries -> {args.output}")


if __name__ == "__main__":
    main()
