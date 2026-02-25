#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Captioning Script for YouTubeVOS (YTBV) Dataset
====================================================

为 YTVOS 数据集生成 caption，适配其 JPEGImages/{video_id}/ 目录结构。
使用 Qwen2.5-VL 模型生成场景描述，输出 YAML 格式供训练使用。

运行环境: conda activate qwen_env

使用示例:
    CUDA_VISIBLE_DEVICES=0 python generate_captions_ytvos.py \
        --ytvos_root /home/hj/Train_Diffueraser/dataset/YTBV \
        --model_path /home/hj/DiffuEraser_new/weights/Qwen2.5-VL-7B-Instruct \
        --output_dir captions \
        --device cuda \
        --force
"""

import os
import sys
import argparse
import warnings
import logging
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

import cv2
import numpy as np
from PIL import Image
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scene captions for YouTubeVOS dataset using VLM"
    )
    parser.add_argument("--ytvos_root", type=str, required=True,
                        help="Root directory of YTVOS dataset (e.g. .../YTBV)")
    parser.add_argument("--output_dir", type=str, default="captions",
                        help="Output directory for YAML files")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VLM model (local or HuggingFace)")
    parser.add_argument("--frame_strategy", type=str, default="middle",
                        choices=["middle", "multi_sample"],
                        help="Frame sampling strategy")
    parser.add_argument("--num_sample_frames", type=int, default=3,
                        help="Number of frames for 'multi_sample' strategy")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing YAML")
    return parser.parse_args()


def load_vlm_model(model_path: str, device: str):
    """Load Qwen2.5-VL model. Returns (processor, model) tuple."""
    print(f"[VLM] Loading model from {model_path}...")
    try:
        from transformers import AutoProcessor
        import torch
    except ImportError as e:
        print(f"[ERROR] Required packages not installed: {e}")
        sys.exit(1)

    model_class = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_class = Qwen2_5_VLForConditionalGeneration
        print("[VLM] Using Qwen2_5_VLForConditionalGeneration")
    except ImportError:
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model_class = Qwen2VLForConditionalGeneration
            print("[VLM] Using Qwen2VLForConditionalGeneration (fallback)")
        except ImportError:
            print("[ERROR] Qwen2 VL model class not available.")
            sys.exit(1)

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        print("[VLM] Model loaded successfully")
        return processor, model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)


def generate_caption(processor, model, image: Image.Image, device: str) -> str:
    """Generate scene caption using VLM."""
    import torch

    system_prompt = """You are a scene description assistant.
Your job is to describe the entire scene in the image in detail.

Rules:
1. Describe the environment, lighting, textures, colors, and spatial layout of the ENTIRE image.
2. Include foreground objects, people, animals, and background elements.
3. Output a single concise English sentence, maximum 30 words.
4. Focus on providing a comprehensive description of the visual content.

Example outputs:
- "a busy city street with cars, pedestrians, tall buildings, and bright neon signs"
- "a young woman running on a treadmill in a gym with large windows and exercise equipment"
- "a serene lake surrounded by pine trees with mountains in the distance under a clear blue sky"
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the scene in this image."},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

    if device == "cuda":
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    caption = output_text.strip().strip('"').strip("'")
    words = caption.split()
    if len(words) > 35:
        caption = " ".join(words[:30])
    return caption


def select_frame_indices(total_frames: int, strategy: str, num_samples: int = 3) -> list:
    if strategy == "middle":
        return [total_frames // 2]
    else:
        if total_frames <= num_samples:
            return list(range(total_frames))
        step = total_frames // (num_samples + 1)
        return [step * (i + 1) for i in range(num_samples)]


def main():
    args = parse_args()

    processor, model = load_vlm_model(args.model_path, args.device)
    model_name = Path(args.model_path).name

    images_root = os.path.join(args.ytvos_root, "JPEGImages")
    if not os.path.exists(images_root):
        print(f"[ERROR] YTVOS JPEGImages not found: {images_root}")
        return

    sequences = sorted([
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    ])
    print(f"[Batch] Found {len(sequences)} YTVOS sequences.")

    unified_results = {}

    for seq in sequences:
        video_dir = os.path.join(images_root, seq)
        output_yaml = os.path.join(args.output_dir, f"ytvos_{seq}.yaml")

        # Check existing
        if os.path.exists(output_yaml) and not args.force:
            try:
                with open(output_yaml, "r") as f:
                    existing = yaml.safe_load(f) or {}
                if existing.get("prompt") and existing["prompt"][0]:
                    print(f"[SKIP] {seq}")
                    unified_results[seq] = existing
                    continue
            except Exception:
                pass

        exts = (".png", ".jpg", ".jpeg")
        frames = sorted([f for f in os.listdir(video_dir) if f.lower().endswith(exts)])
        if not frames:
            print(f"[WARN] No frames in {seq}, skipping.")
            continue

        total_frames = len(frames)
        frame_indices = select_frame_indices(total_frames, args.frame_strategy, args.num_sample_frames)

        captions = []
        try:
            for idx in frame_indices:
                idx = min(idx, total_frames - 1)
                img_path = os.path.join(video_dir, frames[idx])
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)

                caption = generate_caption(processor, model, pil_image, args.device)
                captions.append(caption)
                print(f"  [{seq}] Frame {idx}: {caption}")

            if not captions:
                print(f"[WARN] No captions generated for {seq}")
                continue

            final_caption = max(captions, key=len) if len(captions) > 1 else captions[0]
            n_prompt = "blurry, flickering, distorted, artifacts, text, watermark, low quality"

            result = {
                "prompt": [final_caption],
                "n_prompt": [n_prompt],
                "text_guidance_scale": 2.0,
                "prompt_source": "auto_ytvos",
                "prompt_model": model_name,
                "prompt_timestamp": datetime.now().isoformat(),
            }

            Path(output_yaml).parent.mkdir(parents=True, exist_ok=True)
            with open(output_yaml, "w", encoding="utf-8") as f:
                yaml.dump(result, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            unified_results[seq] = result

        except Exception as e:
            print(f"[ERROR] {seq}: {e}")
            continue

    # Write unified YAML
    unified_path = os.path.join(args.output_dir, "all_captions_ytvos.yaml")
    Path(unified_path).parent.mkdir(parents=True, exist_ok=True)
    with open(unified_path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-generated YTVOS captions\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n")
        f.write(f"# Total videos: {len(unified_results)}\n\n")
        yaml.dump(unified_results, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"\n[Unified YAML] Written {len(unified_results)} entries to: {unified_path}")
    print("[Done] All tasks completed!")


if __name__ == "__main__":
    main()
