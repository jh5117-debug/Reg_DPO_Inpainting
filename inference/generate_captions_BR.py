#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Captioning Script for OR Scene (独立环境运行)
==================================================

从粗修复视频中提取帧，用 mask 遮盖被移除区域，然后用 VLM 生成场景描述。
输出标准 YAML 文件供 run_OR.py 使用。

运行环境要求:
    conda create -n caption_env python=3.10
    pip install transformers>=4.45 torch torchvision pillow pyyaml opencv-python-headless

使用示例:
    python generate_captions.py \
        --video_path results_phase1/bear/diffueraser.mp4 \
        --mask_path /path/to/DAVIS/Annotations/Full-Resolution/bear \
        --output_yaml prompt_cache/bear.yaml \
        --model_path /path/to/Qwen2.5-VL-7B-Instruct

    # 批量处理整个 DAVIS 数据集:
CUDA_VISIBLE_DEVICES=1 python generate_captions.py \
    --dataset_root /home/hj/Train_Diffueraser/dataset/DAVIS \
    --model_path /home/hj/DiffuEraser_new/weights/Qwen2.5-VL-7B-Instruct \
    --batch_output_dir prompt_cache \
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

# ── Suppress noisy warnings ──
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
        description="Generate scene captions for OR inpainting using VLM"
    )
    parser.add_argument("--video_path", type=str,
                        help="Path to coarse inpainting video (Phase 1 output) or directory of frames")
    parser.add_argument("--mask_path", type=str,
                        help="Path to original mask (directory, video, or image)")
    parser.add_argument("--output_yaml", type=str,
                        help="Output YAML file path")
    parser.add_argument("--dataset_root", type=str,
                        help="Root directory of DAVIS dataset (e.g. .../DAVIS). Overrides video_path.")
    parser.add_argument("--batch_output_dir", type=str, default="prompt_cache",
                        help="Output directory for batch processing")
    parser.add_argument("--unified_yaml", type=str, default=None,
                        help="Path for a single unified YAML containing all video captions (batch mode only)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VLM model (local or HuggingFace)")
    parser.add_argument("--frame_strategy", type=str, default="middle",
                        choices=["middle", "multi_sample"],
                        help="Frame sampling strategy: 'middle' or 'multi_sample'")
    parser.add_argument("--num_sample_frames", type=int, default=3,
                        help="Number of frames for 'multi_sample' strategy")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing YAML")
    
    args = parser.parse_args()
    
    if not args.dataset_root and not args.video_path:
        parser.error("Either --video_path or --dataset_root must be provided.")
        
    return args


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a specific frame from video file. Returns RGB numpy array."""
    if os.path.isdir(video_path):
        # Directory of images
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(exts)])
        if not files:
            raise ValueError(f"No images found in {video_path}")
        
        frame_idx = min(frame_idx, len(files) - 1)
        img_path = os.path.join(video_path, files[frame_idx])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = min(frame_idx, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Cannot read frame {frame_idx} from {video_path}")
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_video_info(video_path: str):
    """Get video frame count."""
    if os.path.isdir(video_path):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = [f for f in os.listdir(video_path) if f.lower().endswith(exts)]
        return len(files)
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames


def read_mask_frame(mask_path: str, frame_idx: int, target_size: tuple) -> np.ndarray:
    """Read mask frame. Returns grayscale numpy array."""
    if os.path.isdir(mask_path):
        # Directory of mask images
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith(exts)])
        if not files:
            raise ValueError(f"No mask images found in {mask_path}")
        frame_idx = min(frame_idx, len(files) - 1)
        mask_file = os.path.join(mask_path, files[frame_idx])
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    elif mask_path.lower().endswith((".mp4", ".avi", ".mov")):
        # Video file
        cap = cv2.VideoCapture(mask_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read mask frame {frame_idx}")
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        # Single image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        raise ValueError(f"Cannot read mask from {mask_path}")
    
    # Resize to match video frame
    if mask.shape[:2] != (target_size[1], target_size[0]):
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    return mask


def apply_gray_mask(frame_rgb: np.ndarray, mask_gray: np.ndarray, 
                    fill_color: tuple = (128, 128, 128)) -> Image.Image:
    """Apply gray fill to masked region.
    
    Args:
        frame_rgb: RGB frame (H, W, 3)
        mask_gray: Grayscale mask (H, W), 255=hole
        fill_color: RGB color for fill (default gray)
    
    Returns:
        PIL Image with mask region filled with gray
    """
    mask_binary = (mask_gray > 127).astype(np.float32)
    mask_3ch = np.stack([mask_binary] * 3, axis=-1)
    
    fill = np.array(fill_color, dtype=np.float32).reshape(1, 1, 3)
    fill_img = np.ones_like(frame_rgb, dtype=np.float32) * fill
    
    result = frame_rgb.astype(np.float32) * (1 - mask_3ch) + fill_img * mask_3ch
    return Image.fromarray(result.astype(np.uint8))


def load_vlm_model(model_path: str, device: str):
    """Load Qwen2.5-VL model.
    
    Returns (processor, model) tuple.
    """
    print(f"[VLM] Loading model from {model_path}...")
    
    try:
        from transformers import AutoProcessor
        import torch
    except ImportError as e:
        print(f"[ERROR] Required packages not installed: {e}")
        print("Please run: pip install transformers>=4.45 torch")
        sys.exit(1)
    
    # Try to import Qwen2.5-VL specific class first, then fall back to Qwen2-VL
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
            print("[ERROR] Neither Qwen2_5_VLForConditionalGeneration nor Qwen2VLForConditionalGeneration available.")
            print("Please run: pip install --upgrade transformers>=4.49")
            sys.exit(1)
    
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = model_class.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"[VLM] Model loaded successfully")
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
    
    user_prompt = "Describe the scene in this image."
    
    # Build conversation for Qwen2-VL
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    
    # Process with Qwen2-VL format
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    
    if device == "cuda":
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
        )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    # Clean up
    caption = output_text.strip().strip('"').strip("'")
    
    # Truncate to max 30 words if needed
    words = caption.split()
    if len(words) > 35:
        caption = " ".join(words[:30])
    
    return caption


def select_frame_indices(total_frames: int, strategy: str, num_samples: int = 3) -> list:
    """Select frame indices based on strategy."""
    if strategy == "middle":
        return [total_frames // 2]
    else:  # multi_sample
        if total_frames <= num_samples:
            return list(range(total_frames))
        step = total_frames // (num_samples + 1)
        return [step * (i + 1) for i in range(num_samples)]


def write_yaml(output_path: str, prompt: str, n_prompt: str, 
               model_name: str, frame_idx: int, video_path: str):
    """Write prompt configuration to YAML."""
    config = {
        "prompt": [prompt],
        "n_prompt": [n_prompt],
        "text_guidance_scale": 2.0,
        "prompt_source": "auto_masked",
        "prompt_model": model_name,
        "prompt_timestamp": datetime.now().isoformat(),
        "prompt_frame_idx": frame_idx,
        "source_video": video_path,
    }
    
    # Create parent directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by generate_captions.py\n")
        f.write(f"# Video: {video_path}\n\n")
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"[YAML] Written to: {output_path}")


def process_single_video(
    video_path: str, mask_path: str, output_yaml: str, 
    processor, model, model_name: str, args
):
    """Process a single video sequence.
    
    Returns:
        dict with caption info (for unified YAML), or None if skipped/failed.
    """
    # Check if output exists
    if os.path.exists(output_yaml) and not args.force:
        try:
            with open(output_yaml, "r") as f:
                existing = yaml.safe_load(f) or {}
            if existing.get("prompt") and existing["prompt"][0]:
                print(f"[SKIP] YAML already exists: {output_yaml}")
                # Return existing data for unified YAML
                return existing
        except Exception:
            pass

    print(f"[{Path(video_path).name}] Processing...")

    # Get video info
    try:
        total_frames = get_video_info(video_path)
    except Exception as e:
        print(f"[ERROR] Failed to get info for {video_path}: {e}")
        return None

    print(f"  Frames: {total_frames}")
    
    # Select frames
    frame_indices = select_frame_indices(total_frames, args.frame_strategy, args.num_sample_frames)
    
    # Process frames
    captions = []
    selected_frame_idx = frame_indices[0]
    
    try:
        for idx in frame_indices:
            # Read video frame
            frame_rgb = read_video_frame(video_path, idx)
            h, w = frame_rgb.shape[:2]
            
            # Use original frame directly
            pil_image = Image.fromarray(frame_rgb)
            
            # Generate caption
            caption = generate_caption(processor, model, pil_image, args.device)
            captions.append(caption)
            print(f"  Frame {idx}: {caption}")

        # Select best caption (longest if multi_sample, or just the one)
        if len(captions) > 1:
            final_caption = max(captions, key=len)
            print(f"  Selected: {final_caption}")
        else:
            final_caption = captions[0]
        
        # Default negative prompt
        n_prompt = "blurry, flickering, distorted, artifacts, text, watermark, low quality, person, people, human"
        
        # Write per-video YAML
        write_yaml(
            output_yaml,
            final_caption,
            n_prompt,
            model_name,
            selected_frame_idx,
            video_path,
        )
        
        # Return data for unified YAML
        return {
            "prompt": [final_caption],
            "n_prompt": [n_prompt],
            "text_guidance_scale": 2.0,
            "prompt_source": "auto_masked",
            "prompt_model": model_name,
            "prompt_timestamp": datetime.now().isoformat(),
            "prompt_frame_idx": selected_frame_idx,
            "source_video": video_path,
        }
    except Exception as e:
        print(f"[ERROR] Failed processing {video_path}: {e}")
        return None


def write_unified_yaml(unified_path: str, all_results: dict):
    """Write a single unified YAML containing all video captions.
    
    Format:
        video_name_1:
            prompt: [...]
            n_prompt: [...]
            text_guidance_scale: 2.0
            ...
        video_name_2:
            ...
    """
    Path(unified_path).parent.mkdir(parents=True, exist_ok=True)
    with open(unified_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated unified YAML by generate_captions.py\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n")
        f.write(f"# Total videos: {len(all_results)}\n\n")
        yaml.dump(all_results, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"\n[Unified YAML] Written {len(all_results)} entries to: {unified_path}")


def main():
    args = parse_args()
    
    # Load VLM model once
    processor, model = load_vlm_model(args.model_path, args.device)
    model_name = Path(args.model_path).name

    if args.dataset_root:
        # Batch processing mode
        print(f"[Batch] Processing dataset: {args.dataset_root}")
        images_root = os.path.join(args.dataset_root, "JPEGImages", "480p")
        masks_root = os.path.join(args.dataset_root, "Annotations", "480p")
        
        if not os.path.exists(images_root):
            print(f"[ERROR] Dataset images not found: {images_root}")
            return
            
        # List all subdirectories
        sequences = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
        print(f"[Batch] Found {len(sequences)} sequences.")
        
        # Accumulate results for unified YAML
        unified_results = {}
        
        for seq in sequences:
            video_path = os.path.join(images_root, seq)
            mask_path = os.path.join(masks_root, seq)
            output_yaml = os.path.join(args.batch_output_dir, f"{seq}.yaml")
            
            if not os.path.exists(mask_path):
                print(f"[WARN] Mask not found for {seq}, skipping.")
                continue
                
            result = process_single_video(
                video_path, mask_path, output_yaml, 
                processor, model, model_name, args
            )
            if result is not None:
                unified_results[seq] = result
        
        # Write unified YAML (default: batch_output_dir/all_captions_BR.yaml)
        unified_path = args.unified_yaml or os.path.join(args.batch_output_dir, "all_captions_BR.yaml")
        write_unified_yaml(unified_path, unified_results)
            
    else:
        # Single video mode
        process_single_video(
            args.video_path, args.mask_path, args.output_yaml, 
            processor, model, model_name, args
        )
    
    print(f"\n[Done] All tasks completed!")


if __name__ == "__main__":
    main()
