#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM Captioning Script for OR (Object Removal) Scene
=====================================================

为 OR 场景生成双阶段 caption：
  - prompt:   描述 mask 以外的背景（引导模型生成正确的背景内容）
  - n_prompt: 描述 mask 以内的物体 + 通用质量负面词（抑制模型按 mask 形状生成物体）

这是解决生成式 Video Inpainting 痛点的关键：
  模型容易根据 mask 的形状"想象"出物体，而不是填充背景。
  通过 n_prompt 明确告诉模型"不要生成什么"，从而引导正确的背景填充。

输出到独立的 all_captions_OR.yaml（与 BR 的 all_captions.yaml 分离）。

运行环境要求:
    conda activate caption_env
    pip install transformers>=4.45 torch torchvision pillow pyyaml opencv-python-headless

使用示例:
    # 单视频模式
    python generate_captions_OR.py \
        --video_path /path/to/DAVIS/JPEGImages/480p/bear \
        --mask_path /path/to/DAVIS/Annotations/480p/bear \
        --output_yaml prompt_cache/bear_OR.yaml \
        --model_path /path/to/Qwen2.5-VL-7B-Instruct

    # 批量处理 DAVIS 数据集
    CUDA_VISIBLE_DEVICES=0 python generate_captions_OR.py \
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


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION A — CLI Arguments
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scene captions for OR (Object Removal) inpainting using VLM.\n"
                    "Produces background-only prompts (prompt) and object descriptions (n_prompt)."
    )
    parser.add_argument("--video_path", type=str,
                        help="Path to video frames directory or video file (single-video mode)")
    parser.add_argument("--mask_path", type=str,
                        help="Path to mask directory, video, or single image (single-video mode)")
    parser.add_argument("--output_yaml", type=str,
                        help="Output YAML file path (single-video mode)")
    parser.add_argument("--dataset_root", type=str,
                        help="Root directory of DAVIS dataset (e.g. .../DAVIS). Overrides video_path.")
    parser.add_argument("--batch_output_dir", type=str, default="prompt_cache",
                        help="Output directory for batch processing")
    parser.add_argument("--unified_yaml", type=str, default=None,
                        help="Path for a single unified YAML containing all OR video captions (batch mode). "
                             "Default: batch_output_dir/all_captions_OR.yaml")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to VLM model (local or HuggingFace)")
    parser.add_argument("--frame_strategy", type=str, default="middle",
                        choices=["middle", "multi_sample"],
                        help="Frame sampling strategy: 'middle' or 'multi_sample'")
    parser.add_argument("--num_sample_frames", type=int, default=3,
                        help="Number of frames for 'multi_sample' strategy")
    parser.add_argument("--bbox_expand_ratio", type=float, default=0.2,
                        help="Expand ratio for mask bounding box crop (default 0.2 = 20%%)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite existing YAML")

    args = parser.parse_args()

    if not args.dataset_root and not args.video_path:
        parser.error("Either --video_path or --dataset_root must be provided.")

    return args


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION B — Frame / Mask I/O
# ═══════════════════════════════════════════════════════════════════════════

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


def read_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Read a specific frame from video file. Returns RGB numpy array."""
    if os.path.isdir(video_path):
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


def read_mask_frame(mask_path: str, frame_idx: int, target_size: tuple) -> np.ndarray:
    """Read mask frame. Returns grayscale numpy array (255=mask/hole)."""
    if os.path.isdir(mask_path):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        files = sorted([f for f in os.listdir(mask_path) if f.lower().endswith(exts)])
        if not files:
            raise ValueError(f"No mask images found in {mask_path}")
        frame_idx = min(frame_idx, len(files) - 1)
        mask_file = os.path.join(mask_path, files[frame_idx])
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    elif mask_path.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(mask_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read mask frame {frame_idx}")
        mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError(f"Cannot read mask from {mask_path}")

    if mask.shape[:2] != (target_size[1], target_size[0]):
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

    return mask


def select_frame_indices(total_frames: int, strategy: str, num_samples: int = 3) -> list:
    """Select frame indices based on strategy."""
    if strategy == "middle":
        return [total_frames // 2]
    else:  # multi_sample
        if total_frames <= num_samples:
            return list(range(total_frames))
        step = total_frames // (num_samples + 1)
        return [step * (i + 1) for i in range(num_samples)]


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION C — Image Preprocessing for VLM
# ═══════════════════════════════════════════════════════════════════════════

def apply_gray_mask(frame_rgb: np.ndarray, mask_gray: np.ndarray,
                    fill_color: tuple = (128, 128, 128)) -> Image.Image:
    """Apply gray fill to masked region (for background-only captioning).

    Args:
        frame_rgb: RGB frame (H, W, 3)
        mask_gray: Grayscale mask (H, W), 255=hole/object
        fill_color: RGB color for fill (default neutral gray)

    Returns:
        PIL Image with mask region filled with gray
    """
    # DAVIS masks use palette index values (0, 1, 2, 38...), not 0/255
    # Any non-zero pixel = object/mask area
    mask_binary = (mask_gray > 0).astype(np.float32)
    mask_3ch = np.stack([mask_binary] * 3, axis=-1)

    fill = np.array(fill_color, dtype=np.float32).reshape(1, 1, 3)
    fill_img = np.ones_like(frame_rgb, dtype=np.float32) * fill

    result = frame_rgb.astype(np.float32) * (1 - mask_3ch) + fill_img * mask_3ch
    return Image.fromarray(result.astype(np.uint8))


def crop_mask_region(frame_rgb: np.ndarray, mask_gray: np.ndarray,
                     expand_ratio: float = 0.2) -> Image.Image:
    """Crop the bounding box region of the mask from the original frame.

    Used for object identification — VLM sees the original object in context.

    Args:
        frame_rgb: RGB frame (H, W, 3)
        mask_gray: Grayscale mask (H, W), 255=object area
        expand_ratio: Expand the bounding box by this ratio (default 20%)

    Returns:
        PIL Image of cropped region, or None if mask is empty
    """
    # DAVIS masks: any non-zero pixel = object area
    binary = (mask_gray > 0).astype(np.uint8)

    # 检查 mask 是否真的有非零像素
    if binary.sum() == 0:
        return None

    # 找到 mask 的 bounding box
    ys, xs = np.where(binary > 0)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # 扩展 bounding box
    h, w = frame_rgb.shape[:2]
    box_h = y_max - y_min
    box_w = x_max - x_min
    expand_h = int(box_h * expand_ratio)
    expand_w = int(box_w * expand_ratio)

    y_min = max(0, y_min - expand_h)
    y_max = min(h - 1, y_max + expand_h)
    x_min = max(0, x_min - expand_w)
    x_max = min(w - 1, x_max + expand_w)

    cropped = frame_rgb[y_min:y_max + 1, x_min:x_max + 1]

    # 确保裁剪区域足够大
    if cropped.shape[0] < 16 or cropped.shape[1] < 16:
        return None

    return Image.fromarray(cropped)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION D — VLM Model
# ═══════════════════════════════════════════════════════════════════════════

def load_vlm_model(model_path: str, device: str):
    """Load Qwen2.5-VL model. Returns (processor, model) tuple."""
    print(f"[VLM] Loading model from {model_path}...")

    try:
        from transformers import AutoProcessor
        import torch
    except ImportError as e:
        print(f"[ERROR] Required packages not installed: {e}")
        print("Please run: pip install transformers>=4.45 torch")
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
            print("[ERROR] Neither Qwen2_5_VL nor Qwen2VL available.")
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


def _vlm_inference(processor, model, image: Image.Image, system_prompt: str,
                   user_prompt: str, device: str, max_words: int = 30) -> str:
    """Run VLM inference with given prompts. Returns cleaned caption string."""
    import torch

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

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    caption = output_text.strip().strip('"').strip("'")
    words = caption.split()
    if len(words) > max_words + 5:
        caption = " ".join(words[:max_words])

    return caption


def generate_background_caption(processor, model, image: Image.Image, device: str) -> str:
    """生成背景描述（灰色区域以外的场景）。

    System prompt 引导 VLM 忽略灰色遮盖区域，只描述可见的背景。
    """
    system_prompt = """You are a background scene description assistant.
The image has a gray area covering an object that has been removed.
Your job is to describe ONLY the visible background scene OUTSIDE the gray area.

Rules:
1. DO NOT mention the gray area, the removed object, or any occluded content.
2. Describe the environment, lighting, textures, colors, and spatial layout of the VISIBLE background.
3. Include ground, sky, buildings, vegetation, road, water, furniture, and other background elements.
4. Output a single concise English sentence, maximum 30 words.
5. Start directly with the scene description, no preamble.

Example outputs:
- "a sunlit park path with green grass, tall trees, and a wooden bench under a clear blue sky"
- "an urban intersection with asphalt road, white crosswalk markings, traffic lights, and distant buildings"
- "a rocky shoreline with calm blue water, sparse vegetation, and white houses on distant hills"
"""

    user_prompt = "Describe only the visible background scene in this image, ignoring the gray masked area."
    return _vlm_inference(processor, model, image, system_prompt, user_prompt, device, max_words=30)


def generate_object_caption(processor, model, image: Image.Image, device: str) -> str:
    """生成 mask 内物体描述（用于 negative prompt）。

    对裁剪出的 mask 区域，让 VLM 识别物体/实体。
    """
    system_prompt = """You are an object identification assistant.
Identify the main object or entity visible in this cropped image region.

Rules:
1. Give a brief, specific description of the object (what it is, its color, type).
2. Output ONLY the object description as a short noun phrase, maximum 10 words.
3. Do NOT describe the background or surroundings.
4. Be specific: say "brown bear" not just "animal", say "red sports car" not just "vehicle".

Example outputs:
- "brown bear"
- "person in blue jacket riding bicycle"
- "white fishing boat"
- "black swan with red beak"
- "silver sedan car"
"""

    user_prompt = "What is the main object in this image? Give a brief noun phrase description."
    return _vlm_inference(processor, model, image, system_prompt, user_prompt, device, max_words=10)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION E — YAML Output
# ═══════════════════════════════════════════════════════════════════════════

QUALITY_NEGATIVE = "blurry, flickering, distorted, artifacts, text, watermark, low quality"

# 抑制模型根据 mask 形状生成物体（而非背景）的关键负面词
MASK_SHAPE_NEGATIVE = "mask-shaped object, object matching mask shape, foreground object, new object in masked area, silhouette, shape-filling object"


def build_or_n_prompt(object_desc: str) -> str:
    """Combine object description + mask-shape suppression + quality negative words for n_prompt."""
    parts = []
    if object_desc and object_desc.strip():
        parts.append(object_desc.strip())
    parts.append(MASK_SHAPE_NEGATIVE)
    parts.append(QUALITY_NEGATIVE)
    return ", ".join(parts)


def write_yaml(output_path: str, bg_prompt: str, n_prompt: str,
               object_desc: str, model_name: str, frame_idx: int, video_path: str):
    """Write OR prompt configuration to per-video YAML."""
    config = {
        "prompt": [bg_prompt],
        "n_prompt": [n_prompt],
        "object_description": object_desc,
        "text_guidance_scale": 2.0,
        "task_type": "OR",
        "prompt_source": "auto_masked_OR",
        "prompt_model": model_name,
        "prompt_timestamp": datetime.now().isoformat(),
        "prompt_frame_idx": frame_idx,
        "source_video": video_path,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by generate_captions_OR.py (Object Removal)\n")
        f.write(f"# Video: {video_path}\n\n")
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"[YAML] Written to: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION F — Single Video Processing
# ═══════════════════════════════════════════════════════════════════════════

def process_single_video(
    video_path: str, mask_path: str, output_yaml: str,
    processor, model, model_name: str, args
):
    """Process a single video sequence for OR captioning.

    双阶段推理:
      1. 背景描述: 灰色遮盖 mask 区域 -> VLM 描述背景
      2. 物体识别: 裁剪 mask 区域 -> VLM 识别物体

    Returns:
        dict with caption info (for unified YAML), or None if skipped/failed.
    """
    # Check if output exists
    if output_yaml and os.path.exists(output_yaml) and not args.force:
        try:
            with open(output_yaml, "r") as f:
                existing = yaml.safe_load(f) or {}
            if existing.get("prompt") and existing["prompt"][0]:
                print(f"[SKIP] YAML already exists: {output_yaml}")
                return existing
        except Exception:
            pass

    video_name = Path(video_path).name
    print(f"[{video_name}] Processing (OR mode)...")

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
    bg_captions = []
    obj_captions = []
    selected_frame_idx = frame_indices[0]

    try:
        for idx in frame_indices:
            frame_rgb = read_video_frame(video_path, idx)
            h, w = frame_rgb.shape[:2]

            # Read corresponding mask
            mask_gray = read_mask_frame(mask_path, idx, (w, h))

            # ── Stage 1: Background caption (gray-masked image) ──
            masked_image = apply_gray_mask(frame_rgb, mask_gray)
            bg_caption = generate_background_caption(processor, model, masked_image, args.device)
            bg_captions.append(bg_caption)
            print(f"  Frame {idx} [BG]:  {bg_caption}")

            # ── Stage 2: Object caption (cropped mask region) ──
            cropped = crop_mask_region(frame_rgb, mask_gray, expand_ratio=args.bbox_expand_ratio)
            if cropped is not None:
                obj_caption = generate_object_caption(processor, model, cropped, args.device)
                obj_captions.append(obj_caption)
                print(f"  Frame {idx} [OBJ]: {obj_caption}")
            else:
                print(f"  Frame {idx} [OBJ]: (mask too small or empty, skipped)")

        # Select best captions
        if len(bg_captions) > 1:
            final_bg = max(bg_captions, key=len)
            print(f"  Selected BG: {final_bg}")
        else:
            final_bg = bg_captions[0] if bg_captions else ""

        if obj_captions:
            # 去重并取最长的物体描述
            unique_objs = list(set(obj_captions))
            final_obj = max(unique_objs, key=len) if unique_objs else ""
            print(f"  Selected OBJ: {final_obj}")
        else:
            final_obj = ""

        # Build n_prompt
        n_prompt = build_or_n_prompt(final_obj)

        # Write per-video YAML (only in single-video mode, not batch)
        if output_yaml:
            write_yaml(
                output_yaml, final_bg, n_prompt, final_obj,
                model_name, selected_frame_idx, video_path,
            )

        # Return data for unified YAML
        return {
            "prompt": [final_bg],
            "n_prompt": [n_prompt],
            "object_description": final_obj,
            "text_guidance_scale": 2.0,
            "task_type": "OR",
            "prompt_source": "auto_masked_OR",
            "prompt_model": model_name,
            "prompt_timestamp": datetime.now().isoformat(),
            "prompt_frame_idx": selected_frame_idx,
            "source_video": video_path,
        }
    except Exception as e:
        print(f"[ERROR] Failed processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION G — Unified OR YAML (独立文件，不与 BR 混合)
# ═══════════════════════════════════════════════════════════════════════════

def write_unified_yaml(unified_path: str, or_results: dict):
    """Write all OR caption results to a unified flat YAML file.

    Format (same flat structure as BR's all_captions.yaml):
        video_name:
          prompt: [...]
          n_prompt: [...]
          object_description: "..."
          text_guidance_scale: 2.0
          task_type: OR
          ...
    """
    Path(unified_path).parent.mkdir(parents=True, exist_ok=True)
    with open(unified_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by generate_captions_OR.py (Object Removal)\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n")
        f.write(f"# Total videos: {len(or_results)}\n\n")
        yaml.dump(or_results, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f"\n[Unified OR YAML] Written {len(or_results)} entries → {unified_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION H — Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Load VLM model once
    processor, model = load_vlm_model(args.model_path, args.device)
    model_name = Path(args.model_path).name

    if args.dataset_root:
        # Batch processing mode
        print(f"[Batch] Processing dataset (OR mode): {args.dataset_root}")
        images_root = os.path.join(args.dataset_root, "JPEGImages", "480p")
        masks_root = os.path.join(args.dataset_root, "Annotations", "480p")

        if not os.path.exists(images_root):
            print(f"[ERROR] Dataset images not found: {images_root}")
            return

        sequences = sorted([d for d in os.listdir(images_root)
                            if os.path.isdir(os.path.join(images_root, d))])
        print(f"[Batch] Found {len(sequences)} sequences.")

        # Accumulate OR results
        or_results = {}

        for seq_i, seq in enumerate(sequences, 1):
            print(f"\n[{seq_i}/{len(sequences)}] {seq}")
            video_path = os.path.join(images_root, seq)
            mask_path = os.path.join(masks_root, seq)

            if not os.path.exists(mask_path):
                print(f"[WARN] Mask not found for {seq}, skipping.")
                continue

            result = process_single_video(
                video_path, mask_path, None,
                processor, model, model_name, args
            )
            if result is not None:
                or_results[seq] = result

        # Write unified OR YAML (独立文件，不与 BR 的 all_captions.yaml 混合)
        unified_path = args.unified_yaml or os.path.join(args.batch_output_dir, "all_captions_OR.yaml")
        write_unified_yaml(unified_path, or_results)

    else:
        # Single video mode
        process_single_video(
            args.video_path, args.mask_path, args.output_yaml,
            processor, model, model_name, args
        )

    print(f"\n[Done] All tasks completed!")


if __name__ == "__main__":
    main()
