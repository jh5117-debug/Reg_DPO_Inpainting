"""
DPO 偏好对数据集 — 用于 DiffuEraser DPO Finetune

从预生成的 DPO_Finetune_data 目录读取正样本 (GT) 和双负样本 (neg_frames_1/neg_frames_2)。
每个视频的 neg_frames_1/neg_frames_2 展开为独立 entry，全局 shuffle 确保覆盖。

目录结构:
  DPO_Finetune_data/
  ├── manifest.json
  ├── {video_name}/
  │   ├── gt_frames/        ← GT 帧
  │   ├── masks/            ← mask 序列
  │   ├── neg_frames_1/     ← 纵向缝合最差负样本
  │   ├── neg_frames_2/     ← 纵向缝合第二差负样本
  │   └── meta.json         ← chunk 边界信息
  └── ...

关键设计:
  - BrushNet 条件统一使用 GT masked image（防信息泄漏）
  - DAVIS 视频 10x 过采样以平衡数据量
  - 支持读取 meta.json chunk 边界进行对齐采样（Stage 2 可选）
"""

import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class DPODataset(torch.utils.data.Dataset):
    """
    DPO 偏好对数据集。

    每个 sample 返回一组 (正样本, 负样本) pair + 统一的 GT masked image 条件。
    正样本: GT 帧
    负样本: neg_frames_1 或 neg_frames_2 (展开为独立 entry)
    """

    def __init__(self, args, tokenizer, dpo_data_root: Optional[str] = None):
        self.args = args
        self.nframes = args.nframes
        self.size = args.resolution
        self.tokenizer = tokenizer
        self.dpo_data_root = dpo_data_root or getattr(args, "dpo_data_root", "data/DPO_Finetune_data")
        self.davis_oversample = getattr(args, "davis_oversample", 10)
        self.chunk_aligned = getattr(args, "chunk_aligned", False)

        self.img_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])

        self.entries = self._load_manifest()
        self._print_stats()

    def _load_manifest(self) -> list[dict]:
        manifest_path = os.path.join(self.dpo_data_root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"manifest.json not found at {manifest_path}. "
                "Run generate_dpo_negatives.py first."
            )
        with open(manifest_path) as f:
            manifest = json.load(f)

        entries = []
        for video_name, info in manifest.items():
            # 优先用 key 作为目录名，若不存在则从 manifest 路径字段提取实际目录
            video_dir = os.path.join(self.dpo_data_root, video_name)
            if not os.path.isdir(video_dir):
                # manifest 的 gt_frames 字段形如 "dpo_data/davis_bear/gt_frames"
                # 或直接 "davis_bear/gt_frames"，从中提取视频目录名
                gt_path_field = info.get("gt_frames", "")
                if gt_path_field:
                    # 取 gt_frames 路径的父目录名作为实际目录名
                    actual_dir_name = os.path.basename(os.path.dirname(gt_path_field))
                    video_dir = os.path.join(self.dpo_data_root, actual_dir_name)

            gt_dir = os.path.join(video_dir, "gt_frames")
            mask_dir = os.path.join(video_dir, "masks")
            neg_dir_1 = os.path.join(video_dir, "neg_frames_1")
            neg_dir_2 = os.path.join(video_dir, "neg_frames_2")

            if not os.path.isdir(gt_dir) or not os.path.isdir(mask_dir):
                continue

            num_frames = info.get("num_frames", len(os.listdir(gt_dir)))
            if num_frames < self.nframes:
                continue

            # 读取 chunk 边界信息
            meta_path = os.path.join(video_dir, "meta.json")
            chunks = None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    chunks = meta.get("chunks", None)
                except Exception:
                    pass

            base = {
                "video_name": video_name,
                "gt_dir": gt_dir,
                "mask_dir": mask_dir,
                "num_frames": num_frames,
                "chunks": chunks,
            }

            # 展开 neg_frames_1 和 neg_frames_2 为独立 entry
            for neg_dir, neg_id in [(neg_dir_1, "neg_1"), (neg_dir_2, "neg_2")]:
                if os.path.isdir(neg_dir):
                    entries.append({**base, "neg_dir": neg_dir, "neg_id": neg_id})

        # DAVIS 10x 过采样
        if self.davis_oversample > 1:
            davis_entries = [e for e in entries if e["video_name"].startswith("davis_")]
            ytbv_entries = [e for e in entries if not e["video_name"].startswith("davis_")]
            entries = davis_entries * self.davis_oversample + ytbv_entries

        return entries

    def _print_stats(self):
        type_counts = {"davis": 0, "ytbv": 0}
        neg_counts = {"neg_1": 0, "neg_2": 0}
        for e in self.entries:
            if e["video_name"].startswith("davis_"):
                type_counts["davis"] += 1
            else:
                type_counts["ytbv"] += 1
            neg_counts[e.get("neg_id", "unknown")] = neg_counts.get(e.get("neg_id", "unknown"), 0) + 1
        stats = f"davis={type_counts['davis']}, ytbv={type_counts['ytbv']}, " \
                f"neg_1={neg_counts.get('neg_1', 0)}, neg_2={neg_counts.get('neg_2', 0)}"
        print(f"DPODataset: {len(self.entries)} entries from {self.dpo_data_root} ({stats})")

    def __len__(self):
        return len(self.entries)

    def _read_frames(self, frame_dir, frame_indices):
        all_files = sorted(f for f in os.listdir(frame_dir)
                           if f.endswith(('.jpg', '.png', '.jpeg')))
        return [Image.open(os.path.join(frame_dir, all_files[i])).convert("RGB")
                for i in frame_indices]

    def _read_masks(self, mask_dir, frame_indices):
        all_files = sorted(f for f in os.listdir(mask_dir)
                           if f.endswith(('.png', '.jpg')))
        return [Image.open(os.path.join(mask_dir, all_files[i])).convert("L")
                for i in frame_indices]

    def _get_chunk_aligned_start(self, entry):
        """根据 meta.json chunk 边界选择采样起始点，避免跨缝合线。"""
        chunks = entry.get("chunks")
        if not chunks:
            # fallback: 随机采样
            max_start = entry["num_frames"] - self.nframes
            return random.randint(0, max(0, max_start))

        # 从所有 chunk 中随机选择一个能容纳 nframes 的 chunk
        valid_starts = []
        for chunk in chunks:
            c_start = chunk.get("start", 0)
            c_end = chunk.get("end", 0)
            chunk_len = c_end - c_start
            if chunk_len >= self.nframes:
                # 可以在这个 chunk 内随机选起点
                for s in range(c_start, c_end - self.nframes + 1):
                    valid_starts.append(s)

        if valid_starts:
            return random.choice(valid_starts)
        else:
            # fallback
            max_start = entry["num_frames"] - self.nframes
            return random.randint(0, max(0, max_start))

    def tokenize_captions(self, caption):
        if random.random() < self.args.proportion_empty_prompts:
            caption = ""
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return inputs.input_ids

    def __getitem__(self, index):
        entry = self.entries[index]

        # 选择连续 nframes 帧的起始位置
        if self.chunk_aligned:
            start = self._get_chunk_aligned_start(entry)
        else:
            max_start = entry["num_frames"] - self.nframes
            start = random.randint(0, max(0, max_start))

        frame_indices = list(range(start, start + self.nframes))

        gt_frames = self._read_frames(entry["gt_dir"], frame_indices)
        masks_pil = self._read_masks(entry["mask_dir"], frame_indices)
        neg_frames = self._read_frames(entry["neg_dir"], frame_indices)

        pos_tensors, neg_tensors = [], []
        cond_tensors = []  # 统一使用 GT masked image
        mask_tensors = []

        state = torch.get_rng_state()

        for i in range(self.nframes):
            mask_np = np.array(masks_pil[i])[:, :, np.newaxis].astype(np.float32) / 255.0
            # BrushNet 条件：统一使用 GT masked image（关键！防信息泄漏）
            gt_masked = Image.fromarray(
                (np.array(gt_frames[i]) * (1.0 - mask_np)).astype(np.uint8))
            mask_inv = Image.fromarray(255 - np.array(masks_pil[i]))

            torch.set_rng_state(state)
            pos_tensors.append(self.img_transform(gt_frames[i]))
            torch.set_rng_state(state)
            neg_tensors.append(self.img_transform(neg_frames[i]))
            torch.set_rng_state(state)
            cond_tensors.append(self.img_transform(gt_masked))
            torch.set_rng_state(state)
            mask_tensors.append(self.mask_transform(mask_inv))

        # 50% 时序翻转
        if random.random() < 0.5:
            pos_tensors.reverse()
            neg_tensors.reverse()
            cond_tensors.reverse()
            mask_tensors.reverse()

        return {
            "pixel_values_pos": torch.stack(pos_tensors),          # [nframes, 3, H, W]
            "pixel_values_neg": torch.stack(neg_tensors),          # [nframes, 3, H, W]
            "conditioning_pixel_values": torch.stack(cond_tensors),  # [nframes, 3, H, W]
            "masks": torch.stack(mask_tensors),                    # [nframes, 1, H, W]
            "input_ids": self.tokenize_captions("clean background")[0],
        }
