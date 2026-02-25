"""
DPO 偏好对数据集

用于 Region-Reg-DPO Stage 3 训练。从预生成的 dpo_data 目录读取
正样本 (GT) 和负样本 (三路对决中最差的一路)。

目录结构:
  dpo_data/
  ├── {video_name}/
  │   ├── gt_frames/      ← 正样本 (GT)
  │   ├── masks/          ← mask 序列
  │   ├── neg_frames/     ← 负样本 (blur/hallucination/flicker 中最差)
  │   └── meta.json       ← 元信息
  └── manifest.json
"""

import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from dataset.region_mask_utils import decompose_mask_regions


class DPODataset(torch.utils.data.Dataset):
    """
    Region-Reg-DPO 训练数据集。

    每个 sample 返回一组正负样本对 + 三区分解 mask。
    正样本: GT 帧
    负样本: 三路对决中评分最差的一路
    """

    def __init__(self, args, tokenizer, dpo_data_root: Optional[str] = None):
        self.args = args
        self.nframes = args.nframes
        self.size = args.resolution
        self.tokenizer = tokenizer
        self.dpo_data_root = dpo_data_root or getattr(args, "dpo_data_root", "dpo_data")
        self.boundary_dilation = getattr(args, "boundary_dilation", 7)

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
            gt_dir = info.get("gt_frames", "")
            mask_dir = info.get("masks", "")
            neg_dir = info.get("neg_frames", "")

            if not all(os.path.isdir(d) for d in [gt_dir, mask_dir, neg_dir]):
                continue

            num_frames = info.get("num_frames", len(os.listdir(gt_dir)))
            if num_frames < self.nframes:
                continue

            entries.append({
                "video_name": video_name,
                "gt_dir": gt_dir,
                "mask_dir": mask_dir,
                "neg_dir": neg_dir,
                "neg_type": info.get("neg_type", "unknown"),
                "num_frames": num_frames,
            })
        return entries

    def _print_stats(self):
        type_counts = {}
        for e in self.entries:
            nt = e.get("neg_type", "unknown")
            type_counts[nt] = type_counts.get(nt, 0) + 1
        stats = ", ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
        print(f"DPODataset: {len(self.entries)} videos from {self.dpo_data_root} ({stats})")

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

        # 随机选连续 nframes 帧
        max_start = entry["num_frames"] - self.nframes
        start = random.randint(0, max_start) if max_start > 0 else 0
        frame_indices = list(range(start, start + self.nframes))

        gt_frames = self._read_frames(entry["gt_dir"], frame_indices)
        masks_pil = self._read_masks(entry["mask_dir"], frame_indices)
        neg_frames = self._read_frames(entry["neg_dir"], frame_indices)

        pos_tensors, neg_tensors = [], []
        cond_pos_tensors, cond_neg_tensors = [], []
        mask_tensors = []

        state = torch.get_rng_state()

        for i in range(self.nframes):
            mask_np = np.array(masks_pil[i])[:, :, np.newaxis].astype(np.float32) / 255.0
            pos_masked = Image.fromarray(
                (np.array(gt_frames[i]) * (1.0 - mask_np)).astype(np.uint8))
            neg_masked = Image.fromarray(
                (np.array(neg_frames[i]) * (1.0 - mask_np)).astype(np.uint8))
            mask_inv = Image.fromarray(255 - np.array(masks_pil[i]))

            torch.set_rng_state(state)
            pos_tensors.append(self.img_transform(gt_frames[i]))
            torch.set_rng_state(state)
            neg_tensors.append(self.img_transform(neg_frames[i]))
            torch.set_rng_state(state)
            cond_pos_tensors.append(self.img_transform(pos_masked))
            torch.set_rng_state(state)
            cond_neg_tensors.append(self.img_transform(neg_masked))
            torch.set_rng_state(state)
            mask_tensors.append(self.mask_transform(mask_inv))

        # 50% 时序翻转
        if random.random() < 0.5:
            pos_tensors.reverse()
            neg_tensors.reverse()
            cond_pos_tensors.reverse()
            cond_neg_tensors.reverse()
            mask_tensors.reverse()

        masks = torch.stack(mask_tensors)
        hole_mask = 1.0 - masks
        M_h_list, M_b_list, M_c_list = [], [], []
        for f_idx in range(self.nframes):
            m_h, m_b, m_c = decompose_mask_regions(
                hole_mask[f_idx].unsqueeze(0),
                dilation_kernel=self.boundary_dilation)
            M_h_list.append(m_h.squeeze(0))
            M_b_list.append(m_b.squeeze(0))
            M_c_list.append(m_c.squeeze(0))

        return {
            "pixel_values_pos": torch.stack(pos_tensors),
            "pixel_values_neg": torch.stack(neg_tensors),
            "conditioning_pos": torch.stack(cond_pos_tensors),
            "conditioning_neg": torch.stack(cond_neg_tensors),
            "masks": masks,
            "mask_hole": torch.stack(M_h_list),
            "mask_boundary": torch.stack(M_b_list),
            "mask_context": torch.stack(M_c_list),
            "input_ids": self.tokenize_captions("clean background")[0],
            "neg_type": entry.get("neg_type", "unknown"),
        }
