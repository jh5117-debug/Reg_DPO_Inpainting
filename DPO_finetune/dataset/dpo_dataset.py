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
  - 每个视频的合法 start 先无放回穷尽一轮，再重新洗牌重复
"""

import json
import os
import random
import time
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
        self.base_seed = int(getattr(args, "seed", 0) or 0)
        self.current_epoch = 0
        self._cycle_order_cache = {}

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

        self.integrity_report_path = os.path.join(self.dpo_data_root, ".dpo_dataset_integrity_report.json")
        self.integrity_lock_path = self.integrity_report_path + ".lock"
        self.runtime_bad_videos = set()
        self.max_resample_attempts = 64
        self.integrity_report = self._load_or_create_integrity_report()
        self.bad_videos = set(self.integrity_report.get("bad_videos", {}).keys())
        self.good_videos = set(self.integrity_report.get("good_videos", {}).keys())
        self.entries = self._load_manifest()
        if not self.entries:
            raise RuntimeError(
                f"No valid DPO entries remain after integrity filtering under {self.dpo_data_root}. "
                "Please repair or replace the corrupted videos."
            )
        self._print_stats()

    def _is_logging_process(self) -> bool:
        return os.environ.get("RANK", "0") == "0"

    def _safe_remove(self, path: str):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def _is_stale_lock(self, path: str, stale_seconds: int = 7200) -> bool:
        try:
            return (time.time() - os.path.getmtime(path)) > stale_seconds
        except FileNotFoundError:
            return False

    def _load_or_create_integrity_report(self) -> dict:
        """
        训练开始前做一次数据完整性扫描：
        - 若存在损坏图片 / Git LFS pointer，则整段视频跳过
        - 多卡场景下只允许一个进程扫描，其余进程等待结果
        """
        while True:
            try:
                fd = os.open(self.integrity_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                os.close(fd)
                owner = True
                break
            except FileExistsError:
                if self._is_stale_lock(self.integrity_lock_path):
                    self._safe_remove(self.integrity_lock_path)
                    continue
                owner = False
                break

        if owner:
            try:
                self._safe_remove(self.integrity_report_path)
                if self._is_logging_process():
                    print(f"DPODataset integrity: scanning dataset at {self.dpo_data_root} ...")
                report = self._scan_dataset_integrity()
                tmp_path = self.integrity_report_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump(report, f, indent=2)
                os.replace(tmp_path, self.integrity_report_path)
                return report
            finally:
                self._safe_remove(self.integrity_lock_path)

        if self._is_logging_process():
            print("DPODataset integrity: waiting for another process to finish dataset scan ...")

        while True:
            if os.path.exists(self.integrity_report_path) and not os.path.exists(self.integrity_lock_path):
                with open(self.integrity_report_path) as f:
                    return json.load(f)
            if self._is_stale_lock(self.integrity_lock_path):
                self._safe_remove(self.integrity_lock_path)
                return self._load_or_create_integrity_report()
            time.sleep(2)

    def _inspect_image_file(self, path: str) -> tuple[str, Optional[str]]:
        try:
            with open(path, "rb") as f:
                head = f.read(256)
        except Exception as e:
            return "unreadable_file", f"{type(e).__name__}: {e}"

        if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
            return "lfs_pointer", "git-lfs-pointer"

        try:
            with Image.open(path) as img:
                img.verify()
            return "ok", None
        except Exception as e:
            return "invalid_image", f"{type(e).__name__}: {e}"

    def _scan_dataset_integrity(self) -> dict:
        image_exts = (".png", ".jpg", ".jpeg")
        relevant_dirs = ("gt_frames", "masks", "neg_frames_1", "neg_frames_2")

        summary = {
            "total_video_dirs_scanned": 0,
            "usable_video_dirs": 0,
            "skipped_video_dirs": 0,
            "total_image_files_scanned": 0,
            "clean_image_files": 0,
            "lfs_pointer_files": 0,
            "invalid_image_files": 0,
            "unreadable_file_count": 0,
            "trainable_image_files": 0,
        }
        good_videos = {}
        bad_videos = {}

        video_dirs = sorted(
            d for d in os.listdir(self.dpo_data_root)
            if os.path.isdir(os.path.join(self.dpo_data_root, d))
        )

        for idx, video_name in enumerate(video_dirs, 1):
            summary["total_video_dirs_scanned"] += 1
            video_root = os.path.join(self.dpo_data_root, video_name)
            video_total_images = 0
            video_bad_count = 0
            video_bad_files = []

            for subdir in relevant_dirs:
                subdir_path = os.path.join(video_root, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                image_files = sorted(
                    f for f in os.listdir(subdir_path)
                    if f.lower().endswith(image_exts)
                )

                for filename in image_files:
                    full_path = os.path.join(subdir_path, filename)
                    rel_path = os.path.join(video_name, subdir, filename)
                    video_total_images += 1
                    summary["total_image_files_scanned"] += 1

                    status, detail = self._inspect_image_file(full_path)
                    if status == "ok":
                        summary["clean_image_files"] += 1
                        continue

                    if status == "lfs_pointer":
                        summary["lfs_pointer_files"] += 1
                    elif status == "invalid_image":
                        summary["invalid_image_files"] += 1
                    else:
                        summary["unreadable_file_count"] += 1

                    video_bad_count += 1
                    if len(video_bad_files) < 20:
                        video_bad_files.append({
                            "path": rel_path,
                            "status": status,
                            "detail": detail,
                        })

            if video_bad_files:
                summary["skipped_video_dirs"] += 1
                bad_videos[video_name] = {
                    "total_image_files": video_total_images,
                    "bad_file_count": video_bad_count,
                    "example_bad_files": video_bad_files,
                }
            else:
                summary["usable_video_dirs"] += 1
                summary["trainable_image_files"] += video_total_images
                good_videos[video_name] = {
                    "total_image_files": video_total_images,
                }

            if self._is_logging_process() and (idx % 200 == 0 or idx == len(video_dirs)):
                print(
                    f"DPODataset integrity progress: {idx}/{len(video_dirs)} videos scanned, "
                    f"skipped={summary['skipped_video_dirs']}, "
                    f"bad_images={summary['lfs_pointer_files'] + summary['invalid_image_files'] + summary['unreadable_file_count']}"
                )

        summary["all_videos_clean"] = summary["skipped_video_dirs"] == 0

        return {
            "dataset_root": self.dpo_data_root,
            "summary": summary,
            "good_videos": good_videos,
            "bad_videos": bad_videos,
        }

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
        seen = set()  # 去重: (video_dir, neg_id)，避免 part1/part2 fallback 到同一目录后重复展开
        for video_name, info in manifest.items():
            # 优先用 key 作为目录名，若不存在则从 manifest 路径字段提取实际目录
            actual_dir_name = video_name
            video_dir = os.path.join(self.dpo_data_root, actual_dir_name)
            if not os.path.isdir(video_dir):
                # manifest 的 gt_frames 字段形如 "dpo_data/davis_bear/gt_frames"
                # 或直接 "davis_bear/gt_frames"，从中提取视频目录名
                gt_path_field = info.get("gt_frames", "")
                if gt_path_field:
                    # 取 gt_frames 路径的父目录名作为实际目录名
                    actual_dir_name = os.path.basename(os.path.dirname(gt_path_field))
                    video_dir = os.path.join(self.dpo_data_root, actual_dir_name)

            if actual_dir_name in self.bad_videos:
                continue

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
                "video_dir_name": actual_dir_name,
                "gt_dir": gt_dir,
                "mask_dir": mask_dir,
                "num_frames": num_frames,
                "chunks": chunks,
            }

            # 展开 neg_frames_1 和 neg_frames_2 为独立 entry
            # 用 (video_dir, neg_id) 去重，防止 manifest 中 part1/part2 fallback 到同一目录后重复
            for neg_dir, neg_id in [(neg_dir_1, "neg_1"), (neg_dir_2, "neg_2")]:
                dedup_key = (video_dir, neg_id)
                if os.path.isdir(neg_dir) and dedup_key not in seen:
                    seen.add(dedup_key)
                    entry = {**base, "neg_dir": neg_dir, "neg_id": neg_id}
                    valid_starts = self._enumerate_valid_starts(entry)
                    if not valid_starts:
                        continue
                    entry["valid_starts"] = tuple(valid_starts)
                    entry["cycle_key"] = f"{actual_dir_name}:{neg_id}"
                    entries.append(entry)

        # DAVIS 10x 过采样
        expanded_entries = []
        for entry in entries:
            repeats = self.davis_oversample if entry["video_name"].startswith("davis_") and self.davis_oversample > 1 else 1
            for cycle_slot in range(repeats):
                expanded_entry = dict(entry)
                expanded_entry["cycle_repeats"] = repeats
                expanded_entry["cycle_slot"] = cycle_slot
                expanded_entries.append(expanded_entry)

        return expanded_entries

    def _print_stats(self):
        if not self._is_logging_process():
            return

        integrity = self.integrity_report.get("summary", {})
        total_bad_images = (
            integrity.get("lfs_pointer_files", 0)
            + integrity.get("invalid_image_files", 0)
            + integrity.get("unreadable_file_count", 0)
        )
        print(
            "DPODataset integrity: "
            f"videos_scanned={integrity.get('total_video_dirs_scanned', 0)}, "
            f"usable_videos={integrity.get('usable_video_dirs', 0)}, "
            f"skipped_videos={integrity.get('skipped_video_dirs', 0)}, "
            f"images_scanned={integrity.get('total_image_files_scanned', 0)}, "
            f"trainable_images={integrity.get('trainable_image_files', 0)}, "
            f"bad_images={total_bad_images}, "
            f"all_videos_clean={integrity.get('all_videos_clean', False)}"
        )
        if self.integrity_report.get("bad_videos"):
            bad_examples = list(self.integrity_report["bad_videos"].keys())[:10]
            print(f"DPODataset integrity: skipped bad videos (first 10) = {bad_examples}")
        print(f"DPODataset integrity report saved to {self.integrity_report_path}")

        type_counts = {"davis": 0, "ytbv": 0}
        neg_counts = {"neg_1": 0, "neg_2": 0}
        start_counts = []
        for e in self.entries:
            if e["video_name"].startswith("davis_"):
                type_counts["davis"] += 1
            else:
                type_counts["ytbv"] += 1
            neg_counts[e.get("neg_id", "unknown")] = neg_counts.get(e.get("neg_id", "unknown"), 0) + 1
            start_counts.append(len(e.get("valid_starts", ())))
        stats = f"davis={type_counts['davis']}, ytbv={type_counts['ytbv']}, " \
                f"neg_1={neg_counts.get('neg_1', 0)}, neg_2={neg_counts.get('neg_2', 0)}"
        print(f"DPODataset: {len(self.entries)} entries from {self.dpo_data_root} ({stats})")
        if start_counts:
            avg_starts = sum(start_counts) / len(start_counts)
            print(
                "DPODataset starts: "
                f"avg_valid_starts={avg_starts:.1f}, "
                f"min_valid_starts={min(start_counts)}, "
                f"max_valid_starts={max(start_counts)}"
            )

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

    def _enumerate_valid_starts(self, entry):
        """列举当前 entry 的所有合法 start，供无放回轮转采样。"""
        chunks = entry.get("chunks")
        if self.chunk_aligned and chunks:
            valid_starts = []
            for chunk in chunks:
                c_start = int(chunk.get("start", 0))
                c_end = int(chunk.get("end", 0))
                chunk_len = c_end - c_start
                if chunk_len >= self.nframes:
                    valid_starts.extend(range(c_start, c_end - self.nframes + 1))
            if valid_starts:
                return sorted(set(valid_starts))

        max_start = entry["num_frames"] - self.nframes
        return list(range(0, max(0, max_start) + 1))

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _get_cycle_order(self, entry, cycle_id: int):
        cache_key = (entry["cycle_key"], cycle_id)
        if cache_key not in self._cycle_order_cache:
            start_order = list(entry["valid_starts"])
            rng = random.Random(f"{self.base_seed}:{entry['cycle_key']}:{cycle_id}")
            rng.shuffle(start_order)
            self._cycle_order_cache[cache_key] = tuple(start_order)
        return self._cycle_order_cache[cache_key]

    def _next_start(self, entry):
        total_starts = len(entry["valid_starts"])
        global_visit_index = self.current_epoch * entry.get("cycle_repeats", 1) + entry.get("cycle_slot", 0)
        cycle_id = global_visit_index // total_starts
        cycle_pos = global_visit_index % total_starts
        start_order = self._get_cycle_order(entry, cycle_id)
        return start_order[cycle_pos]

    def tokenize_captions(self, caption):
        if random.random() < self.args.proportion_empty_prompts:
            caption = ""
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return inputs.input_ids

    def _sample_replacement_index(self):
        for _ in range(128):
            candidate = random.randrange(len(self.entries))
            if self.entries[candidate]["video_dir_name"] not in self.runtime_bad_videos:
                return candidate
        for idx, entry in enumerate(self.entries):
            if entry["video_dir_name"] not in self.runtime_bad_videos:
                return idx
        return None

    def _load_entry(self, entry):
        if entry["video_dir_name"] in self.runtime_bad_videos:
            raise RuntimeError(f"Video already marked bad at runtime: {entry['video_dir_name']}")

        # 每个视频的合法 start 先无放回穷尽一轮，再重新洗牌重复
        start = self._next_start(entry)
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

    def __getitem__(self, index):
        candidate_index = index
        last_error = None

        for _ in range(self.max_resample_attempts):
            entry = self.entries[candidate_index]
            video_dir_name = entry["video_dir_name"]

            if video_dir_name in self.runtime_bad_videos:
                next_index = self._sample_replacement_index()
                if next_index is None:
                    break
                candidate_index = next_index
                continue

            try:
                return self._load_entry(entry)
            except (FileNotFoundError, OSError, IndexError, ValueError) as e:
                self.runtime_bad_videos.add(video_dir_name)
                last_error = e
                if self._is_logging_process():
                    print(
                        f"DPODataset runtime warning: skipping video {video_dir_name} due to data error: "
                        f"{type(e).__name__}: {e}"
                    )
                next_index = self._sample_replacement_index()
                if next_index is None:
                    break
                candidate_index = next_index

        raise RuntimeError(
            "DPODataset could not find a valid replacement sample after skipping bad videos. "
            f"Runtime-skipped videos={len(self.runtime_bad_videos)}. Last error: {last_error}"
        )
