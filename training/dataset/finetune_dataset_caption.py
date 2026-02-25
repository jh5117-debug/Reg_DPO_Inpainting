import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from dataset.utils import create_random_shape_with_random_motion


class FinetuneDatasetWithCaption(torch.utils.data.Dataset):
    """
    Dataset for finetuning DiffuEraser on DAVIS-2017 and YouTubeVOS-2019,
    with real caption/prompt support loaded from YAML files.

    Key difference from FinetuneDataset:
    - Loads captions from a YAML file (video_name -> caption mapping)
    - Falls back to "clean background" if no caption found for a video

    Data processing pipeline (identical to FinetuneDataset):
    1. Read consecutive frames from video directory
    2. Generate random brush-stroke masks with motion
    3. Create masked images at original resolution
    4. Resize (short-edge) + center crop to target resolution
    5. Normalize to [-1, 1] for images, [0, 1] for masks
    6. 50% probability temporal flip
    """

    DEFAULT_CAPTION = "clean background"

    def __init__(self, args, tokenizer):
        self.args = args
        self.nframes = args.nframes
        self.size = args.resolution
        self.tokenizer = tokenizer

        # Transform matching original TrainDataset
        self.img_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])  # output: [-1, 1]

        self.mask_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
        ])  # output: [0, 1]

        # Load caption YAML
        self.captions = {}
        if hasattr(args, 'caption_yaml') and args.caption_yaml:
            self._load_captions(args.caption_yaml)

        # Scan datasets: store (jpeg_dir, frame_list, video_name) tuples
        self.video_list = []
        self._scan_davis(args.davis_root)
        self._scan_ytvos(args.ytvos_root)
        print(f"FinetuneDatasetWithCaption: total {len(self.video_list)} videos, "
              f"{len(self.captions)} captions loaded")

    def _load_captions(self, caption_yaml_path):
        """Load caption YAML file.
        
        Expected format (merged YAML):
            davis_bear:
                prompt: ["a brown bear walking through shallow water..."]
                ...
            ytvos_2d9a1a1d49:
                prompt: ["a person riding a bicycle on a sunny road..."]
                ...
        """
        if not os.path.exists(caption_yaml_path):
            print(f"Warning: caption YAML not found: {caption_yaml_path}")
            return

        try:
            with open(caption_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            for key, val in data.items():
                if isinstance(val, dict) and 'prompt' in val:
                    prompt_list = val['prompt']
                    if isinstance(prompt_list, list) and len(prompt_list) > 0:
                        self.captions[key] = prompt_list[0]
                    elif isinstance(prompt_list, str):
                        self.captions[key] = prompt_list
                elif isinstance(val, str):
                    self.captions[key] = val

            print(f"  Loaded {len(self.captions)} captions from {caption_yaml_path}")
        except Exception as e:
            print(f"Warning: failed to load captions from {caption_yaml_path}: {e}")

    def _scan_davis(self, davis_root):
        """Scan DAVIS dataset using train.txt, 10x oversampling."""
        if davis_root is None:
            return
        train_list = os.path.join(davis_root, 'ImageSets', '2017', 'train.txt')
        if not os.path.exists(train_list):
            print(f"Warning: {train_list} not found, skipping DAVIS")
            return
        with open(train_list) as f:
            names = [l.strip() for l in f if l.strip()]
        cnt = 0
        for vname in names:
            d = os.path.join(davis_root, 'JPEGImages', '480p', vname)
            if not os.path.isdir(d):
                continue
            flist = sorted([fn for fn in os.listdir(d) if fn.endswith(('.jpg', '.png'))])
            if len(flist) >= self.nframes:
                # caption key: 'davis_{vname}' (matching merged YAML)
                caption_key = f"davis_{vname}"
                for _ in range(10):  # 10x oversampling
                    self.video_list.append((d, flist, caption_key))
                cnt += 1
        print(f"  DAVIS: {cnt} videos (x10 = {cnt * 10} entries)")

    def _scan_ytvos(self, ytvos_root):
        """Scan YouTubeVOS dataset."""
        if ytvos_root is None:
            return
        base = os.path.join(ytvos_root, 'JPEGImages')
        if not os.path.isdir(base):
            print(f"Warning: {base} not found, skipping YouTubeVOS")
            return
        cnt = 0
        for vid in sorted(os.listdir(base)):
            d = os.path.join(base, vid)
            if not os.path.isdir(d):
                continue
            flist = sorted([fn for fn in os.listdir(d) if fn.endswith(('.jpg', '.png'))])
            if len(flist) >= self.nframes:
                # caption key: 'ytvos_{vid}' (matching merged YAML)
                caption_key = f"ytvos_{vid}"
                self.video_list.append((d, flist, caption_key))
                cnt += 1
        print(f"  YouTubeVOS: {cnt} videos")

    def __len__(self):
        return len(self.video_list)

    def _get_caption(self, caption_key):
        """Look up caption by key, fallback to default."""
        return self.captions.get(caption_key, self.DEFAULT_CAPTION)

    def tokenize_captions(self, caption):
        if random.random() < self.args.proportion_empty_prompts:
            caption = ""
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def __getitem__(self, index):
        jpeg_dir, frame_list, caption_key = self.video_list[index]

        # 1. Random consecutive nframes
        start = random.randint(0, len(frame_list) - self.nframes)
        selected = frame_list[start: start + self.nframes]

        # 2. Read frames
        images = [Image.open(os.path.join(jpeg_dir, f)).convert('RGB') for f in selected]

        # 3. Generate random masks
        w, h = images[0].size
        all_masks = create_random_shape_with_random_motion(
            len(images), imageHeight=h, imageWidth=w)

        # 4. Process each frame
        frames = []
        masks = []
        masked_images = []
        state = torch.get_rng_state()

        for idx in range(self.nframes):
            img = images[idx]
            mask_pil = all_masks[idx]

            masked_image = np.array(img) * (1.0 - np.array(mask_pil)[:, :, np.newaxis].astype(np.float32) / 255)
            masked_image = Image.fromarray(masked_image.astype(np.uint8))

            torch.set_rng_state(state)
            frames.append(self.img_transform(img))

            torch.set_rng_state(state)
            masked_images.append(self.img_transform(masked_image))

            mask_inv = Image.fromarray(255 - np.array(mask_pil))
            torch.set_rng_state(state)
            masks.append(self.mask_transform(mask_inv))

        # 5. 50% temporal flip
        if random.random() < 0.5:
            frames.reverse()
            masks.reverse()
            masked_images.reverse()

        # 6. Tokenize caption (real caption from YAML)
        caption = self._get_caption(caption_key)
        input_ids = self.tokenize_captions(caption)[0]

        return {
            "pixel_values": torch.stack(frames),
            "conditioning_pixel_values": torch.stack(masked_images),
            "masks": torch.stack(masks),
            "input_ids": input_ids,
        }
