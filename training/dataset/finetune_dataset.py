import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset.utils import create_random_shape_with_random_motion


class FinetuneDataset(torch.utils.data.Dataset):
    """
    Dataset for finetuning DiffuEraser on DAVIS-2017 and YouTubeVOS-2019.
    
    Loads clean video frames from both datasets, generates random masks
    on-the-fly using create_random_shape_with_random_motion.
    
    Data processing pipeline matches the original TrainDataset:
    1. Read consecutive frames from video directory
    2. Generate random brush-stroke masks with motion
    3. Create masked images at original resolution
    4. Resize (short-edge) + center crop to target resolution
    5. Normalize to [-1, 1] for images, [0, 1] for masks
    6. 50% probability temporal flip
    """

    def __init__(self, args, tokenizer):
        self.args = args
        self.nframes = args.nframes
        self.size = args.resolution
        self.tokenizer = tokenizer

        # Transform matching original TrainDataset:
        # Resize short edge to self.size, then center crop to square
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

        self.video_list = []  # list of (jpeg_dir, frame_list)
        self._scan_davis(args.davis_root)
        self._scan_ytvos(args.ytvos_root)
        print(f"FinetuneDataset: total {len(self.video_list)} videos")

    def _scan_davis(self, davis_root):
        """Scan DAVIS dataset using train.txt, 10x oversampling to balance with YTBV."""
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
                for _ in range(10):  # 10x oversampling
                    self.video_list.append((d, flist))
                cnt += 1
        print(f"  DAVIS: {cnt} videos (x10 = {cnt * 10} entries)")

    def _scan_ytvos(self, ytvos_root):
        """Scan YouTubeVOS dataset by listing JPEGImages/ subdirectories."""
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
                self.video_list.append((d, flist))
                cnt += 1
        print(f"  YouTubeVOS: {cnt} videos")

    def __len__(self):
        return len(self.video_list)

    def tokenize_captions(self, caption):
        if random.random() < self.args.proportion_empty_prompts:
            caption = ""
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    def __getitem__(self, index):
        jpeg_dir, frame_list = self.video_list[index]

        # 1. Random consecutive nframes
        start = random.randint(0, len(frame_list) - self.nframes)
        selected = frame_list[start: start + self.nframes]

        # 2. Read frames
        images = [Image.open(os.path.join(jpeg_dir, f)).convert('RGB') for f in selected]

        # 3. Generate random masks (matching original code)
        w, h = images[0].size
        all_masks = create_random_shape_with_random_motion(
            len(images), imageHeight=h, imageWidth=w)

        # 4. Process each frame (matching original TrainDataset logic)
        frames = []
        masks = []
        masked_images = []
        state = torch.get_rng_state()

        for idx in range(self.nframes):
            img = images[idx]
            mask_pil = all_masks[idx]  # L-mode PIL, white=mask region

            # Create masked image at original resolution (matching original L129)
            masked_image = np.array(img) * (1.0 - np.array(mask_pil)[:, :, np.newaxis].astype(np.float32) / 255)
            masked_image = Image.fromarray(masked_image.astype(np.uint8))

            # Apply transforms with same RNG state (matching original L132-138)
            torch.set_rng_state(state)
            frames.append(self.img_transform(img))

            torch.set_rng_state(state)
            masked_images.append(self.img_transform(masked_image))

            # Invert mask: hole=0, valid=255 (matching original L137)
            mask_inv = Image.fromarray(255 - np.array(mask_pil))
            torch.set_rng_state(state)
            masks.append(self.mask_transform(mask_inv))

        # 5. 50% temporal flip (matching original L141-145)
        if random.random() < 0.5:
            frames.reverse()
            masks.reverse()
            masked_images.reverse()

        # 6. Tokenize caption
        input_ids = self.tokenize_captions("clean background")[0]

        return {
            "pixel_values": torch.stack(frames),           # [nframes, 3, H, W], [-1, 1]
            "conditioning_pixel_values": torch.stack(masked_images),  # [nframes, 3, H, W], [-1, 1]
            "masks": torch.stack(masks),                   # [nframes, 1, H, W], [0, 1]
            "input_ids": input_ids,                        # [seq_len]
        }
