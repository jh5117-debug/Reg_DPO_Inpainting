# DiffuEraser Finetune 完整需求文档（最终版）

> 项目目录：`/home/hj/Train_Diffueraser/`
> 原始代码参考：`/home/hj/Train_Diffueraser/DiffuEraser_orign/`

---

## 一、项目目标

在 YouTubeVOS-2019 和 DAVIS-2017 的干净视频帧上，使用随机合成 mask，
对 DiffuEraser 进行两阶段 finetune，获得微调权重。

**核心事实：训练时 mask 是随机合成的，不使用数据集的物体分割标注。**

---

## 二、所有路径汇总

### 2.1 项目结构

```
/home/hj/Train_Diffueraser/
├── dataset/
│   ├── DAVIS/                          ← DAVIS 数据集
│   │   ├── JPEGImages/480p/            ← 训练用（干净帧GT）
│   │   ├── Annotations/480p/           ← 训练不用，仅验证用
│   │   └── ImageSets/2017/train.txt    ← 训练集视频列表
│   └── YTBV/                           ← YouTubeVOS 数据集
│       ├── JPEGImages/                 ← 训练用（干净帧GT）
│       ├── Annotations/                ← 训练不用
│       └── meta.json                   ← 不用
├── DiffuEraser_orign/                  ← 原始 DiffuEraser 代码（参考）
│   ├── dataset/
│   │   ├── load_dataset.py
│   │   └── utils.py                    ← create_random_shape_with_random_motion
│   ├── libs/
│   ├── diffueraser/
│   └── ...
├── dataset/                            ← finetune 用的 dataset 模块
│   ├── __init__.py
│   ├── utils.py                        ← 从 DiffuEraser_orign 复制
│   └── finetune_dataset.py             ← 【新建】
├── libs/                               ← 从 DiffuEraser_orign 复制或软链接
├── diffueraser/                        ← 从 DiffuEraser_orign 复制或软链接
├── train_DiffuEraser_stage1.py         ← 【修改】
├── train_DiffuEraser_stage2.py         ← 【修改】
├── finetune_stage1.sh                  ← 【新建】
├── finetune_stage2.sh                  ← 【新建】
├── save_checkpoint_stage1.py           ← 【新建】
├── save_checkpoint_stage2.py           ← 【修改】
├── finetune-stage1/                    ← 训练输出（自动生成）
├── finetune-stage2/                    ← 训练输出（自动生成）
└── train.md
```

### 2.2 权重路径（不搬，引用原位置）

```
/home/hj/DiffuEraser_new/weights/
├── stable-diffusion-v1-5/              ← base_model_name_or_path
├── sd-vae-ft-mse/                      ← vae_path
├── diffuEraser/                        ← DiffuEraser 原始权重
│   └── brushnet/                       ← brushnet_model_name_or_path
├── animatediff-motion-adapter-v1-5-2/  ← motion_adapter_path（需确认名称）
├── converted_weights/
│   ├── diffuEraser-model-stage1/
│   │   └── checkpoint-1/              ← pretrained_stage1_path
│   │       ├── unet_main/
│   │       └── brushnet/
│   └── diffuEraser-model-stage2/
│       └── checkpoint-2/
│           ├── unet_main/
│           └── brushnet/
├── propainter/
├── PCM_Weights/
└── i3d_rgb_imagenet.pt
```

### 2.3 数据集路径

| 用途 | 路径 |
|------|------|
| DAVIS 帧 | `/home/hj/Train_Diffueraser/dataset/DAVIS/JPEGImages/480p/` |
| DAVIS 标注（仅验证） | `/home/hj/Train_Diffueraser/dataset/DAVIS/Annotations/480p/` |
| DAVIS 训练列表 | `/home/hj/Train_Diffueraser/dataset/DAVIS/ImageSets/2017/train.txt` |
| YTBV 帧 | `/home/hj/Train_Diffueraser/dataset/YTBV/JPEGImages/` |
| davis_root 参数 | `/home/hj/Train_Diffueraser/dataset/DAVIS` |
| ytvos_root 参数 | `/home/hj/Train_Diffueraser/dataset/YTBV` |

> ⚠️ **请先在服务器上确认以下命令有正确输出：**
> ```bash
> ls /home/hj/Train_Diffueraser/dataset/DAVIS/JPEGImages/480p/ | head -5
> ls /home/hj/Train_Diffueraser/dataset/DAVIS/ImageSets/2017/train.txt
> ls /home/hj/Train_Diffueraser/dataset/YTBV/JPEGImages/ | head -5
> ls /home/hj/DiffuEraser_new/weights/converted_weights/diffuEraser-model-stage1/checkpoint-1/unet_main/
> ```
> 如果 DAVIS 内部还有一层 `DAVIS/` 目录（即 `dataset/DAVIS/DAVIS/JPEGImages/`），
> 则 davis_root 应该改为 `/home/hj/Train_Diffueraser/dataset/DAVIS/DAVIS`。

---

## 三、训练原理回顾

### 3.1 mask 生成

训练时调用 `dataset/utils.py` 中的 `create_random_shape_with_random_motion`：
- 生成随机笔刷 stroke mask
- mask 在帧间有连续运动
- 每次 __getitem__ 随机生成新 mask
- **完全不使用 Annotations 目录**

### 3.2 两阶段分工

| | Stage 1 | Stage 2 |
|--|---------|---------|
| 模型 | UNet2D + BrushNet | UNetMotion(=UNet2D+MotionAdapter) + BrushNet |
| 训练对象 | UNet2D + BrushNet 全部参数 | 仅 MotionAdapter temporal layers |
| 冻结 | 无 | UNet2D空间层 + BrushNet |
| nframes | 10 | 22 |
| 能力 | 空间/单帧 inpainting | 时序一致性 |

---

## 四、新建 dataset/finetune_dataset.py

```python
import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from dataset.utils import create_random_shape_with_random_motion


class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.nframes = args.nframes
        self.size = args.resolution
        self.tokenizer = tokenizer

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])  # [-1, 1]
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # [0, 1]

        self.video_list = []  # (jpeg_dir, frame_list)
        self._scan_davis(args.davis_root)
        self._scan_ytvos(args.ytvos_root)
        print(f"FinetuneDataset: total {len(self.video_list)} videos")

    def _scan_davis(self, davis_root):
        train_list = os.path.join(davis_root, 'ImageSets/2017/train.txt')
        if not os.path.exists(train_list):
            print(f"Warning: {train_list} not found, skipping DAVIS")
            return
        with open(train_list) as f:
            names = [l.strip() for l in f if l.strip()]
        cnt = 0
        for vname in names:
            d = os.path.join(davis_root, 'JPEGImages/480p', vname)
            if not os.path.isdir(d):
                continue
            flist = sorted([f for f in os.listdir(d) if f.endswith(('.jpg','.png'))])
            if len(flist) >= self.nframes:
                for _ in range(10):  # 10x oversampling
                    self.video_list.append((d, flist))
                cnt += 1
        print(f"  DAVIS: {cnt} videos (x10 = {cnt*10} entries)")

    def _scan_ytvos(self, ytvos_root):
        base = os.path.join(ytvos_root, 'JPEGImages')
        if not os.path.isdir(base):
            print(f"Warning: {base} not found, skipping YouTubeVOS")
            return
        cnt = 0
        for vid in sorted(os.listdir(base)):
            d = os.path.join(base, vid)
            if not os.path.isdir(d):
                continue
            flist = sorted([f for f in os.listdir(d) if f.endswith(('.jpg','.png'))])
            if len(flist) >= self.nframes:
                self.video_list.append((d, flist))
                cnt += 1
        print(f"  YouTubeVOS: {cnt} videos")

    def __len__(self):
        return len(self.video_list)

    def _resize_and_crop(self, img, is_mask=False):
        w, h = img.size
        scale = self.size / min(w, h)
        new_w, new_h = round(w * scale), round(h * scale)
        interp = Image.NEAREST if is_mask else Image.BILINEAR
        img = img.resize((new_w, new_h), interp)
        left = (new_w - self.size) // 2
        top = (new_h - self.size) // 2
        return img.crop((left, top, left + self.size, top + self.size))

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

        # 1. 随机选 nframes 个连续帧
        start = random.randint(0, len(frame_list) - self.nframes)
        selected = frame_list[start : start + self.nframes]

        # 2. 读取帧
        images = [Image.open(os.path.join(jpeg_dir, f)).convert('RGB') for f in selected]

        # 3. 随机生成 mask（与原代码一致）
        w, h = images[0].size
        all_masks = create_random_shape_with_random_motion(
            len(images), imageHeight=h, imageWidth=w)

        # 4. 处理每帧
        frames, masks, masked_images = [], [], []
        state = torch.get_rng_state()

        for idx in range(self.nframes):
            img = images[idx]
            mask_pil = all_masks[idx]

            # masked image（原始分辨率下做）
            mask_np = np.array(mask_pil)[:,:,np.newaxis].astype(np.float32) / 255.0
            masked_img = Image.fromarray(
                (np.array(img) * (1.0 - mask_np)).astype(np.uint8))

            # resize + crop
            img = self._resize_and_crop(img)
            masked_img = self._resize_and_crop(masked_img)
            mask_gray = Image.fromarray(255 - np.array(mask_pil))  # hole=0, valid=255
            mask_gray = self._resize_and_crop(mask_gray, is_mask=True)

            # to tensor
            torch.set_rng_state(state)
            frames.append(self.img_transform(img))
            torch.set_rng_state(state)
            masked_images.append(self.img_transform(masked_img))
            torch.set_rng_state(state)
            masks.append(self.mask_transform(mask_gray))

        # 5. 50% 时序翻转
        if random.random() < 0.5:
            frames.reverse(); masks.reverse(); masked_images.reverse()

        # 6. tokenize
        input_ids = self.tokenize_captions("clean background")[0]

        return {
            "pixel_values": torch.stack(frames),
            "conditioning_pixel_values": torch.stack(masked_images),
            "masks": torch.stack(masks),
            "input_ids": input_ids,
        }
```

---

## 五、训练脚本修改

### 5.1 train_DiffuEraser_stage1.py 修改点

**修改 1：新增参数**（在 parse_args 中添加）
```python
parser.add_argument("--davis_root", type=str,
    default="/home/hj/Train_Diffueraser/dataset/DAVIS")
parser.add_argument("--ytvos_root", type=str,
    default="/home/hj/Train_Diffueraser/dataset/YTBV")
parser.add_argument("--pretrained_stage1_path", type=str, default=None,
    help="已有stage1权重路径，用于finetune")
```

**修改 2：替换 Dataset 导入和实例化**
```python
# 删除：
# from dataset.load_dataset import TrainDataset
# train_dataset = TrainDataset(args, tokenizer)

# 替换为：
from dataset.finetune_dataset import FinetuneDataset
train_dataset = FinetuneDataset(args, tokenizer)
```

**修改 3：模型加载逻辑**
```python
# 在 main() 中，替换 unet 和 brushnet 的加载逻辑：

if args.pretrained_stage1_path:
    logger.info("Loading from pretrained stage1 for finetuning")
    brushnet = BrushNetModel.from_pretrained(
        args.pretrained_stage1_path, subfolder="brushnet")
    unet_main = UNet2DConditionModel.from_pretrained(
        args.pretrained_stage1_path, subfolder="unet_main")
else:
    logger.info("Loading from base model + brushnet")
    unet_main = UNet2DConditionModel.from_pretrained(
        args.base_model_name_or_path, subfolder="unet",
        revision=args.revision, variant=args.variant)
    if args.brushnet_model_name_or_path:
        brushnet = BrushNetModel.from_pretrained(args.brushnet_model_name_or_path)
    else:
        brushnet = BrushNetModel.from_unet(unet_main)
```

### 5.2 train_DiffuEraser_stage2.py 修改点

修改 1（新增参数）和修改 2（替换Dataset）与 Stage 1 相同。
模型加载不需要改（Stage 2 已有 --pretrained_stage1 参数）。

---

## 六、Shell 脚本

### 6.1 finetune_stage1.sh

```bash
#!/bin/bash
cd /home/hj/Train_Diffueraser

WEIGHTS="/home/hj/DiffuEraser_new/weights"
DAVIS="/home/hj/Train_Diffueraser/dataset/DAVIS"

validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"

accelerate launch --mixed_precision "fp16" \
  train_DiffuEraser_stage1.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1_path="${WEIGHTS}/converted_weights/diffuEraser-model-stage1/checkpoint-1" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --davis_root="/home/hj/Train_Diffueraser/dataset/DAVIS" \
  --ytvos_root="/home/hj/Train_Diffueraser/dataset/YTBV" \
  --resolution=512 \
  --nframes=10 \
  --mixed_precision="fp16" \
  --train_batch_size=1 \
  --dataloader_num_workers=2 \
  --learning_rate=5e-06 \
  --resume_from_checkpoint="latest" \
  --validation_steps=2000 \
  --output_dir="finetune-stage1" \
  --logging_dir="logs-finetune-stage1" \
  --validation_image="$validation_image" \
  --validation_mask="$validation_mask" \
  --validation_prompt="$validation_prompt" \
  --checkpointing_steps=2000 \
  >finetune-stage1.log 2>&1
```

### 6.2 finetune_stage2.sh

```bash
#!/bin/bash
cd /home/hj/Train_Diffueraser

WEIGHTS="/home/hj/DiffuEraser_new/weights"
DAVIS="/home/hj/Train_Diffueraser/dataset/DAVIS"

# ⚠️ 改为你 Stage 1 finetune 转换后的实际路径
FINETUNED_STAGE1="/home/hj/Train_Diffueraser/converted_weights/finetuned-stage1"

validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"

accelerate launch --mixed_precision "fp16" \
  train_DiffuEraser_stage2.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1="${FINETUNED_STAGE1}" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --motion_adapter_path="${WEIGHTS}/animatediff-motion-adapter-v1-5-2" \
  --davis_root="/home/hj/Train_Diffueraser/dataset/DAVIS" \
  --ytvos_root="/home/hj/Train_Diffueraser/dataset/YTBV" \
  --resolution=512 \
  --nframes=22 \
  --mixed_precision="fp16" \
  --train_batch_size=1 \
  --dataloader_num_workers=2 \
  --learning_rate=5e-06 \
  --resume_from_checkpoint="latest" \
  --validation_steps=2000 \
  --output_dir="finetune-stage2" \
  --logging_dir="logs-finetune-stage2" \
  --validation_image="$validation_image" \
  --validation_mask="$validation_mask" \
  --validation_prompt="$validation_prompt" \
  --checkpointing_steps=2000 \
  >finetune-stage2.log 2>&1
```

---

## 七、权重转换脚本

### 7.1 save_checkpoint_stage1.py（新建）

```python
from accelerate import Accelerator
import os
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel

## ===== 修改这三个路径 =====
# 用于初始化模型结构的预训练权重
pretrained_path = "/home/hj/DiffuEraser_new/weights/converted_weights/diffuEraser-model-stage1/checkpoint-1"
# accelerator 保存的 checkpoint 目录
input_dir = "/home/hj/Train_Diffueraser/finetune-stage1/checkpoint-xxxx"  # 改为实际步数
# 转换后输出目录
output_dir = "/home/hj/Train_Diffueraser/converted_weights/finetuned-stage1"
## =========================

accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='fp16')

unet_main = UNet2DConditionModel.from_pretrained(pretrained_path, subfolder="unet_main")
brushnet = BrushNetModel.from_pretrained(pretrained_path, subfolder="brushnet")

unet_main, brushnet = accelerator.prepare(unet_main, brushnet)
accelerator.load_state(input_dir)

unet_main = accelerator.unwrap_model(unet_main)
brushnet = accelerator.unwrap_model(brushnet)

os.makedirs(os.path.join(output_dir, "unet_main"), exist_ok=True)
unet_main.save_pretrained(os.path.join(output_dir, "unet_main"))

os.makedirs(os.path.join(output_dir, "brushnet"), exist_ok=True)
brushnet.save_pretrained(os.path.join(output_dir, "brushnet"))

print('Stage 1 checkpoint saved!')
```

### 7.2 save_checkpoint_stage2.py（修改路径）

```python
from accelerate import Accelerator
import os
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import MotionAdapter, UNetMotionModel

## ===== 修改这些路径 =====
base_model_name_or_path = "/home/hj/DiffuEraser_new/weights/stable-diffusion-v1-5"
pretrained_brushnet_path = "/home/hj/Train_Diffueraser/converted_weights/finetuned-stage1/brushnet"
motion_path = "/home/hj/DiffuEraser_new/weights/animatediff-motion-adapter-v1-5-2"
input_dir = "/home/hj/Train_Diffueraser/finetune-stage2/checkpoint-xxxx"  # 改为实际步数
output_dir = "/home/hj/Train_Diffueraser/converted_weights/finetuned-stage2"
## =========================

accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision='fp16')

unet = UNet2DConditionModel.from_pretrained(base_model_name_or_path, subfolder="unet")
motion_adapter = MotionAdapter.from_pretrained(motion_path)
unet_main = UNetMotionModel.from_unet2d(unet, motion_adapter)
brushnet = BrushNetModel.from_pretrained(pretrained_brushnet_path)

unet_main, brushnet = accelerator.prepare(unet_main, brushnet)
accelerator.load_state(input_dir)

unet_main = accelerator.unwrap_model(unet_main)
brushnet = accelerator.unwrap_model(brushnet)

os.makedirs(os.path.join(output_dir, "unet_main"), exist_ok=True)
unet_main.save_pretrained(os.path.join(output_dir, "unet_main"))

os.makedirs(os.path.join(output_dir, "brushnet"), exist_ok=True)
brushnet.save_pretrained(os.path.join(output_dir, "brushnet"))

print('Stage 2 checkpoint saved!')
```

---

## 八、初始化项目（首次执行）

在开始写代码前，先准备好项目目录：

```bash
cd /home/hj/Train_Diffueraser

# 1. 确认数据集
ls dataset/DAVIS/JPEGImages/480p/ | head -5
ls dataset/DAVIS/ImageSets/2017/train.txt
ls dataset/YTBV/JPEGImages/ | head -5

# 2. 确认权重
ls /home/hj/DiffuEraser_new/weights/converted_weights/diffuEraser-model-stage1/checkpoint-1/

# 3. 从原始代码复制必要模块（如果还没复制）
cp -r DiffuEraser_orign/libs/ .
cp -r DiffuEraser_orign/diffueraser/ .
cp DiffuEraser_orign/dataset/utils.py dataset/
cp DiffuEraser_orign/dataset/__init__.py dataset/ 2>/dev/null
cp DiffuEraser_orign/dataset/file_client.py dataset/
cp DiffuEraser_orign/dataset/img_util.py dataset/

# 4. 复制训练脚本
cp DiffuEraser_orign/train_DiffuEraser_stage1.py .
cp DiffuEraser_orign/train_DiffuEraser_stage2.py .

# 5. 创建输出目录
mkdir -p converted_weights
```

---

## 九、完整执行流程

```
Step 1: 初始化项目（第八节的命令）

Step 2: 新建 dataset/finetune_dataset.py（第四节）

Step 3: 修改 train_DiffuEraser_stage1.py（第五节）

Step 4: 新建 finetune_stage1.sh（第六节）→ bash finetune_stage1.sh

Step 5: 等训练完成 → 修改 save_checkpoint_stage1.py 中的 checkpoint 步数
        → python save_checkpoint_stage1.py

Step 6: 修改 train_DiffuEraser_stage2.py（第五节）

Step 7: 新建 finetune_stage2.sh（第六节）→ bash finetune_stage2.sh

Step 8: 等训练完成 → 修改 save_checkpoint_stage2.py 中的 checkpoint 步数
        → python save_checkpoint_stage2.py

最终产出：
  /home/hj/Train_Diffueraser/converted_weights/finetuned-stage1/
      ├── unet_main/
      └── brushnet/
  /home/hj/Train_Diffueraser/converted_weights/finetuned-stage2/
      ├── unet_main/     (UNetMotionModel)
      └── brushnet/
```

---

## 十、文件清单

| 文件 | 操作 | 位置 |
|------|------|------|
| dataset/finetune_dataset.py | **新建** | /home/hj/Train_Diffueraser/dataset/ |
| train_DiffuEraser_stage1.py | **修改** | /home/hj/Train_Diffueraser/ |
| train_DiffuEraser_stage2.py | **修改** | /home/hj/Train_Diffueraser/ |
| finetune_stage1.sh | **新建** | /home/hj/Train_Diffueraser/ |
| finetune_stage2.sh | **新建** | /home/hj/Train_Diffueraser/ |
| save_checkpoint_stage1.py | **新建** | /home/hj/Train_Diffueraser/ |
| save_checkpoint_stage2.py | **新建** | /home/hj/Train_Diffueraser/ |

---

## 十一、注意事项

1. **mask 是随机合成的**，完全不用 Annotations 目录。

2. **数据量不均衡**：YTBV ~3471 视频 vs DAVIS ~60 视频。
   DAVIS 做 10x oversampling → 约 600 条目 → 占比 ~15%。

3. **motion_adapter_path 需确认**：
   ```bash
   ls /home/hj/DiffuEraser_new/weights/ | grep -i anim
   ```
   确认目录名是否确实是 `animatediff-motion-adapter-v1-5-2`。

4. **dataset/utils.py 必须存在**且包含 `create_random_shape_with_random_motion` 函数。

5. **显存估算**：
   - Stage 1 (nframes=10, batch=1, fp16): ~20-24GB
   - Stage 2 (nframes=22, batch=1, fp16): ~30-40GB

6. **finetune 步数建议**：先跑 10000-20000 步看效果。

7. **学习率**：finetune 用 5e-06（原始训练用 1e-05）。