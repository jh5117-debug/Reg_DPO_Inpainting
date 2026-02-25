# Region-Reg-DPO 数据集构建 PRD

> 项目目录：`/home/hj/Train_Diffueraser/`

---

## 一、需求概述

为 Region-Reg-DPO Stage 3 训练构建偏好对数据集。包含：
1. **负样本批量生成脚本** (`generate_dpo_negatives.py`)
2. **DPO Dataset 类** (`dataset/dpo_dataset.py`)
3. **三区 Mask 分解工具** (`dataset/region_mask_utils.py`)

---

## 二、文件清单

| 文件 | 操作 | 位置 |
|------|------|------|
| `dataset/region_mask_utils.py` | **新建** | 三区 mask 分解 (M_h, M_b, M_c) |
| `dataset/dpo_dataset.py` | **新建** | DPO 偏好对训练 Dataset |
| `generate_dpo_negatives.py` | **新建** | 负样本批量生成脚本 |

---

## 三、负样本生成策略

| 类型 | 方法 | 质量缺陷 |
|------|------|----------|
| hallucination | DiffuEraser (priori=None) | 语义幻觉 |
| blur | 仅 ProPainter | 缺乏高频纹理 |
| flicker | 分段不同 seed 推理拼接 | 时间不连续 |

---

## 四、输出目录结构

```
dpo_data/
├── {video_name}/
│   ├── gt_frames/          ← 正样本 GT 帧
│   ├── masks/              ← mask 序列 (L-mode PNG)
│   ├── hallucination/      ← 幻觉负样本帧
│   ├── blur/               ← 模糊负样本帧
│   └── flicker/            ← 闪烁负样本帧
└── manifest.json           ← 全局索引
```

---

## 五、使用方法

### 5.1 生成负样本（小规模验证）

```bash
cd /home/hj/Train_Diffueraser

CUDA_VISIBLE_DEVICES=1 python generate_dpo_negatives.py \
    --davis_root dataset/DAVIS \
    --ytvos_root dataset/YTBV \
    --output_dir dpo_data \
    --nframes 22 \
    --max_videos 100 \
    --neg_types hallucination blur flicker \
    --base_model_path /home/hj/DiffuEraser_new/weights/stable-diffusion-v1-5 \
    --vae_path /home/hj/DiffuEraser_new/weights/sd-vae-ft-mse \
    --diffueraser_path /home/hj/DiffuEraser_new/weights/diffuEraser \
    --propainter_model_dir /home/hj/DiffuEraser_new/weights/propainter \
    --pcm_weights_path /home/hj/DiffuEraser_new/weights/PCM_Weights
```

### 5.2 断点恢复

```bash
python generate_dpo_negatives.py \
    ... (同上参数) \
    --resume
```

### 5.3 DPO Dataset 使用

```python
from dataset.dpo_dataset import DPODataset

args.dpo_data_root = "dpo_data"
args.nframes = 22
args.resolution = 512
args.proportion_empty_prompts = 0
args.boundary_dilation = 7

dataset = DPODataset(args, tokenizer)
sample = dataset[0]

# 返回字段:
# pixel_values_pos, pixel_values_neg,
# conditioning_pos, conditioning_neg,
# masks, mask_hole, mask_boundary, mask_context,
# input_ids, neg_type
```

---

## 六、注意事项

1. **生成耗时**：每个视频需跑 ProPainter + DiffuEraser，全量约数天
2. **建议先小规模验证**：100-200 个视频确认质量后再全量
3. **Mask 格式**：L-mode PNG, 白色(255)=hole, 黑色(0)=valid
4. **三区分解**：latent space 膨胀核自动缩放，boundary dilation=7(pixel) → 3(latent)
5. **DPO Dataset 返回的 masks 字段**：0=hole, 1=valid（与 BrushNet 一致）
