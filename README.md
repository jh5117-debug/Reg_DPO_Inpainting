# DiffuEraser Finetune

> **Non-Prompt Finetune** for DiffuEraser video inpainting model.
>
> 代码托管于 **GitHub**，数据集与权重托管于 **HuggingFace**。

---

## 📂 仓库结构

```
DiffuEraser_finetune/
├── train_DiffuEraser_stage1.py     # Stage 1 训练 (UNet2D + BrushNet)
├── train_DiffuEraser_stage2.py     # Stage 2 训练 (Motion Modules)
├── run_train_stage1.py             # Stage 1 训练入口
├── run_train_stage2.py             # Stage 2 训练入口
├── run_train_all.py                # 一键 Stage 1+2 训练入口
├── convert_checkpoint.py           # 手动权重转换 (accelerator → safetensors)
├── validation_metrics.py           # 验证指标 (PSNR/SSIM)
├── 02_train_stage1.sbatch          # SLURM - Stage 1
├── 02_train_stage2.sbatch          # SLURM - Stage 2
├── 02_train_all.sbatch             # SLURM - 一键训练
├── environment.yml                 # Conda 环境配置
├── requirements.txt                # pip 依赖
├── diffueraser/                    # 模型核心 (pipeline, metrics)
├── libs/                           # 自定义 UNet/BrushNet/MotionAdapter
├── dataset/                        # 数据加载模块 (finetune_dataset.py 等)
├── inference/                      # 推理 & 评估
├── data/                           # (gitignored) DAVIS/YTBV eval 数据
└── weights/                        # (gitignored) 预训练权重
```

---

## ⚡ 快速开始

### 1. 环境变量配置

在 `~/.bashrc` 中添加以下 **3 个环境变量**：

```bash
export PROJECT_HOME="/sc-projects/sc-proj-cc09-repair/hongyou"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

| 变量 | 说明 |
|------|------|
| `PROJECT_HOME` | 合作者根路径，代码位于 `$PROJECT_HOME/dev/DiffuEraser_finetune/` |
| `HF_TOKEN` | HuggingFace 下载凭证（[创建 Token](https://huggingface.co/settings/tokens)） |
| `WANDB_API_KEY` | Weights & Biases 远程监控 Key（[获取 Key](https://wandb.ai/authorize)） |

### 2. Clone 代码

```bash
cd ${PROJECT_HOME}/dev/DiffuEraser_finetune

# 清理旧代码文件（保留数据集和权重）
find . -maxdepth 1 \
  ! -name 'dataset' ! -name 'weights' ! -name 'data' \
  ! -name 'finetune-stage1' ! -name 'finetune-stage2' \
  ! -name '.' -exec rm -rf {} +

# 从 GitHub 拉取最新代码
git clone https://github.com/jh5117-debug/Reg_DPO_Inpainting.git .
```

### 3. 下载数据集 & 权重（首次搭建）

从 HuggingFace 下载并解压：

```bash
pip install -U huggingface_hub
huggingface-cli login --token $HF_TOKEN

# 下载代码 + 数据集
huggingface-cli download jh5117/DiffuEraser-finetune-code \
  --local-dir DiffuEraser_downloads --repo-type dataset

# 下载权重
huggingface-cli download jh5117/DiffuEraser-finetune-weights \
  --local-dir DiffuEraser_downloads --repo-type dataset
```

> 解压步骤见各 HF 仓库 README，将 DAVIS/YTBV 放入 `dataset/`，权重放入 `weights/`。

### 4. 安装依赖 & W&B 登录

```bash
conda activate diffueraser
pip install wandb weave
wandb login
```

### 5. 一键训练（推荐 🚀）

```bash
sbatch 02_train_all.sbatch
```

Stage 1 完成后**自动衔接** Stage 2，权重**自动转换**，训练曲线**实时上传** W&B。

### 6. 或者分开训练

```bash
sbatch 02_train_stage1.sbatch
# 等 Stage 1 完成后...
sbatch 02_train_stage2.sbatch
```

---

## 🏗️ 训练架构

### Stage 1: BrushNet + UNet2D 微调

| 组件 | 状态 |
|------|------|
| UNet2D (SD1.5 unet) | ✅ **训练** |
| BrushNet (diffuEraser) | ✅ **训练** |
| VAE (sd-vae-ft-mse) | ❄️ 冻结 |
| Text Encoder (CLIP) | ❄️ 冻结 |

- **输入**: DAVIS + YTBV 视频帧 (默认 nframes=10)
- **输出**: `finetune-stage1/converted_weights/{unet_main, brushnet}/`

### Stage 2: Motion Module 微调

| 组件 | 状态 |
|------|------|
| Motion Modules (AnimateDiff) | ✅ **训练** |
| UNet2D 空间层 (from Stage 1) | ❄️ 冻结 |
| BrushNet (from Stage 1) | ❄️ 冻结 |
| VAE | ❄️ 冻结 |
| Text Encoder | ❄️ 冻结 |

- **输入**: DAVIS + YTBV 视频帧 (默认 nframes=22)
- **前置**: Stage 1 转换权重
- **输出**: `finetune-stage2/converted_weights/{unet_main, brushnet}/`

---

## 📊 W&B 远程监控

训练启动后，在 [W&B Dashboard](https://wandb.ai) 实时查看：

| 指标 | 来源 | 频率 |
|------|------|------|
| `train/loss` | `accelerator.log()` | 每 step |
| `train/lr` | `accelerator.log()` | 每 step |
| `val/video` | 保存的验证视频 | 每 `validation_steps` |

**W&B Project**: `DPO_Diffueraser`

> 合作者在 HPC 上训练时所有 log 自动上传至你的 W&B 账户，无需 SSH 即可监控。

---

## 🔧 手动权重转换

训练结束时权重**自动转换**。如需从历史 checkpoint 手动转换：

```bash
# Stage 1
python convert_checkpoint.py \
  --stage 1 \
  --checkpoint_dir finetune-stage1/checkpoint-50000 \
  --base_model_path weights/stable-diffusion-v1-5 \
  --brushnet_path weights/diffuEraser \
  --output_dir converted_weights/finetuned-stage1

# Stage 2
python convert_checkpoint.py \
  --stage 2 \
  --checkpoint_dir finetune-stage2/checkpoint-50000 \
  --base_model_path weights/stable-diffusion-v1-5 \
  --brushnet_path weights/diffuEraser \
  --motion_adapter_path weights/animatediff-motion-adapter-v1-5-2 \
  --pretrained_stage1 converted_weights/finetuned-stage1 \
  --output_dir converted_weights/finetuned-stage2
```

---

## 🔧 训练参数修改指引

参数通过 `run_train_*.py` 的命令行控制：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--num_gpus` | GPU 数量 | `1` |
| `--max_train_steps` | 最大训练步数 | `50000` |
| `--learning_rate` | 学习率 | `5e-6` |
| `--batch_size` | 批大小 | `1` |
| `--nframes` | 每次采样帧数 | Stage1: `10`, Stage2: `22` |
| `--checkpointing_steps` | 保存间隔 | `2000` |
| `--validation_steps` | 验证间隔 | `2000` |
| `--wandb_project` | W&B 项目名 | `DPO_Diffueraser` |

> [!TIP]
> **CUDA OOM?** → 减小 `--batch_size`、`--nframes` 或 `--resolution`。训练脚本默认启用 `--gradient_checkpointing`。

---

## 📦 预训练权重目录

```
weights/
├── stable-diffusion-v1-5/              # SD1.5 base model
├── diffuEraser/                        # DiffuEraser BrushNet + UNet
├── sd-vae-ft-mse/                      # VAE
└── animatediff-motion-adapter-v1-5-2/  # AnimateDiff MotionAdapter
```

---

## 🔄 断点续训

所有脚本均默认启用 `--resume_from_checkpoint="latest"`。训练中断后直接重新 `sbatch` 即可从最后 checkpoint 恢复。

---

## 📋 附录

<details>
<summary>附录 A：完成后的目录结构</summary>

```
$PROJECT_HOME/dev/DiffuEraser_finetune/
├── train_DiffuEraser_stage1.py
├── train_DiffuEraser_stage2.py
├── run_train_stage1.py
├── run_train_stage2.py
├── run_train_all.py
├── convert_checkpoint.py
├── validation_metrics.py
├── 02_train_stage1.sbatch
├── 02_train_stage2.sbatch
├── 02_train_all.sbatch
├── environment.yml
├── requirements.txt
├── .gitignore
├── README.md
├── diffueraser/
├── libs/
├── dataset/
│   ├── finetune_dataset.py
│   ├── file_client.py
│   ├── img_util.py
│   ├── DAVIS/                          ← (gitignored) 训练数据
│   └── YTBV/                           ← (gitignored) 训练数据
├── data/eval/DAVIS/                    ← (gitignored) 验证数据
├── weights/                            ← (gitignored) 预训练权重
├── finetune-stage1/                    ← (gitignored) 训练 checkpoint
├── finetune-stage2/                    ← (gitignored) 训练 checkpoint
├── inference/
└── PRD/
```

</details>

<details>
<summary>附录 B：DPO 数据集 (后续使用)</summary>

DPO 数据集用于后续 DPO (Direct Preference Optimization) 训练：

```bash
huggingface-cli download jh5117/DPO-dataset \
    --repo-type dataset \
    --local-dir "$PROJECT_HOME/DPO-dataset"
```

**统计**: 60 DAVIS + 1964 YTBV = **2024 个视频对**

</details>

<details>
<summary>附录 C：常见问题</summary>

**Q1: CUDA OOM 怎么办？**
减小 `--nframes` / `--batch_size`，训练脚本已默认启用 `--gradient_checkpointing`。

**Q2: 下载缓存可以删除吗？**
搭建完成后即可安全删除 `$PROJECT_HOME/DiffuEraser_downloads`。

**Q3: 训练中断了怎么办？**
脚本已启用 `--resume_from_checkpoint="latest"`，直接重新 `sbatch` 即可恢复。

**Q4: GPU partition 名不是 `pgpu` 怎么办？**
修改 `.sbatch` 文件中的 `#SBATCH --partition=你的partition名`。

</details>

---

## 📎 相关仓库

- **数据集 & 权重**: [HuggingFace](https://huggingface.co/jh5117)
- **推理代码**: 见 `inference/` 目录
- **PRD 文档**: 见 `PRD/` 目录
