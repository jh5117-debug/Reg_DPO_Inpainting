# DiffuEraser Finetune

> **Non-Prompt Finetune** for DiffuEraser video inpainting model.
>
> 代码托管于 **GitHub**，数据集与权重托管于 **HuggingFace**。

---

## 📂 仓库结构

```
DiffuEraser_finetune/
│
├── train_DiffuEraser_stage1.py     # Stage 1 核心训练脚本 (UNet2D + BrushNet)
├── train_DiffuEraser_stage2.py     # Stage 2 核心训练脚本 (Motion Modules)
├── convert_checkpoint.py           # 手动权重转换 (accelerator → safetensors)
├── validation_metrics.py           # 训练验证指标 (PSNR/SSIM, 调用 inference/metrics.py)
│
├── scripts/                        # 🚀 训练入口 & SLURM
│   ├── run_train_stage1.py         #    Stage 1 Python 入口
│   ├── run_train_stage2.py         #    Stage 2 Python 入口
│   ├── run_train_all.py            #    一键 Stage 1+2 Python 入口
│   ├── 02_train_stage1.sbatch      #    SLURM - Stage 1
│   ├── 02_train_stage2.sbatch      #    SLURM - Stage 2
│   └── 02_train_all.sbatch         #    SLURM - 一键训练
│
├── diffueraser/                    # 🧠 模型核心 (pipeline, metrics)
├── libs/                           # 🔧 自定义网络层 (UNet/BrushNet/MotionAdapter)
├── dataset/                        # 📊 数据加载模块
├── inference/                      # 🔍 推理 & 评估
│
├── tools/                          # 🛠️ 工具脚本
│   └── score_inpainting_quality.py
│
├── docs/                           # 📖 文档 & PRD
│   ├── pipeline_refactoring_plan.md
│   ├── experiment_report.md
│   └── ...
│
├── environment.yml                 # Conda 环境
├── requirements.txt                # pip 依赖
└── .gitignore
```

---

## ⚡ 快速开始

### 1. 环境变量配置

在 `~/.bashrc` 中添加：

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

| 变量 | 说明 |
|------|------|
| `HF_TOKEN` | HuggingFace 下载凭证（[创建 Token](https://huggingface.co/settings/tokens)） |
| `WANDB_API_KEY` | Weights & Biases 远程监控 Key（[获取 Key](https://wandb.ai/authorize)） |

### 2. Clone 代码（到已有数据目录）

假设你已经在 `DiffuEraser_finetune/` 下准备好了 `dataset/` 和 `weights/`：

```bash
cd /path/to/DiffuEraser_finetune

# 清理旧代码文件（保留数据集和权重）
find . -maxdepth 1 \
  ! -name 'dataset' ! -name 'weights' ! -name 'data' \
  ! -name 'finetune-stage1' ! -name 'finetune-stage2' \
  ! -name 'logs' ! -name '.' -exec rm -rf {} +

# 拉取最新代码（所有脚本会自动检测自身所在目录作为项目根路径）
git clone https://github.com/jh5117-debug/Reg_DPO_Inpainting.git .
```

> [!IMPORTANT]
> **无需设置 `PROJECT_HOME`** — 所有脚本通过自身文件路径自动检测项目根目录。
> 只要 `dataset/`、`weights/` 在项目根目录下即可。

### 3. 下载数据集 & 权重（首次搭建）

```bash
pip install -U huggingface_hub
huggingface-cli login --token $HF_TOKEN

huggingface-cli download jh5117/DiffuEraser-finetune-code \
  --local-dir DiffuEraser_downloads --repo-type dataset

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
sbatch scripts/02_train_all.sbatch
```

Stage 1 完成后**自动衔接** Stage 2，权重**自动转换**，训练曲线**实时上传** W&B。

### 6. 分开训练

```bash
sbatch scripts/02_train_stage1.sbatch
# 等 Stage 1 完成后...
sbatch scripts/02_train_stage2.sbatch
```

### 7. 自定义参数训练

所有参数可通过**环境变量**覆盖：

```bash
LR=1e-5 BATCH_SIZE=2 MAX_STEPS=100000 NFRAMES=12 \
  sbatch scripts/02_train_stage1.sbatch

NUM_GPUS=8 sbatch --gres=gpu:8 scripts/02_train_stage2.sbatch
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
| `val/psnr` | `inference/metrics.py` → `validation_metrics.py` | 每 `validation_steps` |
| `val/ssim` | `inference/metrics.py` → `validation_metrics.py` | 每 `validation_steps` |
| `val/video_*` | 验证推理 GIF | 每 `validation_steps` |

**W&B Project**: `DPO_Diffueraser`

---

## 🔧 训练参数一览

通过环境变量控制（每个都有默认值）：

| 环境变量 | 含义 | Stage1 默认 | Stage2 默认 |
|---------|------|-----------|-----------|
| `NUM_GPUS` | GPU 数量 | 1 | 1 |
| `BATCH_SIZE` | 批大小 | 1 | 1 |
| `GRAD_ACCUM` | 梯度累积步数 | 4 | 4 |
| `LR` | 学习率 | 5e-6 | 5e-6 |
| `LR_SCHEDULER` | 调度器 | constant | constant |
| `LR_WARMUP` | 预热步数 | 500 | 500 |
| `MAX_STEPS` | 最大训练步 | 50000 | 50000 |
| `CKPT_STEPS` | checkpoint 间隔 | 2000 | 2000 |
| `CKPT_LIMIT` | checkpoint 上限 | 3 | 3 |
| `VAL_STEPS` | 验证间隔 | 2000 | 2000 |
| `NFRAMES` | 采样帧数 | **10** | **22** |
| `SEED` | 随机种子 | 42 | 42 |

> [!TIP]
> **CUDA OOM?** → 减小 `BATCH_SIZE`、`NFRAMES`。训练脚本默认启用 `--gradient_checkpointing`。

---

## 🔧 手动权重转换

训练结束时权重**自动转换**。如需从历史 checkpoint 手动转换：

```bash
python convert_checkpoint.py \
  --stage 1 \
  --checkpoint_dir finetune-stage1/checkpoint-50000 \
  --base_model_path weights/stable-diffusion-v1-5 \
  --brushnet_path weights/diffuEraser \
  --output_dir converted_weights/finetuned-stage1
```

---

## 🔄 断点续训

所有脚本均默认启用 `--resume_from_checkpoint="latest"`。训练中断后直接重新 `sbatch` 即可恢复。

---

## 📎 相关仓库

- **数据集 & 权重**: [HuggingFace](https://huggingface.co/jh5117)
- **推理代码**: 见 `inference/` 目录
- **设计文档**: 见 `docs/` 目录
