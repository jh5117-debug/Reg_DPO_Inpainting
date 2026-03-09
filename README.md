# DiffuEraser Finetune

> **Non-Prompt Finetune** for DiffuEraser video inpainting model.
>
> 代码托管于 **GitHub**，数据集与权重托管于 **HuggingFace**。

---

## 📂 仓库结构

```
项目根目录/
│
├── train_DiffuEraser_stage1.py     # Stage 1 核心训练脚本 (UNet2D + BrushNet)
├── train_DiffuEraser_stage2.py     # Stage 2 核心训练脚本 (Motion Modules)
├── convert_checkpoint.py           # 手动权重转换 (accelerator → safetensors)
├── validation_metrics.py           # 训练验证指标 (PSNR/SSIM → inference/metrics.py)
│
├── scripts/                        # 🚀 训练入口 & SLURM
│   ├── run_train_stage1.py         #    Stage 1 Python 入口
│   ├── run_train_stage2.py         #    Stage 2 Python 入口
│   ├── run_train_all.py            #    一键 Stage 1+2 入口
│   ├── 02_train_stage1.sbatch      #    SLURM - Stage 1
│   ├── 02_train_stage2.sbatch      #    SLURM - Stage 2
│   └── 02_train_all.sbatch         #    SLURM - 一键训练
│
├── dataset/                        # 📊 数据加载 Python 模块
│   ├── finetune_dataset.py
│   ├── file_client.py
│   ├── img_util.py
│   └── ...
│
├── data/                           # 📁 训练数据 (gitignored)
│   ├── DAVIS/
│   └── YTBV/
│
├── weights/                        # 📁 预训练权重 (gitignored)
│   ├── stable-diffusion-v1-5/
│   ├── diffuEraser/
│   ├── sd-vae-ft-mse/
│   └── animatediff-motion-adapter-v1-5-2/
│
├── diffueraser/                    # 🧠 模型核心 pipeline
├── libs/                           # 🔧 自定义网络层
├── inference/                      # 🔍 推理 & 评估
├── tools/                          # 🛠️ 工具脚本
├── docs/                           # 📖 文档 & PRD
│
├── environment.yml
├── requirements.txt
└── .gitignore
```

> [!IMPORTANT]
> `dataset/` = **Python 代码模块**（数据加载逻辑，Git 跟踪）
> `data/` = **实际训练数据**（DAVIS/YTBV 视频，Git 忽略）

---

## ⚡ 快速开始

### 1. 环境变量

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 2. Clone 代码到已有数据目录

```bash
cd /path/to/your_project_dir    # 已有 data/ weights/ 的目录

find . -maxdepth 1 \
  ! -name 'data' ! -name 'weights' \
  ! -name 'finetune-stage1' ! -name 'finetune-stage2' \
  ! -name 'logs' ! -name '.' -exec rm -rf {} +

git clone https://github.com/jh5117-debug/Reg_DPO_Inpainting.git .
```

> 脚本自动检测自身所在目录作为项目根路径，**无需任何环境变量**。
> 只需 `data/` 和 `weights/` 在项目根目录下即可。

### 3. 一键训练

```bash
sbatch scripts/02_train_all.sbatch
```

### 4. 分开训练

```bash
sbatch scripts/02_train_stage1.sbatch
sbatch scripts/02_train_stage2.sbatch   # Stage 1 完成后
```

### 5. 自定义参数

```bash
# 修改学习率和步数
LR=1e-5 MAX_STEPS=100000 sbatch scripts/02_train_stage1.sbatch

# 多卡
NUM_GPUS=2 sbatch --gres=gpu:2 scripts/02_train_stage1.sbatch

# 自定义数据/权重路径
DATA_DIR=/other/path/data WEIGHTS_DIR=/other/path/weights \
  sbatch scripts/02_train_stage1.sbatch
```

---

## 🔧 可控参数一览

| 环境变量 | 含义 | 默认值 |
|---------|------|--------|
| `DATA_DIR` | 数据目录 (含 DAVIS/ YTBV/) | `<项目根>/data/` |
| `WEIGHTS_DIR` | 预训练权重目录 | `<项目根>/weights/` |
| `NUM_GPUS` | GPU 数量 | `1` |
| `BATCH_SIZE` | 批大小 | `1` |
| `GRAD_ACCUM` | 梯度累积 | `4` |
| `LR` | 学习率 | `5e-6` |
| `MAX_STEPS` | 最大训练步 | `50000` |
| `CKPT_STEPS` | checkpoint 间隔 | `2000` |
| `CKPT_LIMIT` | checkpoint 保留数 | `3` |
| `VAL_STEPS` | 验证间隔 | `2000` |
| `NFRAMES` | 采样帧数 | Stage1:`10` / Stage2:`22` |

---

## 📊 W&B 远程监控

| 指标 | 频率 |
|------|------|
| `train/loss`, `train/lr` | 每 step |
| `val/psnr`, `val/ssim` | 每 `VAL_STEPS` |
| `val/video_*` | 每 `VAL_STEPS` |

**Project**: `DPO_Diffueraser` — 登录 [wandb.ai](https://wandb.ai) 即可查看。

---

## 🔄 断点续训

默认启用 `--resume_from_checkpoint="latest"`。中断后重新 `sbatch` 即可恢复。
