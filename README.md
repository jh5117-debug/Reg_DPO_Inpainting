# DiffuEraser Finetune

> **Non-Prompt Finetune** for DiffuEraser video inpainting model.

---

## 📂 仓库结构

```
Reg_DPO_Inpainting/
│
├── train_DiffuEraser_stage1.py     # Stage 1 核心训练脚本 (UNet2D + BrushNet)
├── train_DiffuEraser_stage2.py     # Stage 2 核心训练脚本 (Motion Modules)
├── convert_checkpoint.py           # 手动权重转换 (accelerator → safetensors)
├── validation_metrics.py           # 训练验证指标 (PSNR/SSIM → inference/metrics.py)
│
├── scripts/                        # 🚀 训练入口 & SLURM
│   ├── run_train_stage1.py
│   ├── run_train_stage2.py
│   ├── run_train_all.py
│   ├── 02_train_stage1.sbatch
│   ├── 02_train_stage2.sbatch
│   └── 02_train_all.sbatch
│
├── dataset/                        # 📊 数据加载 Python 模块 (Git 跟踪)
│   ├── finetune_dataset.py
│   ├── file_client.py
│   └── ...
│
├── data/                           # 📁 训练数据 (Git 忽略)
│   ├── DAVIS/
│   └── YTBV/
│
├── weights/                        # 📁 预训练权重 (Git 忽略)
│   ├── stable-diffusion-v1-5/
│   ├── diffuEraser/
│   ├── sd-vae-ft-mse/
│   └── animatediff-motion-adapter-v1-5-2/
│
├── diffueraser/                    # 模型核心 pipeline
├── libs/                           # 自定义网络层
├── inference/                      # 推理 & 评估
├── tools/                          # 工具脚本
├── docs/                           # 文档
├── environment.yml
├── requirements.txt
└── .gitignore
```

> [!IMPORTANT]
> `dataset/` = Python 代码模块（数据加载逻辑，Git 跟踪）
> `data/` = 实际训练数据（DAVIS/YTBV 视频帧，Git 忽略）
> ⚠️ 如果 `data/` 里有 `.py` 文件，那是 HF 下载残留，可以安全删除：`cd data && rm -f *.py && rm -rf __pycache__`

---

## ⚡ 快速开始

### 1. 环境变量（~/.bashrc）

```bash
export PROJECT_HOME="/sc-projects/sc-proj-cc09-repair/hongyou"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
export WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 2. 拉取最新代码

```bash
cd ${PROJECT_HOME}/dev/Reg_DPO_Inpainting
git pull
```

### 3. 清理 data/ 中的 .py 残留文件（首次执行）

```bash
cd data && rm -f *.py && rm -rf __pycache__ && cd ..
```

### 4. 训练

```bash
sbatch scripts/02_train_all.sbatch
```

---

## 🔧 参数控制

所有训练参数通过**环境变量**覆盖，sbatch 命令行中显式传递 `--data_dir` 和 `--weights_dir`：

```bash
# 默认参数直接提交
sbatch scripts/02_train_stage1.sbatch

# 自定义参数
LR=1e-5 MAX_STEPS=100000 NUM_GPUS=2 \
  sbatch --gres=gpu:2 scripts/02_train_stage1.sbatch

# 自定义数据/权重路径
DATA_DIR=/other/path/data WEIGHTS_DIR=/other/path/weights \
  sbatch scripts/02_train_stage1.sbatch
```

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
| `VAL_STEPS` | 验证间隔 | `2000` |
| `NFRAMES` | 采样帧数 | Stage1:`10` / Stage2:`22` |

---

## 📊 W&B 远程监控

需要设置环境变量：`export WANDB_API_KEY="xxx"`

| 指标 | 频率 |
|------|------|
| `train/loss`, `train/lr` | 每 step |
| `val/psnr`, `val/ssim` | 每 `VAL_STEPS` |
| `val/video_*` (GIF) | 每 `VAL_STEPS` |

**W&B Project**: `DPO_Diffueraser`

---

## 🔄 断点续训

默认启用 `--resume_from_checkpoint="latest"`。中断后重新 `sbatch` 即可恢复。
