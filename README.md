# DiffuEraser Finetune

> 合作者完整操作指南 — 从零开始搭建 DiffuEraser Finetune 训练环境
>
> **全流程 SLURM 自动化**：只需修改配置 → 提交两个 sbatch → 等待训练完成

---

## ⚡ 快速开始（3 步完成）

1. **编辑配置** → 修改 `01_setup.sbatch` 和 `02_train.sbatch` 中的 `HF_TOKEN` 和 `TRAIN_MODE`
2. **提交解压** → `sbatch 01_setup.sbatch` （下载 + 解压 + 环境安装）
3. **提交训练** → `sbatch 02_train.sbatch` （Stage1 → 转换 → Stage2 → 转换 → 完成）

> ⚠️ 必须等 `01_setup.sbatch` 成功完成后再提交 `02_train.sbatch`

---

## 📦 HuggingFace 仓库

| # | 仓库 | 内容 | 大小 |
|---|------|------|------|
| 1 | `JiaHuang01/DiffuEraser-finetune-code` | 基础代码 + DAVIS/YTBV 数据集 | ~10.1 GB |
| 2 | `JiaHuang01/DiffuEraser-finetune-prompt-code` | Prompt 版代码 + VLM Captions | ~0.8 MB |
| 3 | `JiaHuang01/DiffuEraser-finetune-weights` | 模型权重 (SD1.5 + DiffuEraser + VAE + MotionAdapter) | ~48.4 GB |
| 4 | `JiaHuang01/DPO-dataset` | DPO 训练数据 (60 DAVIS + 1964 YTBV 视频对) | ~84 GB |

> [!NOTE]
> 仓库 4 (`DPO-dataset`) 为后续 DPO 训练保留，当前 finetune 流程不使用。

---

## 📖 详细操作指南

### 第零步：前置准备

**必须完成以下 3 项设置：**

```bash
# 1. 安装 HuggingFace CLI
pip install -U huggingface_hub

# 2. 登录 HuggingFace
huggingface-cli login
```

> ⚠️ **HuggingFace Token**：在 https://huggingface.co/settings/tokens 创建一个 Token（Read 权限即可）.
> 该 Token 将在 **SLURM 脚本** 中使用。

```bash
# 3. 设置 PROJECT_HOME 环境变量（所有资源都放在这个目录下）
export PROJECT_HOME="/path/to/your/project"    # <-- 替换为你的实际路径
```

> 💡 **建议**：将 `export PROJECT_HOME=...` 添加到 `~/.bashrc` 中，避免每次重新设置。

---

### 第一步：创建并提交解压脚本 (`01_setup.sbatch`)

将以下内容保存为 `01_setup.sbatch`，**修改前 3 个配置变量后**直接提交：

```bash
#!/bin/bash
#SBATCH --job-name=DiffuEraser_Setup
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/setup-%j.out

set -euo pipefail

############################################################
# ========== ⚙️ 用户配置区 (只需修改这里) ⚙️ =========== #
############################################################
# 1. 你的 HuggingFace Token
HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"              # <-- 必填

# 2. 训练模式: "nonprompt" 或 "prompt" 或 "both"
TRAIN_MODE="nonprompt"                        # <-- 必填

# 3. 项目根目录 (如果已在 ~/.bashrc 中设置 PROJECT_HOME 则无需修改)
PROJECT_HOME="${PROJECT_HOME:-$HOME/project}"  # <-- 按需修改
############################################################

echo "============================================"
echo "[SETUP] Job ID: $SLURM_JOB_ID"
echo "[SETUP] Node: $(hostname)"
echo "[SETUP] PROJECT_HOME: $PROJECT_HOME"
echo "[SETUP] TRAIN_MODE: $TRAIN_MODE"
echo "============================================"

export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# ── 1. 下载 HuggingFace 数据 ──────────────────────────
DOWNLOADS="$PROJECT_HOME/DiffuEraser_downloads"
mkdir -p "$DOWNLOADS"

echo "[SETUP] ===== [1/3] 下载基础代码 + 数据集 ====="
huggingface-cli download JiaHuang01/DiffuEraser-finetune-code \
    --repo-type dataset \
    --local-dir "$DOWNLOADS/finetune-code"

echo "[SETUP] ===== [2/3] 下载 Prompt 版代码 + Captions ====="
huggingface-cli download JiaHuang01/DiffuEraser-finetune-prompt-code \
    --repo-type dataset \
    --local-dir "$DOWNLOADS/finetune-prompt-code"

echo "[SETUP] ===== [3/3] 下载模型权重 ====="
huggingface-cli download JiaHuang01/DiffuEraser-finetune-weights \
    --repo-type dataset \
    --local-dir "$DOWNLOADS/finetune-weights"

echo "[SETUP] 下载完成！"

# ── 2. 搭建 Non-Prompt 版 ─────────────────────────────
FINETUNE_DIR="$PROJECT_HOME/DiffuEraser_finetune"

if [ "$TRAIN_MODE" = "nonprompt" ] || [ "$TRAIN_MODE" = "both" ]; then
  echo "[SETUP] 搭建 Non-Prompt 版..."
  mkdir -p "$FINETUNE_DIR"

  cp "$DOWNLOADS/finetune-code/code_base.tar.gz" "$FINETUNE_DIR/"
  cp "$DOWNLOADS/finetune-code/DAVIS.tar"         "$FINETUNE_DIR/"
  cp "$DOWNLOADS/finetune-code/YTBV.tar"          "$FINETUNE_DIR/"
  cp "$DOWNLOADS/finetune-code/environment.yml"   "$FINETUNE_DIR/"
  cp "$DOWNLOADS/finetune-code/setup_project.sh"  "$FINETUNE_DIR/"

  mkdir -p "$FINETUNE_DIR/weights"
  cp -r "$DOWNLOADS/finetune-weights/"* "$FINETUNE_DIR/weights/"

  cd "$FINETUNE_DIR"
  bash setup_project.sh
  mkdir -p logs converted_weights

  echo "[SETUP] Non-Prompt 版搭建完成 ✅"
fi

# ── 3. 搭建 Prompt 版 ─────────────────────────────────
FINETUNE_PROMPT_DIR="$PROJECT_HOME/DiffuEraser_finetune_prompt"

if [ "$TRAIN_MODE" = "prompt" ] || [ "$TRAIN_MODE" = "both" ]; then
  echo "[SETUP] 搭建 Prompt 版..."

  # 如果是 both 模式，Non-Prompt 版已搭建好，直接符号链接
  if [ "$TRAIN_MODE" = "both" ]; then
    mkdir -p "$FINETUNE_PROMPT_DIR"
    cd "$FINETUNE_PROMPT_DIR"

    tar -xzf "$DOWNLOADS/finetune-code/code_base.tar.gz"
    tar -xzf "$DOWNLOADS/finetune-prompt-code/code_base_prompt.tar.gz"
    tar -xzf "$DOWNLOADS/finetune-prompt-code/captions.tar.gz"

    mkdir -p dataset
    ln -sf "$FINETUNE_DIR/dataset/DAVIS" dataset/DAVIS
    ln -sf "$FINETUNE_DIR/dataset/YTBV"  dataset/YTBV
    ln -sf "$FINETUNE_DIR/weights"       weights
  else
    # prompt-only: 需要先搭建基础环境
    mkdir -p "$FINETUNE_DIR"
    cp "$DOWNLOADS/finetune-code/code_base.tar.gz" "$FINETUNE_DIR/"
    cp "$DOWNLOADS/finetune-code/DAVIS.tar"         "$FINETUNE_DIR/"
    cp "$DOWNLOADS/finetune-code/YTBV.tar"          "$FINETUNE_DIR/"
    cp "$DOWNLOADS/finetune-code/environment.yml"   "$FINETUNE_DIR/"
    cp "$DOWNLOADS/finetune-code/setup_project.sh"  "$FINETUNE_DIR/"

    mkdir -p "$FINETUNE_DIR/weights"
    cp -r "$DOWNLOADS/finetune-weights/"* "$FINETUNE_DIR/weights/"

    cd "$FINETUNE_DIR"
    bash setup_project.sh

    mkdir -p "$FINETUNE_PROMPT_DIR"
    cd "$FINETUNE_PROMPT_DIR"

    tar -xzf "$DOWNLOADS/finetune-code/code_base.tar.gz"
    tar -xzf "$DOWNLOADS/finetune-prompt-code/code_base_prompt.tar.gz"
    tar -xzf "$DOWNLOADS/finetune-prompt-code/captions.tar.gz"

    mkdir -p dataset
    ln -sf "$FINETUNE_DIR/dataset/DAVIS" dataset/DAVIS
    ln -sf "$FINETUNE_DIR/dataset/YTBV"  dataset/YTBV
    ln -sf "$FINETUNE_DIR/weights"       weights
  fi

  mkdir -p logs converted_weights
  echo "[SETUP] Prompt 版搭建完成 ✅"
fi

# ── 4. 安装 Conda 环境 ────────────────────────────────
echo "[SETUP] 安装 Conda 环境..."
source ~/.bashrc

if conda env list 2>/dev/null | grep -q "diffueraser"; then
  echo "[SETUP] 环境 diffueraser 已存在，跳过安装"
else
  cd "$FINETUNE_DIR"
  conda env create -f environment.yml
  echo "[SETUP] 环境创建完成 ✅"
fi

# ── 5. 配置 Accelerate (非交互式) ─────────────────────
echo "[SETUP] 配置 accelerate..."
source activate diffueraser 2>/dev/null || conda activate diffueraser

mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'ACCEL_EOF'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
ACCEL_EOF
echo "[SETUP] Accelerate 配置完成 ✅"

echo "============================================"
echo "[SETUP] 🎉 全部搭建完成！"
echo "[SETUP] 请检查无误后提交训练脚本："
echo "[SETUP]   sbatch 02_train.sbatch"
echo "============================================"
```

提交方式：

```bash
mkdir -p logs
sbatch 01_setup.sbatch
```

查看日志：`tail -f logs/setup-*.out`

---

### 第二步：创建并提交训练脚本 (`02_train.sbatch`)

将以下内容保存为 `02_train.sbatch`，**确保用户配置区与 `01_setup.sbatch` 一致**后提交：

```bash
#!/bin/bash
#SBATCH --job-name=DiffuEraser_Train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train-%j.out

set -euo pipefail

############################################################
# ========== ⚙️ 用户配置区 (只需修改这里) ⚙️ =========== #
############################################################
# 1. 你的 HuggingFace Token
HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"              # <-- 必填

# 2. 训练模式: "nonprompt" 或 "prompt"
TRAIN_MODE="nonprompt"                        # <-- 必填

# 3. 项目根目录
PROJECT_HOME="${PROJECT_HOME:-$HOME/project}"  # <-- 按需修改

# 4. GPU 数量 (根据你的集群配置修改，也可命令行传入: bash 02_train.sbatch 1)
NUM_GPUS=${1:-8}                              # <-- 按需修改
############################################################

echo "============================================"
echo "[TRAIN] Job ID: $SLURM_JOB_ID"
echo "[TRAIN] Node: $(hostname)"
echo "[TRAIN] GPUs: $NUM_GPUS"
echo "[TRAIN] TRAIN_MODE: $TRAIN_MODE"
echo "[TRAIN] PROJECT_HOME: $PROJECT_HOME"
echo "============================================"

export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# ── 激活环境 ──────────────────────────────────────────
source ~/.bashrc
conda activate diffueraser 2>/dev/null || source activate diffueraser

# ── 确定工作目录 ──────────────────────────────────────
if [ "$TRAIN_MODE" = "prompt" ]; then
  WORK_DIR="$PROJECT_HOME/DiffuEraser_finetune_prompt"
  CAPTION_YAML="$WORK_DIR/captions/all_captions_merged.yaml"
  CAPTION_ARG="--caption_yaml=${CAPTION_YAML}"
  JOB_TAG="Prompt"
else
  WORK_DIR="$PROJECT_HOME/DiffuEraser_finetune"
  CAPTION_ARG=""
  JOB_TAG="NonPrompt"
fi

cd "$WORK_DIR"
echo "[TRAIN] Working dir: $(pwd)"

WEIGHTS="$WORK_DIR/weights"
DAVIS="$WORK_DIR/dataset/DAVIS"

# ── 路径替换 (自动修复硬编码) ─────────────────────────
echo "[TRAIN] 替换硬编码路径..."

for f in finetune_stage1.sh finetune_stage2.sh \
         save_checkpoint_stage1.py save_checkpoint_stage2.py \
         train_DiffuEraser_stage1.py train_DiffuEraser_stage2.py; do
    if [ -f "$f" ]; then
        sed -i "s|/home/hj/Train_Diffueraser_prompt|${WORK_DIR}|g"  "$f"
        sed -i "s|/home/hj/Train_Diffueraser|${WORK_DIR}|g"         "$f"
        sed -i "s|/home/hj/DiffuEraser_new/weights|${WEIGHTS}|g"    "$f"
        echo "  Fixed: $f"
    fi
done

# ── 验证数据集 ────────────────────────────────────────
echo "[TRAIN] 验证数据集..."
[ -d "$WORK_DIR/dataset/DAVIS" ] || { echo "ERROR: DAVIS not found!"; exit 1; }
[ -d "$WORK_DIR/dataset/YTBV" ]  || { echo "ERROR: YTBV not found!"; exit 1; }
[ -d "$WEIGHTS/stable-diffusion-v1-5" ] || { echo "ERROR: SD1.5 weights not found!"; exit 1; }

if [ "$TRAIN_MODE" = "prompt" ]; then
  [ -f "$CAPTION_YAML" ] || { echo "ERROR: Caption YAML not found!"; exit 1; }
fi
echo "[TRAIN] 数据集验证通过 ✅"

# ── 公共验证参数 ──────────────────────────────────────
validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"

mkdir -p logs converted_weights

############################################################
# STAGE 1: FINETUNE
############################################################
echo "========================================================"
echo "[TRAIN] Stage 1 Training ($JOB_TAG)..."
echo "========================================================"

accelerate launch \
  --multi_gpu \
  --num_processes $NUM_GPUS \
  --mixed_precision bf16 \
  train_DiffuEraser_stage1.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1_path="${WEIGHTS}/diffuEraser" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --davis_root="$WORK_DIR/dataset/DAVIS" \
  --ytvos_root="$WORK_DIR/dataset/YTBV" \
  $CAPTION_ARG \
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
  --gradient_checkpointing \
  --max_train_steps=50000

echo "[TRAIN] Stage 1 完成 ✅"

############################################################
# CONVERT STAGE 1 CHECKPOINT
############################################################
echo "[TRAIN] 转换 Stage 1 权重..."
LATEST_CKPT=$(ls -d finetune-stage1/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
CKPT_NAME=$(basename "$LATEST_CKPT")

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found!"
    exit 1
fi
echo "[TRAIN] Found: $LATEST_CKPT"

cp save_checkpoint_stage1.py save_checkpoint_stage1_temp.py
sed -i "s|checkpoint-xxxx|$CKPT_NAME|g" save_checkpoint_stage1_temp.py
python save_checkpoint_stage1_temp.py
rm save_checkpoint_stage1_temp.py
echo "[TRAIN] Stage 1 权重转换完成 ✅"

############################################################
# STAGE 2: FINETUNE
############################################################
echo "========================================================"
echo "[TRAIN] Stage 2 Training ($JOB_TAG)..."
echo "========================================================"

FINETUNED_STAGE1="$WORK_DIR/converted_weights/finetuned-stage1"

accelerate launch \
  --multi_gpu \
  --num_processes $NUM_GPUS \
  --mixed_precision fp16 \
  train_DiffuEraser_stage2.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1="${FINETUNED_STAGE1}" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --motion_adapter_path="${WEIGHTS}/animatediff-motion-adapter-v1-5-2" \
  --davis_root="$WORK_DIR/dataset/DAVIS" \
  --ytvos_root="$WORK_DIR/dataset/YTBV" \
  $CAPTION_ARG \
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
  --max_train_steps=50000

echo "[TRAIN] Stage 2 完成 ✅"

############################################################
# CONVERT STAGE 2 CHECKPOINT
############################################################
echo "[TRAIN] 转换 Stage 2 权重..."
LATEST_CKPT_S2=$(ls -d finetune-stage2/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
CKPT_NAME_S2=$(basename "$LATEST_CKPT_S2")

if [ -z "$LATEST_CKPT_S2" ]; then
    echo "ERROR: Stage 2 checkpoint not found!"
    exit 1
fi
echo "[TRAIN] Found: $LATEST_CKPT_S2"

cp save_checkpoint_stage2.py save_checkpoint_stage2_temp.py
sed -i "s|checkpoint-xxxx|$CKPT_NAME_S2|g" save_checkpoint_stage2_temp.py
python save_checkpoint_stage2_temp.py
rm save_checkpoint_stage2_temp.py

echo "============================================"
echo "[TRAIN] 🎉 全部训练完成！"
echo "[TRAIN] 最终权重位于:"
echo "[TRAIN]   Stage 1: $WORK_DIR/converted_weights/finetuned-stage1/"
echo "[TRAIN]   Stage 2: $WORK_DIR/converted_weights/finetuned-stage2/"
echo "============================================"
```

提交方式（`01_setup.sbatch` 成功完成后）：

```bash
sbatch 02_train.sbatch
```

查看日志：`tail -f logs/train-*.out`

---

## 🔧 训练参数修改指引

如果需要调整训练参数，**在 `02_train.sbatch` 中直接修改以下内容**：

| 参数 | 含义 | 默认值 | 位置 |
|------|------|--------|------|
| `NUM_GPUS` | GPU 数量 | `8` | 用户配置区 |
| `--max_train_steps` | 最大训练步数 | `50000` | Stage 1/2 的 `accelerate launch` 命令 |
| `--learning_rate` | 学习率 | `5e-06` | Stage 1/2 的 `accelerate launch` 命令 |
| `--nframes` | 每次采样帧数 | Stage1: `10`, Stage2: `22` | Stage 1/2 的 `accelerate launch` 命令 |
| `--resolution` | 训练分辨率 | `512` | Stage 1/2 的 `accelerate launch` 命令 |
| `--train_batch_size` | 批大小 | `1` | Stage 1/2 的 `accelerate launch` 命令 |
| `--checkpointing_steps` | 保存间隔 | `2000` | Stage 1/2 的 `accelerate launch` 命令 |
| `--validation_steps` | 验证间隔 | `2000` | Stage 1/2 的 `accelerate launch` 命令 |
| `--gradient_checkpointing` | 梯度检查点 (省显存) | 仅 Stage1 | Stage 1 的 `accelerate launch` 命令 |

> [!TIP]
> **显存不足 (CUDA OOM)?** → 添加 `--gradient_checkpointing` 到 Stage 2，或减小 `--nframes` / `--resolution`

> [!TIP]
> **SLURM 时间限制**：默认 `--time=48:00:00`。如训练需要更久，可修改 `02_train.sbatch` 头部的 `#SBATCH --time`

> [!IMPORTANT]
> **GPU Partition 名称**：`02_train.sbatch` 中默认使用 `--partition=gpu`。如果你的集群 GPU partition 名不同（如 `pgpu`、`v100` 等），请修改 `#SBATCH --partition=` 行。

---

## 📋 附录

<details>
<summary>附录 A：完成后的目录结构</summary>

```
$PROJECT_HOME/
├── DiffuEraser_downloads/                  ← 下载缓存（搭建完成后可删除）
│   ├── finetune-code/
│   ├── finetune-prompt-code/
│   └── finetune-weights/
│
├── DiffuEraser_finetune/                   ← Non-Prompt 版
│   ├── train_DiffuEraser_stage1.py
│   ├── train_DiffuEraser_stage2.py
│   ├── finetune_stage1.sh
│   ├── finetune_stage2.sh
│   ├── save_checkpoint_stage1.py
│   ├── save_checkpoint_stage2.py
│   ├── environment.yml
│   ├── libs/
│   ├── diffueraser/
│   ├── dataset/
│   │   ├── DAVIS/                          ← 真实数据 (~844 MB)
│   │   ├── YTBV/                           ← 真实数据 (~5.26 GB)
│   │   ├── finetune_dataset.py
│   │   └── utils.py ...
│   ├── weights/
│   │   ├── stable-diffusion-v1-5/
│   │   ├── diffuEraser/
│   │   ├── sd-vae-ft-mse/
│   │   └── animatediff-motion-adapter-v1-5-2/
│   ├── finetune-stage1/                    ← 训练 checkpoint
│   ├── finetune-stage2/                    ← 训练 checkpoint
│   ├── converted_weights/                  ← ⭐ 最终输出权重
│   │   ├── finetuned-stage1/
│   │   └── finetuned-stage2/
│   └── logs/
│
├── DiffuEraser_finetune_prompt/            ← Prompt 版 (仅 TRAIN_MODE=both 时存在)
│   ├── train_DiffuEraser_stage1.py         ← Prompt 版 (使用 caption)
│   ├── train_DiffuEraser_stage2.py         ← Prompt 版 (使用 caption)
│   ├── captions/
│   │   └── all_captions_merged.yaml        ← 训练用 (3561 条)
│   ├── dataset/
│   │   ├── DAVIS → 符号链接
│   │   └── YTBV  → 符号链接
│   ├── weights → 符号链接
│   ├── converted_weights/                  ← ⭐ 最终输出权重
│   └── logs/
│
└── DPO-dataset/                            ← 🔮 DPO 训练数据 (后续使用)
    ├── davis_bear/
    │   ├── gt_frames/                      ← 原始帧 (PNG)
    │   ├── masks/                          ← 对象 mask
    │   ├── neg_frames_1/                   ← 负样本帧 (类型1)
    │   ├── neg_frames_2/                   ← 负样本帧 (类型2)
    │   ├── meta.json                       ← 生成参数 + 评分
    │   └── comparison.mp4                  ← 可视化对比视频
    ├── davis_bmx-bumps/
    ├── ... (共 60 个 davis_ 场景)
    ├── ytbv_003234408d/
    ├── ... (共 1964 个 ytbv_ 场景)
    └── (总计 2024 个视频对, ~84 GB)
```

</details>

<details>
<summary>附录 B：两种版本的核心区别</summary>

| 项目 | Non-Prompt 版 | Prompt 版 |
|------|--------------|-----------| 
| 训练 Caption | 硬编码 `"clean background"` | VLM 生成的真实场景描述 |
| Dataset 类 | `FinetuneDataset` | `FinetuneDatasetWithCaption` |
| 新增训练参数 | 无 | `--caption_yaml` |
| Caption 数据 | 无 | `captions/all_captions_merged.yaml` |
| Caption 数量 | 0 | 3561 (90 DAVIS + 3471 YTVOS) |
| 权重 / 数据集 | 相同 | 相同 (符号链接) |

</details>

<details>
<summary>附录 C：DPO 数据集结构说明</summary>

`DPO-dataset` 用于后续 **DPO (Direct Preference Optimization)** 训练，包含通过 DiffuEraser 推理生成的正负样本对：

| 字段 | 说明 |
|------|------|
| `gt_frames/` | Ground truth 原始帧 |
| `masks/` | 对象 mask (标注需要擦除的区域) |
| `neg_frames_1/` | 负样本 — 最差维度1 (ghosting/hallucination/flicker) |
| `neg_frames_2/` | 负样本 — 最差维度2 |
| `meta.json` | 生成元数据：neg 生成方式、chunk 分段、评分 |
| `comparison.mp4` | 可视化对比视频 |

**统计**：60 DAVIS + 1964 YTBV = **2024 个视频对**

**下载 DPO 数据集** (后续需要时使用):

```bash
huggingface-cli download JiaHuang01/DPO-dataset \
    --repo-type dataset \
    --local-dir "$PROJECT_HOME/DPO-dataset"
```

</details>

<details>
<summary>附录 D：常见问题</summary>

**Q1: CUDA OOM 怎么办？**
在 `02_train.sbatch` 中添加或确认 `--gradient_checkpointing`，或减小 `--nframes` / `--resolution`

**Q2: 如何查看训练是否使用了 Caption？**
查看训练启动日志，Prompt 版会打印 `"FinetuneDatasetWithCaption: total X videos, 3561 captions loaded"`，Non-Prompt 版会打印 `"FinetuneDataset: total X videos"`

**Q3: 两个版本可以同时运行吗？**
可以。它们在不同目录，互不影响。确保 GPU 数量足够即可。

**Q4: 下载缓存可以删除吗？**
搭建完成后即可安全删除 `$PROJECT_HOME/DiffuEraser_downloads`，节省磁盘空间。

**Q5: 训练中断了怎么办？**
脚本已启用 `--resume_from_checkpoint="latest"`，直接重新 `sbatch 02_train.sbatch` 即可从最后的 checkpoint 恢复。

**Q6: GPU partition 名不是 `gpu` 怎么办？**
修改 `02_train.sbatch` 第 3 行：`#SBATCH --partition=你的partition名`

**Q7: 没有 8 张 GPU 怎么办？/ 如何单卡调试？**
修改 `02_train.sbatch` 中的 `NUM_GPUS=1`（脚本会自动切换单卡/多卡模式）。如果通过 SLURM 提交，同时修改 `#SBATCH --gres=gpu:N`。

</details>
