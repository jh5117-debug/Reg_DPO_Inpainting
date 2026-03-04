# DiffuEraser Finetune

> 合作者完整操作指南 — 从零开始搭建 DiffuEraser Non-Prompt Finetune 训练环境
>
> **全流程 SLURM 自动化**：环境准备 → 提交 Stage1 → 提交 Stage2

---

## ⚡ 快速开始（4 步完成）

1. **前置准备** → 设置 `PROJECT_HOME` 和 `HF_TOKEN` 环境变量
2. **提交解压** → `sbatch 01_setup.sbatch` （下载 + 解压 + 环境安装）
3. **提交 Stage1** → `sbatch 02_train_stage1.sbatch` （Stage1 训练 + 权重转换）
4. **提交 Stage2** → `sbatch 02_train_stage2.sbatch` （Stage2 训练 + 权重转换）

> ⚠️ 必须按顺序执行：`01_setup` 完成后提交 `stage1`，`stage1` 完成后提交 `stage2`

---

## 📦 HuggingFace 仓库

| # | 仓库 | 内容 | 大小 |
|---|------|------|------|
| 1 | `JiaHuang01/DiffuEraser-finetune-code` | 基础代码 + DAVIS/YTBV 数据集 | ~10.1 GB |
| 2 | `JiaHuang01/DiffuEraser-finetune-weights` | 模型权重 (SD1.5 + DiffuEraser + VAE + MotionAdapter) | ~48.4 GB |
| 3 | `JiaHuang01/DPO-dataset` | DPO 训练数据 (60 DAVIS + 1964 YTBV 视频对) | ~84 GB |

> [!NOTE]
> 仓库 3 (`DPO-dataset`) 为后续 DPO 训练保留，当前 finetune 流程不使用。

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
> 将 Token 设置为环境变量：`export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"` （建议添加到 `~/.bashrc`）

```bash
# 3. 设置 PROJECT_HOME 环境变量（所有资源都放在这个目录下）
export PROJECT_HOME="/path/to/your/project"    # <-- 替换为你的实际路径
```

> 💡 **建议**：将 `export PROJECT_HOME=...` 添加到 `~/.bashrc` 中，避免每次重新设置。

---

### 第一步：创建并提交解压脚本 (`01_setup.sbatch`)

将以下内容保存为 `01_setup.sbatch`，确保已设置 `PROJECT_HOME` 和 `HF_TOKEN` 环境变量后直接提交：

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

set -e

export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

echo "============================================"
echo "[SETUP] Job ID: $SLURM_JOB_ID"
echo "[SETUP] Node: $(hostname)"
echo "[SETUP] PROJECT_HOME: $PROJECT_HOME"
echo "============================================"

# ── 1. 下载 HuggingFace 数据 ──────────────────────────
DOWNLOADS="$PROJECT_HOME/DiffuEraser_downloads"
mkdir -p "$DOWNLOADS"

echo "[SETUP] ===== [1/2] 下载基础代码 + 数据集 ====="
huggingface-cli download JiaHuang01/DiffuEraser-finetune-code \
    --repo-type dataset \
    --local-dir "$DOWNLOADS/finetune-code"


echo "[SETUP] ===== [2/2] 下载模型权重 ====="
huggingface-cli download JiaHuang01/DiffuEraser-finetune-weights \
    --repo-type dataset \
    --local-dir "$DOWNLOADS/finetune-weights"

echo "[SETUP] 下载完成！"

# ── 2. 搭建工作目录 ───────────────────────────────────
WORK_DIR="${PROJECT_HOME}/dev"
echo "[SETUP] 搭建工作目录: $WORK_DIR"
mkdir -p "$WORK_DIR"

cp "$DOWNLOADS/finetune-code/code_base.tar.gz" "$WORK_DIR/"
cp "$DOWNLOADS/finetune-code/DAVIS.tar"         "$WORK_DIR/"
cp "$DOWNLOADS/finetune-code/YTBV.tar"          "$WORK_DIR/"
cp "$DOWNLOADS/finetune-code/environment.yml"   "$WORK_DIR/"
cp "$DOWNLOADS/finetune-code/setup_project.sh"  "$WORK_DIR/"

mkdir -p "$WORK_DIR/weights"
cp -r "$DOWNLOADS/finetune-weights/"* "$WORK_DIR/weights/"

cd "$WORK_DIR"
bash setup_project.sh
mkdir -p logs converted_weights

echo "[SETUP] 工作目录搭建完成 ✅"

# ── 3. 安装 Conda 环境 ────────────────────────────────
echo "[SETUP] 安装 Conda 环境..."
source ~/.bashrc

if conda env list 2>/dev/null | grep -q "diffueraser"; then
  echo "[SETUP] 环境 diffueraser 已存在，跳过安装"
else
  cd "$WORK_DIR"
  conda env create -f environment.yml
  echo "[SETUP] 环境创建完成 ✅"
fi

# ── 4. 配置 Accelerate (非交互式) ─────────────────────
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
num_processes: 1
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
echo "[SETUP] 请检查无误后依次提交训练脚本："
echo "[SETUP]   sbatch 02_train_stage1.sbatch"
echo "[SETUP]   sbatch 02_train_stage2.sbatch  (stage1 完成后)"
echo "============================================"
```

提交方式：

```bash
mkdir -p logs
sbatch 01_setup.sbatch
```

查看日志：`tail -f logs/setup-*.out`

---

### 第二步：提交 Stage1 训练脚本 (`02_train_stage1.sbatch`)

将以下内容保存为 `02_train_stage1.sbatch`，确保已设置 `PROJECT_HOME` 和 `HF_TOKEN` 环境变量后提交：

```bash
#!/bin/bash
#SBATCH --job-name=DiffuEraser_Stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train-stage1-%j.out

set -e

# ── 环境变量 ──
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

WORK_DIR="${PROJECT_HOME}/dev"
WEIGHTS="${WORK_DIR}/weights"
DAVIS="${WORK_DIR}/dataset/DAVIS"
NUM_GPUS=${1:-1}

# ── 激活环境 ──
source ~/.bashrc
conda activate diffueraser

cd "$WORK_DIR"

# ── 运行训练 ──
validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"

accelerate launch \
  --multi_gpu \
  --num_processes $NUM_GPUS \
  --mixed_precision bf16 \
  train_DiffuEraser_stage1.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1_path="${WEIGHTS}/diffuEraser" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --davis_root="${WORK_DIR}/dataset/DAVIS" \
  --ytvos_root="${WORK_DIR}/dataset/YTBV" \
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

echo "[TRAIN] Stage 1 完成"

# ── 转换权重 ──
LATEST_CKPT=$(ls -d finetune-stage1/checkpoint-* | sort -V | tail -n 1)
CKPT_NAME=$(basename "$LATEST_CKPT")

cp save_checkpoint_stage1.py save_checkpoint_stage1_temp.py
sed -i "s|checkpoint-xxxx|$CKPT_NAME|g" save_checkpoint_stage1_temp.py
python save_checkpoint_stage1_temp.py
rm save_checkpoint_stage1_temp.py

echo "[TRAIN] Stage 1 权重转换完成: converted_weights/finetuned-stage1/"
```

提交方式（`01_setup.sbatch` 成功完成后）：

```bash
sbatch 02_train_stage1.sbatch                            # 默认 1 卡测试
sbatch --gres=gpu:8 02_train_stage1.sbatch 8             # 8 卡正式训练
```

查看日志：`tail -f logs/train-stage1-*.out`

---

### 第三步：提交 Stage2 训练脚本 (`02_train_stage2.sbatch`)

**等待 Stage1 完成后**，将以下内容保存为 `02_train_stage2.sbatch` 并提交：

```bash
#!/bin/bash
#SBATCH --job-name=DiffuEraser_Stage2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train-stage2-%j.out

set -e

# ── 环境变量 ──
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

WORK_DIR="${PROJECT_HOME}/dev"
WEIGHTS="${WORK_DIR}/weights"
DAVIS="${WORK_DIR}/dataset/DAVIS"
NUM_GPUS=${1:-1}

# ── 激活环境 ──
source ~/.bashrc
conda activate diffueraser

cd "$WORK_DIR"

# ── 运行训练 ──
FINETUNED_STAGE1="${WORK_DIR}/converted_weights/finetuned-stage1"
validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"

accelerate launch \
  --multi_gpu \
  --num_processes $NUM_GPUS \
  --mixed_precision fp16 \
  train_DiffuEraser_stage2.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1="${FINETUNED_STAGE1}" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --motion_adapter_path="${WEIGHTS}/animatediff-motion-adapter-v1-5-2" \
  --davis_root="${WORK_DIR}/dataset/DAVIS" \
  --ytvos_root="${WORK_DIR}/dataset/YTBV" \
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

echo "[TRAIN] Stage 2 完成"

# ── 转换权重 ──
LATEST_CKPT_S2=$(ls -d finetune-stage2/checkpoint-* | sort -V | tail -n 1)
CKPT_NAME_S2=$(basename "$LATEST_CKPT_S2")

cp save_checkpoint_stage2.py save_checkpoint_stage2_temp.py
sed -i "s|checkpoint-xxxx|$CKPT_NAME_S2|g" save_checkpoint_stage2_temp.py
python save_checkpoint_stage2_temp.py
rm save_checkpoint_stage2_temp.py

echo "[TRAIN] Stage 2 权重转换完成: converted_weights/finetuned-stage2/"
```

提交方式（`02_train_stage1.sbatch` 成功完成后）：

```bash
sbatch 02_train_stage2.sbatch                            # 默认 1 卡测试
sbatch --gres=gpu:4 02_train_stage2.sbatch 8             # 8 卡正式训练
```

查看日志：`tail -f logs/train-stage2-*.out`

---

## 🔧 训练参数修改指引

如果需要调整训练参数，**在对应的 stage 脚本中直接修改**：

| 参数 | 含义 | 默认值 | 位置 |
|------|------|--------|------|
| `NUM_GPUS` | GPU 数量 | `1` (命令行传参覆盖) | 脚本第一个参数 |
| `--max_train_steps` | 最大训练步数 | `50000` | `accelerate launch` 命令 |
| `--learning_rate` | 学习率 | `5e-06` | `accelerate launch` 命令 |
| `--nframes` | 每次采样帧数 | Stage1: `10`, Stage2: `22` | `accelerate launch` 命令 |
| `--resolution` | 训练分辨率 | `512` | `accelerate launch` 命令 |
| `--train_batch_size` | 批大小 | `1` | `accelerate launch` 命令 |
| `--checkpointing_steps` | 保存间隔 | `2000` | `accelerate launch` 命令 |
| `--validation_steps` | 验证间隔 | `2000` | `accelerate launch` 命令 |
| `--gradient_checkpointing` | 梯度检查点 (省显存) | 仅 Stage1 | `accelerate launch` 命令 |

> [!TIP]
> **显存不足 (CUDA OOM)?** → 添加 `--gradient_checkpointing` 到 Stage 2，或减小 `--nframes` / `--resolution`

> [!TIP]
> **SLURM 时间限制**：默认 `--time=48:00:00`。如训练需要更久，可修改脚本头部的 `#SBATCH --time`

> [!IMPORTANT]
> **GPU Partition 名称**：脚本中默认使用 `--partition=gpu`。如果你的集群 GPU partition 名不同（如 `pgpu`、`v100` 等），请修改 `#SBATCH --partition=` 行。

---

## 📋 附录

<details>
<summary>附录 A：完成后的目录结构</summary>

```
$PROJECT_HOME/
├── DiffuEraser_downloads/                  ← 下载缓存（搭建完成后可删除）
│   ├── finetune-code/
│   └── finetune-weights/
│
├── dev/                                    ← 工作目录
│   ├── train_DiffuEraser_stage1.py
│   ├── train_DiffuEraser_stage2.py
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
└── DPO-dataset/                            ← 🔮 DPO 训练数据 (后续使用)
    └── ...
```

</details>

<details>
<summary>附录 B：DPO 数据集结构说明</summary>

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
<summary>附录 C：常见问题</summary>

**Q1: CUDA OOM 怎么办？**
在对应 stage 脚本中添加或确认 `--gradient_checkpointing`，或减小 `--nframes` / `--resolution`

**Q2: 下载缓存可以删除吗？**
搭建完成后即可安全删除 `$PROJECT_HOME/DiffuEraser_downloads`，节省磁盘空间。

**Q3: 训练中断了怎么办？**
脚本已启用 `--resume_from_checkpoint="latest"`，直接重新 `sbatch` 对应的 stage 脚本即可从最后的 checkpoint 恢复。

**Q4: GPU partition 名不是 `gpu` 怎么办？**
修改对应 stage 脚本中的 `#SBATCH --partition=你的partition名`

</details>
