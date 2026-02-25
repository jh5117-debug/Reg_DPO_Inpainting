# DiffuEraser Finetune 合作者完整操作指南 (Full Archive + Dependencies)

> **极简流程**：你一键打包上传 -> 合作者下载 -> 运行 `setup_project.sh` 一键还原 -> `pip install -r requirements.txt` -> 开始训练。

---

## 〇、文件结构

所有内容被打包为 5 个主要文件上传到 HuggingFace：

1. **`code_base.tar.gz`**：包含所有训练代码、脚本、libs、diffueraser、dataset模块，以及 **`requirements.txt`** 和 **`environment.yml`**。
2. **`DAVIS.tar`**：DAVIS 数据集。
3. **`YTBV.tar`**：YouTubeVOS 数据集。
4. **`weights_extras.tar`**：DiffuEraser、VAE、AnimateDiff 权重。
5. **`stable-diffusion-v1-5/`**：保持原样（未打包），因为太大。

---

## 一、你这边：一键打包上传

```bash
cd /home/hj/Train_Diffueraser
bash upload_to_hf.sh
```

脚本会自动：
1. **导出当前环境依赖** (`requirements.txt` 和 `environment.yml`)。
2. 打包代码 (`code_base.tar.gz`)。
3. 打包数据集 (`DAVIS.tar`, `YTBV.tar`)。
4. 打包权重 (`weights_extras.tar`)。
5. 上传所有包。

---

## 二、合作者这边：下载与还原

### 2.1 环境下载

```bash
pip install -U huggingface_hub
huggingface-cli login
```

### 2.2 下载所有文件

```bash
mkdir -p ~/DiffuEraser_finetune
cd ~/DiffuEraser_finetune

# 1. 下载代码包、数据集包、设置脚本
huggingface-cli download JiaHuang01/DiffuEraser-finetune-code \
    --repo-type dataset \
    --local-dir ./

# 2. 下载权重包
# 注意：weights_extras.tar 会被下载到 weights/ 目录下
mkdir -p weights
huggingface-cli download JiaHuang01/DiffuEraser-finetune-weights \
    --repo-type dataset \
    --local-dir weights/
```

### 2.3 🚀 一键还原项目结构

运行 `setup_project.sh`，它会自动解压所有压缩包并归位：

```bash
cd ~/DiffuEraser_finetune
bash setup_project.sh
```

如果看到 **`🎉 Setup Complete!`**，说明代码和数据已就绪。

### 2.4 安装 Python 依赖 (Updated)

解压后，你会在根目录看到 `requirements.txt` 和 `environment.yml`。

**方法 A：通用安装 (推荐)**
适用于大多数 Linux 服务器 (CUDA 11.8/12.1)。

```bash
conda create -n diffueraser python=3.10 -y
conda activate diffueraser

# 安装依赖
pip install -r requirements.txt
```

**方法 B：精确复刻 (如果方法 A 失败)**
这将完全复制原作者的 Conda 环境。

```bash
conda env create -f environment.yml
conda activate diffueraser
```

### 2.5 配置 Accelerate

```bash
accelerate config
```

---

## 三、合作者这边：运行训练

### 3.1 修改路径

使用 `sed` 一键替换脚本中的路径，改为合作者本地路径：

```bash
cd ~/DiffuEraser_finetune

PROJECT_DIR=$(pwd)
WEIGHTS_DIR="${PROJECT_DIR}/weights"

for f in finetune_stage1.sh finetune_stage2.sh run_finetune_all.sbatch save_checkpoint_stage1.py save_checkpoint_stage2.py; do
    sed -i "s|/home/hj/Train_Diffueraser|${PROJECT_DIR}|g" $f
    sed -i "s|/home/hj/DiffuEraser_new/weights|${WEIGHTS_DIR}|g" $f
done
```

### 3.2 运行训练

**推荐：使用 SLURM**

```bash
mkdir -p logs converted_weights
sbatch run_finetune_all.sbatch
```
