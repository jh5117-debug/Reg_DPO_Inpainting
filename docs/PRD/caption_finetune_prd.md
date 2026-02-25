# DiffuEraser 带 Prompt/Caption 的 Finetune 管道

## 需求背景

当前 DiffuEraser finetune 代码使用硬编码的 `"clean background"` 作为 text prompt (CFG=0)，不利用文本引导。原始 DiffuEraser 训练代码通过 CSV 文件传入真实的场景描述（caption），从而让模型学习利用文本条件。

## 目标

为 DAVIS + YTVOS 训练数据集生成真实的 caption，并在 finetune 训练中使用。

## 实现方案

### 1. Caption 生成
- **DAVIS**：使用现有 `generate_captions_BR.py`（Qwen2.5-VL 模型，`qwen_env` 环境）
- **YTVOS**：新建 `generate_captions_ytvos.py` 适配其目录结构
- **合并**：`merge_captions.py` 将两个数据集的 YAML 合并为统一文件

### 2. 数据集类
- `FinetuneDatasetWithCaption`：从 YAML 加载 caption，无 caption 时 fallback `"clean background"`

### 3. 训练脚本
- 基于原有 finetune 代码，新增 `--caption_yaml` 参数
- 替换 Dataset 为带 caption 的版本

### 4. 项目结构
所有文件在 `~/Train_Diffueraser_prompt/`，通过符号链接复用共享资源。

## 文件清单

| 文件 | 说明 |
|------|------|
| `generate_captions_ytvos.py` | YTVOS caption 生成脚本 |
| `merge_captions.py` | YAML 合并工具 |
| `dataset/finetune_dataset_caption.py` | 带 caption 的 Dataset 类 |
| `train_DiffuEraser_stage1.py` | Stage1 训练（带 caption） |
| `train_DiffuEraser_stage2.py` | Stage2 训练（带 caption） |
| `finetune_stage1.sh` | Stage1 启动脚本 |
| `finetune_stage2.sh` | Stage2 启动脚本 |
| `save_checkpoint_stage1.py` | Stage1 权重转换 |
| `save_checkpoint_stage2.py` | Stage2 权重转换 |
| `run_finetune_all.sbatch` | SLURM 全流程作业 |
