# DiffuEraser Project

> **统一整理版** — 基于 DiffuEraser 的视频补全框架，包含推理、训练、DPO 优化全流程代码与文档。

---

## 📁 目录导航

| 目录 | 内容 | 说明 |
|------|------|------|
| `inference/` | 推理脚本 | `run_OR.py`(物体移除+VBench), `run_BR.py`(背景恢复), caption 生成 |
| `diffueraser/` | 核心模块 | DiffuEraser 模型、pipeline、指标计算 |
| `libs/` | 模型组件 | BrushNet, UNet, MotionAdapter 等自定义模块 |
| `propainter/` → | ProPainter | 光流传播模型 (symlink) |
| `training/baseline/` | 无prompt微调 | Stage1/2 finetune (caption="clean background") |
| `training/prompt/` | 有prompt微调 | Stage1/2 finetune (VLM生成的真实场景描述) |
| `training/dpo/` | DPO数据集 | 负样本生成、可视化对比 |
| `training/dataset/` | Dataset模块 | FinetuneDataset, DPODataset, mask工具 |
| `captions/` | Caption数据 | 3564条YAML (90 DAVIS + 3471 YTVOS) |
| `evaluation/VBench/` → | 视频评估 | VBench 视频质量评估工具 (symlink) |
| `data/` | 数据集 | DAVIS, YTBV, dpo_data (全部symlink) |
| `weights/` → | 模型权重 | SD1.5, VAE, diffuEraser, ProPainter, PCM等 (symlink) |
| `docs/` | 全部文档 | 设计文档、训练文档、DPO研究、PRD |
| `reference/` → | 原始代码 | DiffuEraser 官方仓库原始代码 (symlink) |
| `results/` | 推理输出 | 推理结果存放目录 |

## 🚀 快速开始

### OR (物体移除) 推理
```bash
cd inference
CUDA_VISIBLE_DEVICES=0 python run_OR.py \
  --dataset davis \
  --video_root ../data/DAVIS_FullRes/DAVIS/JPEGImages/Full-Resolution \
  --mask_root ../data/DAVIS_FullRes/DAVIS/Annotations/Full-Resolution \
  --save_path ../results/OR_baseline \
  --base_model_path ../weights/stable-diffusion-v1-5 \
  --vae_path ../weights/sd-vae-ft-mse \
  --diffueraser_path ../weights/diffuEraser \
  --propainter_model_dir ../weights/propainter \
  --pcm_weights_path ../weights/PCM_Weights \
  --height 360 --width 720 \
  --save_comparison
```

### BR (背景恢复) + 指标评估
```bash
cd inference
CUDA_VISIBLE_DEVICES=0 python run_BR.py \
  --dataset davis \
  --video_root ../data/davis_BR/JPEGImages_432_240/ \
  --mask_root ../data/davis_BR/test_masks/ \
  --gt_root ../data/davis_BR/JPEGImages_432_240/ \
  --save_path ../results/BR_baseline \
  --compute_metrics --save_comparison \
  --base_model_path ../weights/stable-diffusion-v1-5 \
  --vae_path ../weights/sd-vae-ft-mse \
  --diffueraser_path ../weights/diffuEraser \
  --propainter_model_dir ../weights/propainter \
  --pcm_weights_path ../weights/PCM_Weights \
  --i3d_model_path ../weights/i3d_rgb_imagenet.pt \
  --raft_model_path ../weights/propainter/raft-things.pth
```

### Finetune 训练
详见 `docs/training/train_process.md`

### DPO 研究
详见 `docs/dpo/Region-Reg-DPO_完整数学推导.md`

---

## 📋 文件来源

本项目整理自以下 6 个目录（原目录未修改）：

| 源目录 | 角色 |
|--------|------|
| `/home/hj/DiffuEraser_new` | 推理主站 + 权重 + 数据集 |
| `/home/hj/Diffueraser_test` | 推理测试 + VBench 集成 |
| `/home/hj/Train_Diffueraser` | 训练主站(无prompt) + DPO |
| `/home/hj/Train_Diffueraser_prompt` | 训练主站(有prompt) + Captions |
| `/home/hj/DPO如何融入` | DPO 研究文档 |
| `/home/hj/VBench` | 视频评估工具 |
