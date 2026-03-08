# Stage2 冒烟测试环境搭建 PRD

## 需求背景

在本地 `/home/hj/DiffuEraser_Project` 对 Stage2 训练脚本进行冒烟测试，验证训练链路完整性。

## 环境

- GPU: 7 × RTX 3090 (24GB)，单卡测试
- Conda: 复用现有 `diffueraser` 环境
- 数据: DAVIS (60) + YTBV (3467) = 4067 videos

## 实现方案

### 创建的文件

1. `training/baseline/smoke_test_stage2.sh` — 冒烟测试脚本
2. `diffueraser/__init__.py` — 包初始化
3. `libs/__init__.py` — 包初始化

### 修改的文件

1. `training/baseline/train_DiffuEraser_stage2.py` — 添加 `SKIP_VALIDATION` 环境变量支持

### 关键设计决策

| 决策 | 原因 |
|------|------|
| 用原始 DiffuEraser 权重替代 finetuned-stage1 | 本地未完成 Stage1 训练，结构兼容 |
| 创建 patched_stage1 目录（SD-1.5 config + diffuEraser 权重符号链接） | 解决 `UNetMotionModel` vs `UNet2DConditionModel` config 不兼容 |
| `SKIP_VALIDATION` 环境变量 | 避免 validation pipeline 与训练模型同时占用显存导致 OOM |
| `nframes=6` | 降低显存峰值 |

## 验证结果

**通过 ✅** — 3 steps 完成，exit code 0
