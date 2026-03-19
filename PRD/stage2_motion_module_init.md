# Stage 2 Motion Module 初始化策略修改

## 变更概述

修改 Stage 2 训练代码的 Motion Module 初始化来源：从**原始 AnimateDiff** 改为从 **DiffuEraser baseline** 继承已训练的 motion module 权重。

## 变更原因

原始流程中 Stage 2 从原始 AnimateDiff 的 motion module 开始训练，导致 motion module 需要从零学习 video inpainting 的时序模式。DiffuEraser baseline 的 `unet_main` 是完整的 `UNetMotionModel`，其中的 motion module 已由原作者在大规模数据上训练完成。

## 修改文件

| 文件 | 变更内容 |
|------|----------|
| `train_DiffuEraser_stage2.py` | 新增 `--baseline_unet_path` 参数；核心权重加载逻辑改为从 baseline 加载 UNetMotionModel + Stage 1 2D 覆盖 |
| `scripts/run_train_stage2.py` | `--motion_adapter_path` 替换为 `--baseline_unet_path`，默认指向 `weights/diffuEraser/Orign_Diffueraser` |
| `scripts/02_train_stage2.sbatch` | 新增 `OUTPUT_DIR` 环境变量 |

## 使用方式

集群上启动训练（与之前相同的命令）：
```bash
PRETRAINED_STAGE1=/sc-projects/sc-proj-cc09-repair/hongyou/dev/Reg_DPO_Inpainting/finetune-stage1/converted_weights_step18000 \
sbatch scripts/02_train_stage2.sbatch
```

`baseline_unet_path` 默认自动指向 `weights/diffuEraser/Orign_Diffueraser`，无需额外配置。

## 注意事项

- 需要先清除 `finetune-stage2/` 目录下的旧 checkpoint（否则 `--resume_from_checkpoint latest` 会从旧 checkpoint 恢复，不会使用新的 motion module 初始化）
- Stage 1 无需修改
