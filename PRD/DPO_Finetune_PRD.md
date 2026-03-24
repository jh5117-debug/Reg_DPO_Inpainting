# DiffuEraser DPO Finetune 需求文档

## 1. 背景

在已有 YTVOS+DAVIS SFT Finetune 基础上，引入 VideoDPO 的 Direct Preference Optimization (DPO) 机制进一步优化 DiffuEraser 的 video inpainting 质量。利用预生成的偏好对数据集 (GT vs 退化样本) 训练模型学习 "好修复 vs 差修复" 的偏好，无需人工标注。

## 2. 训练策略

### Stage 1 — UNet2D + BrushNet DPO
| 项目 | 值 |
|------|-----|
| Policy | UNet2D + BrushNet (可训练) |
| Reference | 同结构冻结副本 (SFT 权重) |
| 冻结 | VAE, text_encoder |
| 数据 | DPO 偏好对 (GT + neg_frames_1/2) |
| nframes | 16 |
| Loss | Diffusion-DPO loss, beta=2500 |
| 指标 | PSNR + SSIM |

### Stage 2 — MotionModule DPO
| 项目 | 值 |
|------|-----|
| Policy | MotionModule (可训练) |
| Reference | 完整 UNetMotionModel + BrushNet (SFT 权重, 冻结) |
| 冻结 | VAE, text_encoder, UNet2D, BrushNet |
| 指标 | PSNR + SSIM + Ewarp + TC |

## 3. 关键设计决策

1. **BrushNet 条件统一**：pos/neg 共享 GT masked image 作为 BrushNet 条件，防止信息泄漏
2. **DAVIS 10x 过采样**：平衡 DAVIS (~30 视频) 与 YTVOS (~3400 视频) 的样本量差异
3. **nframes=16**：对齐 DPO 数据集生成时的 chunk 大小和评分粒度
4. **权重保存策略**：每个 stage 只保存 best + last 两个权重，节省 W&B 存储

## 4. 文件清单

| 文件 | 用途 |
|------|------|
| `DPO_finetune/dataset/dpo_dataset.py` | DPO 偏好对数据集 |
| `DPO_finetune/train_dpo_stage1.py` | Stage 1 训练脚本 |
| `DPO_finetune/train_dpo_stage2.py` | Stage 2 训练脚本 |
| `DPO_finetune/scripts/run_dpo_stage1.py` | Stage 1 启动入口 |
| `DPO_finetune/scripts/run_dpo_stage2.py` | Stage 2 启动入口 |
| `DPO_finetune/scripts/03_dpo_stage1.sbatch` | Stage 1 SLURM 脚本 |
| `DPO_finetune/scripts/03_dpo_stage2.sbatch` | Stage 2 SLURM 脚本 |
| `inference/metrics.py` (修改) | 新增 TemporalConsistencyMetric |

## 5. 验证计划

1. Smoke test: `--max_train_steps 2`, 确认前向无 crash
2. 小规模: `--max_train_steps 100`, 确认 loss 下降
3. 全量: 使用 SLURM 提交
