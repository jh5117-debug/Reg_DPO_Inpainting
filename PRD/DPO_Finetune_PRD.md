# DiffuEraser DPO Finetune 需求文档

> **目标**：在已完成的 SFT 全量微调基础上，引入 VideoDPO 的 Direct Preference Optimization 机制，进一步优化 DiffuEraser 视频修复质量。
> **版本**：v2.0 | **最后更新**：2026-03-27

---

## 1. 背景

### 1.1 DiffuEraser 模型架构

DiffuEraser 是一个基于 **Stable Diffusion 1.5 + BrushNet** 的视频修复 (Video Inpainting) 模型，核心组件包括：

| 组件 | 参数量 | 作用 |
|------|--------|------|
| UNet2DConditionModel | ~860M | 空间去噪主干 |
| BrushNet | ~886M | 条件编码器（masked image + mask → multi-scale features） |
| MotionModule | ~120M (嵌入 UNetMotionModel) | 时序一致性（temporal attention） |
| VAE | ~84M | 像素空间 ↔ latent 空间 |
| CLIP Text Encoder | ~123M | 文本条件 |

训练采用**两阶段策略**：
- **Stage 1**：训练 UNet2D + BrushNet（空间质量）
- **Stage 2**：冻结 UNet2D + BrushNet，仅训练 MotionModule（时序一致性）

### 1.2 SFT 全量微调（已完成）

我们已在 **YouTube-VOS + DAVIS** 数据集上完成了 SFT 全量微调：
- Stage 1: 30,000 步 → Stage 2: 34,000 步
- 最终权重：`${PROJECT_ROOT}/finetune-stage2/converted_weights_step34000`
- 详细经验总结见：[First_Finetuning_Summary.md](./First_Finetuning_Summary.md)

### 1.3 为什么需要 DPO

SFT 本质上是"模仿学习"——模型学习重建 GT，但无法学习"什么样的修复是好的、什么是差的"。DPO (Direct Preference Optimization) 引入**偏好对**（好修复 vs 差修复），让模型直接从对比信号中学习偏好，无需人工标注或单独训练 reward model。

参考实现：[VideoDPO](https://github.com/...) (`/home/hj/VideoDPO`)

### 1.4 相关设计文档

| 文档 | 内容 |
|------|------|
| `VideoDPO_to_DiffuEraser_Report.md` | 从 VideoDPO 迁移到 DiffuEraser 的技术方案 |
| `DPO_Feasibility_Review.md` | DPO 在视频修复方向的可行性分析 |
| `DPO_Dataset_Generation.md` | DPO 偏好对数据集生成方案 |
| `First_Finetuning_Summary.md` | SFT 阶段的工程经验沉淀 |
| `DPO_code_review.md` | DPO 代码的完整 Code Review 报告 |
| `DPO_Project_Complete_Summary.md` | 项目全流程 + 所有 Bug 记录 |

---

## 2. 训练策略

### Stage 1 — UNet2D + BrushNet DPO

| 项目 | 值 |
|------|-----|
| Policy | UNet2D + BrushNet (可训练, ~1.75B) |
| Reference | 同结构冻结副本 (SFT 权重) |
| 冻结 | VAE, Text Encoder, Ref UNet, Ref BrushNet |
| 数据 | DPO 偏好对 (GT + neg_frames_1/2) |
| nframes | 16 |
| Loss | Diffusion-DPO loss, β=2500 |
| 验证指标 | PSNR + SSIM |
| 最大步数 | 21,000 |
| 验证频率 | 每 500 步 |

### Stage 2 — MotionModule DPO

| 项目 | 值 |
|------|-----|
| Policy | MotionModule only (可训练, ~120M) |
| Reference | 完整 UNetMotionModel + BrushNet (SFT 权重, 冻结) |
| 冻结 | VAE, Text Encoder, UNet2D, BrushNet |
| 验证指标 | PSNR + SSIM + Ewarp + TC (CLIP-ViT-H-b14) |
| 最大步数 | 30,000 |

---

## 3. 关键设计决策

1. **BrushNet 条件统一**：pos/neg 共享 GT masked image 作为 BrushNet 条件，防止信息泄漏——否则模型学的是"条件图长什么样"而非"修复质量好坏"
2. **DAVIS 10x 过采样**：平衡 DAVIS (~30 视频) 与 YouTube-VOS (~3400 视频) 的样本量差异
3. **Chunk-Aligned 采样**：DPO 负样本按 16 帧 chunk 生成，采样必须对齐 chunk 边界，避免跨缝合线 artifact 污染 loss 信号
4. **nframes=16**：对齐 DPO 数据集生成时的 chunk 大小和评分粒度
5. **权重保存策略**：每个 stage 只保存 best + last 两个权重，节省 W&B 存储
6. **Sigma Term 监控**：`sigmoid(inside_term)` 接近 1.0 → DPO loss 饱和 → 降低 β 或 LR
7. **DGR 梯度监控**：`grad_norm / initial_grad_norm < 0.01` → 梯度消散 → 训练退化

---

## 4. DPO 数据集

```
${PROJECT_ROOT}/data/DPO_Finetune_data/   (HF: JiaHuang01/DPO_Finetune_Data, 69.9GB)
├── manifest.json              (2066 entries, 每个 entry 可能有 neg_1 + neg_2)
├── davis_bear/                (~30 DAVIS 视频, 10x 过采样后在 DataLoader 中出现 ~1200 次)
│   ├── gt_frames/             原始 GT 帧 (512×512 PNG)
│   ├── masks/                 二值 mask
│   ├── neg_frames_1/          退化负样本 1 (chimera_chunked)
│   ├── neg_frames_2/          退化负样本 2
│   └── meta.json              chunk 边界标注
└── ytbv_*/                    (~2000 YouTube-VOS 视频, 同结构)
```

总计 5212 条训练样本 (davis=1200 过采样后, ytbv=4012, neg_1=2606, neg_2=2606)

---

## 5. 显存优化

DPO 的显存消耗约为 SFT 的 **4 倍**（4 套模型 + batch 翻倍 + 4 次 forward），在 8×A100-80GB 上仍需精细管理：

| 优化项 | 描述 | 效果 |
|--------|------|------|
| xformers | 4 个模型全部启用 memory efficient attention | 显著降低 attention 显存 |
| gradient_checkpointing | Policy 模型启用 | 用计算换显存 |
| 去掉 step-1 validation | 避免训练态 + pipeline 显存叠加 | 降低 early-step OOM 风险 |
| del BrushNet outputs | UNet 消费完后立即 `del down_samples, mid_sample, up_samples` | 有一定帮助（前提：后续不再访问，autograd 无额外引用） |
| del Ref BrushNet outputs | Ref UNet 消费完后立即 `del` | 有一定帮助（前提：ref 在 `no_grad()` 中） |
| del batch pixel values | VAE encode 后 `del batch["pixel_values_*"]` | 小幅帮助（前提：后续不再使用原始 pixel） |
| Ref forward 在 `no_grad()` 中 | 不产生计算图 | 必要，否则 Ref 也会存中间激活 |
| Fallback: nframes=8 | 如果仍 OOM 则降低帧数 | 显存直接减半 |

---

## 6. 路径与环境规范

### 6.1 路径

| 环境 | 路径 |
|------|------|
| 集群项目根 | `/sc-projects/sc-proj-cc09-repair/hongyou/dev/Reg_DPO_Inpainting/` |
| 本地开发 | `/home/hj/Reg_DPO_Inpainting/` |
| SFT 最终权重 | `${PROJECT_ROOT}/finetune-stage2/converted_weights_step34000` |
| DPO 数据集 | `${PROJECT_ROOT}/data/DPO_Finetune_data/` |
| 验证数据集 | `${PROJECT_ROOT}/data_val/JPEGImages_432_240/` + `test_masks/` |
| SD1.5 基础权重 | `${PROJECT_ROOT}/weights/stable-diffusion-v1-5` |
| VAE 权重 | `${PROJECT_ROOT}/weights/sd-vae-ft-mse` |

### 6.2 缓存隔离（禁止写入 home 目录）

所有第三方库缓存必须在项目目录内，通过 SLURM 脚本设置环境变量：
```bash
export WANDB_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_CACHE_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_DATA_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${PROJECT_ROOT}/.wandb_cache/config"
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/.hf_cache"
```

---

## 7. 文件清单

| 文件 | 用途 |
|------|------|
| `DPO_finetune/dataset/dpo_dataset.py` | DPO 偏好对数据集（含 manifest fallback + chunk-aligned 采样） |
| `DPO_finetune/train_dpo_stage1.py` | Stage 1 训练脚本（UNet2D + BrushNet） |
| `DPO_finetune/train_dpo_stage2.py` | Stage 2 训练脚本（MotionModule） |
| `DPO_finetune/scripts/run_dpo_stage1.py` | Stage 1 Python 启动入口 |
| `DPO_finetune/scripts/run_dpo_stage2.py` | Stage 2 Python 启动入口 |
| `DPO_finetune/scripts/03_dpo_stage1.sbatch` | Stage 1 SLURM 脚本 |
| `DPO_finetune/scripts/03_dpo_stage2.sbatch` | Stage 2 SLURM 脚本 |
| `inference/metrics.py` | 评估指标（含 TC: TemporalConsistencyMetric 待添加） |

---

## 8. WandB 监控

### 8.1 初始化策略
- WandB `init_trackers` 放在 `main()` **最前面**（模型加载之前）
- `__main__` 入口用全局 `try-except` 包裹，崩溃通过 `wandb.alert()` 输出到 Dashboard

### 8.2 每步监控指标（14 项）

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `dpo_loss` | DPO 损失 | 应持续下降 |
| `implicit_acc` | 模型区分 pos/neg 的准确率 | 0.5~0.9 |
| `win_gap` | Policy 在 pos 上偏离 Ref 的程度 | < 0 (neg) |
| `lose_gap` | Policy 在 neg 上偏离 Ref 的程度 | > 0 (pos) |
| `reward_margin` | 偏好 margin | < 0 (neg) |
| `sigma_term` | sigmoid 饱和度 | 0.5~0.9（接近 1.0 → 饱和） |
| `kl_divergence` | Policy vs Ref 的 KL 散度 | 小值 |
| `mse_w` / `mse_l` | Policy 的 pos/neg MSE | ↓ / stable |
| `ref_mse_w` / `ref_mse_l` | Ref 的 pos/neg MSE | baseline |
| `dgr_grad_norm` | 梯度范数比值 | > 0 (alive) |
| `grad_norm_ratio` | 当前/初始梯度比 | > 0.01 |
| `lr` | 学习率 | 1e-6 |

---

## 9. 已解决的 Bug 记录

| # | 严重性 | 描述 | 修复 |
|---|--------|------|------|
| 1 | P0 | `UNetMotionModel` 未 import | 添加 import |
| 2 | P0 | manifest key ≠ 目录名 → 0 entries | `_load_manifest` fallback |
| 3 | P0 | Stage 1 timesteps `(2,)` vs noisy `(32,)` → expand 崩溃 | `repeat_interleave(nframes)` |
| 4 | P0 | Stage 2 BrushNet vs UNetMotionModel 需要不同 timesteps 维度 | 拆分 `timesteps_all_2d` / `timesteps_all_motion` |
| 5 | P0 | OOM (A100-80GB) | 4 项显存优化（del outputs + 去 step-1 val） |
| 6 | P1 | Stage 2 权重加载 `hasattr` 缺失 | 双重 `hasattr` 保护 |
| 7 | P1 | Stage 2 `encoder_hidden_states` DPO concat 后未翻倍 | `.repeat(2, 1, 1)` |
| 8 | P1 | WandB 初始化过晚 → 崩溃时 Dashboard 空白 | 提前到模型加载前 |
| 9 | P1 | WandB artifact staging 撑爆 home 配额 | 环境变量重定向到项目目录 |
| 10 | P2 | DDP 多进程函数属性不安全 | 改为普通局部变量 |
| 11 | P2 | `accelerate launch` mixed_precision 参数冲突 | 删除冗余参数 |

详细的逐条 debug 记录见：[DPO_Project_Complete_Summary.md](./DPO_Project_Complete_Summary.md)

---

## 10. 验证计划

| 阶段 | 方法 |
|------|------|
| Smoke test | `--max_train_steps 2`，确认前向无 crash |
| 小规模 | `--max_train_steps 100`，确认 loss 下降 + 指标合理 |
| 全量 Stage 1 | 21,000 步，监控 `dgr_grad_norm` 和 `sigma_term` |
| 全量 Stage 2 | 30,000 步，额外监控 Ewarp + TC |

---

## 11. 集群运行命令

```bash
cd ${PROJECT_HOME}/dev/Reg_DPO_Inpainting
git pull origin main

# Stage 1
NUM_GPUS=8 MAX_STEPS=21000 VAL_STEPS=500 sbatch DPO_finetune/scripts/03_dpo_stage1.sbatch

# Stage 2 (Stage 1 完成后)
NUM_GPUS=8 MAX_STEPS=30000 VAL_STEPS=500 sbatch DPO_finetune/scripts/03_dpo_stage2.sbatch
```
