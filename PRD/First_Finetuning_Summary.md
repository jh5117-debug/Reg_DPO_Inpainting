# DiffuEraser 首次全量微调需求文档

> **微调方案**：基于 YouTube-VOS + DAVIS 数据集，对 DiffuEraser 进行两阶段 (Stage 1 + Stage 2) 全量微调。

---

## 1. 训练监控与日志体系

### 1.1 日志方案选型

审查训练脚本的日志输出机制，确认当前采用的是纯文本日志、TensorBoard 还是 Weights & Biases (W&B) Dashboard。

### 1.2 W&B 集成方案

| 步骤 | 说明 |
|------|------|
| 创建 Organization | 在 W&B 平台创建 Organization (OR) |
| 创建 Project | 在 OR 下创建训练项目 |
| 添加合作者 | 将合作者账号加入 OR，共享 Dashboard 访问权限 |
| 训练脚本配置 | 在训练代码中集成 W&B，写入 API Key 等配置 |

**W&B Dashboard 应监控的指标**：
- Training Loss Curve
- Learning Rate
- 验证指标：PSNR、SSIM（每次 validation 步输出所有验证视频的**平均值**）

---

## 2. 脚本架构与职责分离

### 2.1 SLURM 脚本定位

SLURM 脚本**仅作为 Python 脚本的启动器**，不承载任何业务逻辑（不进行权重转换、路径拼接等操作）。所有训练、转换、评估逻辑均封装在 Python 文件中。

### 2.2 两组启动脚本

| 模式 | 说明 |
|------|------|
| **联合训练** | Stage 1 → Stage 2 一键串行执行（Stage 1 训练完成后仍保存中间权重） |
| **独立训练** | Stage 1 和 Stage 2 可分别独立提交运行 |

---

## 3. 路径与环境变量规范

### 3.1 项目根目录

所有代码文件直接平铺部署于：

```
/sc-projects/sc-proj-cc09-repair/hongyou/dev/Reg_DPO_Inpainting/
```

合作者通过 `git clone` 至 `/sc-projects/sc-proj-cc09-repair/hongyou/dev/` 目录下，自动生成上述路径。

项目根目录下不允许嵌套多余的中间文件夹。

### 3.2 环境变量

| 变量名 | 用途 |
|--------|------|
| `$HF_TOKEN` | Hugging Face API Token |
| `$WANDB_API_KEY` | Weights & Biases API Key |
| `$PROJECT_HOME` | 项目根目录路径 |

所有脚本应通过上述环境变量加载路径，**禁止硬编码绝对路径**。

---

## 4. 权重保存与转换

### 4.1 训练结束后的权重转换

每个 Stage 训练结束后，在训练脚本内部**自动衔接**权重转换流程：

1. 完成 `accelerator.save_state()` 保存 Accelerator 格式 checkpoint；
2. 紧接着将 checkpoint 转换为标准 Diffusers 格式（`unet_main/config.json` + `model.safetensors`）。

转换逻辑内聚于训练脚本自身，**不拆分为独立的转换脚本文件**。

### 4.2 定期 Checkpoint 转换与 W&B 上传

- 每次 `checkpointing_steps` 保存 checkpoint 时，若权重转换耗时在可接受范围内，**同步执行转换**（而非仅在最后一步转换）。
- 转换后的权重上传至 W&B Artifact，便于从 W&B 网站直接下载任意 checkpoint 的可用权重。

---

## 5. Bug 修复

### 5.1 Stage 1 保存逻辑错误

**问题**：Stage 1 脚本末尾调用了 `unet_main.save_motion_modules()`，但 Stage 1 的 `unet_main` 是 `UNet2DConditionModel`，不包含 Motion Modules。

**修复**：Stage 1 仅保存 `UNet2DConditionModel` 和 `BrushNet` 的权重，`save_motion_modules()` 调用仅适用于 Stage 2。

---

## 6. 多卡训练与步数配置

### 6.1 max_train_steps 语义

`max_train_steps` 表示**每张 GPU 的最大优化步数**，而非多卡的累计总步数。

> 例：`max_train_steps=26000`，8 卡并行时，每张卡各训练 26000 步。

### 6.2 当前配置

| 参数 | 值 |
|------|-----|
| `max_train_steps` | 26000（每卡） |
| `validation_steps` | 2000（每 2000 步执行一次 validation，用于监控过拟合） |

---

## 7. 验证数据集规范

### 7.1 数据源

验证集采用 **ProPainter 预处理的 432×240 分辨率版本**，配合特定的 mask 数据。

验证集**独立于训练数据集**，严禁使用训练数据进行验证。

### 7.2 数据目录结构

```
data_val/
├── JPEGImages_432_240/   # 视频帧 (432×240)
│   ├── bear/
│   ├── boat/
│   └── ...               # 共 50 个视频
└── test_masks/           # 对应的分割 mask
    ├── bear/
    ├── boat/
    └── ...
```

### 7.3 验证流程

训练脚本**自动扫描** `data_val/JPEGImages_432_240/` 下的所有视频子目录，对每个视频执行推理并计算 PSNR/SSIM，最终在 W&B 中仅输出所有视频的**平均 PSNR** 和**平均 SSIM** 两条指标曲线，不生成逐视频的可视化。
