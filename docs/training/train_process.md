# DiffuEraser Finetune 完整训练流程

> 项目目录：`/home/hj/Train_Diffueraser/`
> 权重目录：`/home/hj/DiffuEraser_new/weights/`

---

## 一、训练概述

DiffuEraser 采用两阶段训练策略：

| | Stage 1 | Stage 2 |
|---|---------|---------|
| **目标** | 学习空间 inpainting 能力 | 学习时序一致性 |
| **模型** | UNet2D + BrushNet | UNetMotion (UNet2D + MotionAdapter) + BrushNet |
| **训练参数** | UNet2D + BrushNet **全部参数** | **仅 MotionAdapter temporal layers** |
| **冻结** | 无 | UNet2D 空间层 + BrushNet |
| **帧数** | 10 | 22 |
| **数据** | DAVIS + YouTubeVOS 干净帧 + 随机 mask | 同左 |
| **显存估算** | ~20-24GB (batch=1, fp16) | ~30-40GB (batch=1, fp16) |

---

## 二、数据准备

### 2.1 数据集结构

```
/home/hj/Train_Diffueraser/dataset/
├── DAVIS/
│   ├── JPEGImages/480p/           ← 训练帧（干净GT）
│   │   ├── bear/
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   ├── boat/
│   │   └── ...
│   ├── Annotations/480p/          ← 仅用于 validation（不参与训练）
│   └── ImageSets/2017/train.txt   ← 训练视频名列表
└── YTBV/
    └── JPEGImages/                ← 训练帧（干净GT）
        ├── 003234408d/
        ├── 0043f083b5/
        └── ...
```

### 2.2 验证数据完整性

```bash
cd /home/hj/Train_Diffueraser

# 检查 DAVIS
ls dataset/DAVIS/JPEGImages/480p/ | head -5
cat dataset/DAVIS/ImageSets/2017/train.txt | wc -l

# 检查 YTBV
ls dataset/YTBV/JPEGImages/ | wc -l

# 预期输出：
# DAVIS: ~60 个视频目录
# YTBV: ~3400+ 个视频目录
```

### 2.3 Mask 生成原理

训练时 **不使用任何标注（Annotations）**，mask 完全随机合成：

1. 调用 `dataset/utils.py` 中的 `create_random_shape_with_random_motion()`
2. 生成随机形状的笔刷 stroke mask
3. 50% 概率固定 mask，50% 概率 mask 在帧间有随机运动
4. mask 覆盖面积约为图像面积的 1/3 ~ 全图
5. 每次 `__getitem__` 重新随机生成

### 2.4 数据不均衡处理

- YTBV: ~3400+ 个视频（直接使用）
- DAVIS: ~60 个视频（10x oversampling → ~600 条目，占比约 15%）

---

## 三、权重文件

### 3.1 权重路径

```
/home/hj/DiffuEraser_new/weights/
├── stable-diffusion-v1-5/                     ← SD1.5 base model
├── sd-vae-ft-mse/                             ← VAE
├── diffuEraser/                               ← DiffuEraser 原始预训练权重
│   ├── brushnet/                              ← BrushNet 权重
│   │   ├── config.json
│   │   └── diffusion_pytorch_model.safetensors
│   └── unet_main/                             ← UNet2D 权重
│       ├── config.json
│       └── diffusion_pytorch_model.safetensors
└── animatediff-motion-adapter-v1-5-2/         ← AnimateDiff MotionAdapter（Stage 2 用）
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

### 3.2 权重用途

| 权重 | Stage 1 | Stage 2 |
|------|---------|---------|
| `stable-diffusion-v1-5` | base model（加载 tokenizer, text_encoder, scheduler） | 同左 |
| `sd-vae-ft-mse` | VAE encoder/decoder | 同左 |
| `diffuEraser/unet_main` | ✅ 初始化 UNet2D（通过 `--pretrained_stage1_path`） | ❌ 不直接用，用 finetuned-stage1 |
| `diffuEraser/brushnet` | ✅ 初始化 BrushNet（通过 `--pretrained_stage1_path`） | ❌ 不直接用，用 finetuned-stage1 |
| `animatediff-motion-adapter-v1-5-2` | ❌ 不用 | ✅ 初始化 MotionAdapter |

---

## 四、Stage 1 训练

### 4.1 训练内容

- **输入**: 10 帧连续图像 + 随机 mask + "clean background" prompt
- **训练对象**: UNet2D 和 BrushNet 的全部参数
- **目标**: 学习空间维度的 inpainting 能力（每帧独立处理）

### 4.2 启动训练

```bash
cd /home/hj/Train_Diffueraser

# 直接运行（日志输出到 finetune-stage1.log）
bash finetune_stage1.sh

# 或者前台运行（可以看到实时输出）
bash finetune_stage1.sh &
tail -f finetune-stage1.log
```

### 4.3 启动脚本关键参数说明

```bash
accelerate launch --mixed_precision "fp16" \
  train_DiffuEraser_stage1.py \
  --base_model_name_or_path="..."           # SD1.5 base model
  --pretrained_stage1_path="..."            # DiffuEraser 预训练权重（含 unet_main/ 和 brushnet/）
  --vae_path="..."                          # VAE 权重
  --davis_root="..."                        # DAVIS 数据集根目录
  --ytvos_root="..."                        # YTBV 数据集根目录
  --resolution=512                          # 输入分辨率
  --nframes=10                              # 每个样本的帧数
  --train_batch_size=1                      # batch size（受显存限制）
  --learning_rate=5e-06                     # 学习率（finetune 用 5e-6，原始训练用 1e-5）
  --checkpointing_steps=2000               # 每 2000 步保存 checkpoint
  --validation_steps=2000                   # 每 2000 步运行验证
  --resume_from_checkpoint="latest"         # 从最新 checkpoint 恢复（首次运行会跳过）
  --output_dir="finetune-stage1"            # 训练输出目录
```

### 4.4 监控训练

```bash
# 查看日志
tail -f finetune-stage1.log

# 查看 loss 趋势
grep "loss" finetune-stage1.log | tail -20

# 查看保存的 checkpoint
ls finetune-stage1/

# 查看验证 GIF
ls finetune-stage1/logs-finetune-stage1/samples/
```

### 4.5 训练完成后：转换 checkpoint

训练输出的是 Accelerator 格式的 checkpoint，需要转换为 HuggingFace 格式才能用于推理或 Stage 2：

```bash
cd /home/hj/Train_Diffueraser

# 1. 查看可用的 checkpoint
ls finetune-stage1/
# 输出类似：checkpoint-2000  checkpoint-4000  checkpoint-6000 ...

# 2. 编辑 save_checkpoint_stage1.py，修改 input_dir 中的步数
#    例如选择 checkpoint-6000:
#    input_dir = "/home/hj/Train_Diffueraser/finetune-stage1/checkpoint-6000"

# 3. 运行转换
python save_checkpoint_stage1.py

# 4. 验证输出
ls converted_weights/finetuned-stage1/
# 应该看到：unet_main/  brushnet/
```

**转换后的权重路径：**
```
/home/hj/Train_Diffueraser/converted_weights/finetuned-stage1/
├── unet_main/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── brushnet/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

---

## 五、Stage 2 训练

### 5.1 前提条件

- ✅ Stage 1 训练完成
- ✅ Stage 1 checkpoint 已转换（`converted_weights/finetuned-stage1/` 存在）
- ✅ `animatediff-motion-adapter-v1-5-2` 权重已下载

### 5.2 训练内容

- **输入**: 22 帧连续图像 + 随机 mask + "clean background" prompt
- **训练对象**: 仅 MotionAdapter 的 temporal attention layers
- **冻结**: UNet2D 空间层 + BrushNet
- **目标**: 在 Stage 1 学到的空间能力基础上，学习时序一致性

### 5.3 启动训练

```bash
cd /home/hj/Train_Diffueraser

# 确认 Stage 1 转换后的权重存在
ls converted_weights/finetuned-stage1/unet_main/
ls converted_weights/finetuned-stage1/brushnet/

# 启动 Stage 2
bash finetune_stage2.sh

# 监控
tail -f finetune-stage2.log
```

### 5.4 启动脚本关键参数说明

```bash
accelerate launch --mixed_precision "fp16" \
  train_DiffuEraser_stage2.py \
  --base_model_name_or_path="..."           # SD1.5 base model
  --pretrained_stage1="..."                 # ← Stage 1 finetune 转换后的权重
  --vae_path="..."                          # VAE 权重
  --motion_adapter_path="..."              # AnimateDiff MotionAdapter
  --davis_root="..."                        # DAVIS 数据集根目录
  --ytvos_root="..."                        # YTBV 数据集根目录
  --resolution=512
  --nframes=22                              # ← Stage 2 用更多帧
  --train_batch_size=1
  --learning_rate=5e-06
  --checkpointing_steps=2000
  --validation_steps=2000
  --resume_from_checkpoint="latest"
  --output_dir="finetune-stage2"
```

### 5.5 训练完成后：转换 checkpoint

```bash
cd /home/hj/Train_Diffueraser

# 1. 查看可用的 checkpoint
ls finetune-stage2/

# 2. 编辑 save_checkpoint_stage2.py，修改 input_dir 中的步数

# 3. 运行转换
python save_checkpoint_stage2.py

# 4. 验证输出
ls converted_weights/finetuned-stage2/
# 应该看到：unet_main/  brushnet/
```

---

## 六、最终产出

训练完成后，`converted_weights/` 目录结构如下：

```
/home/hj/Train_Diffueraser/converted_weights/
├── finetuned-stage1/
│   ├── unet_main/              ← UNet2DConditionModel（finetuned）
│   └── brushnet/               ← BrushNetModel（finetuned）
└── finetuned-stage2/
    ├── unet_main/              ← UNetMotionModel（含 finetuned temporal layers）
    └── brushnet/               ← BrushNetModel（来自 Stage 1，Stage 2 冻结未变）
```

将这些权重用于推理时，替换 `/home/hj/DiffuEraser_new/weights/diffuEraser/` 中对应的文件即可。

---

## 七、常见操作

### 7.1 从 checkpoint 恢复训练

如果训练中断，只要 `--resume_from_checkpoint="latest"` 已设置（默认已设），重新运行 `bash finetune_stageX.sh` 即会自动从最新 checkpoint 恢复。

### 7.2 调整超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `learning_rate` | 5e-6 | finetune 建议比原始训练 (1e-5) 小一半 |
| `train_batch_size` | 1 | 受显存限制，可以配合 `gradient_accumulation_steps` 增大等效 batch |
| `nframes` | 10 (S1) / 22 (S2) | 与原始训练一致 |
| `checkpointing_steps` | 2000 | 根据训练速度调整 |
| `max_train_steps` | 不设 (默认跑 epoch) | 可设为 10000-20000 先看效果 |

### 7.3 多 GPU 训练

```bash
# 修改 shell 脚本中的 accelerate launch 命令：
accelerate launch --num_processes=2 --mixed_precision "fp16" \
  train_DiffuEraser_stageX.py \
  ...
```

### 7.4 限制训练步数

在 shell 脚本中添加 `--max_train_steps=10000`：

```bash
accelerate launch --mixed_precision "fp16" \
  train_DiffuEraser_stage1.py \
  ...
  --max_train_steps=10000 \
  ...
```

---

## 八、完整命令速查

```bash
cd /home/hj/Train_Diffueraser

# ===== Stage 1 =====
bash finetune_stage1.sh                      # 启动训练
tail -f finetune-stage1.log                  # 监控日志
# 修改 save_checkpoint_stage1.py 中的 checkpoint 路径
python save_checkpoint_stage1.py             # 转换权重

# ===== Stage 2 =====
bash finetune_stage2.sh                      # 启动训练
tail -f finetune-stage2.log                  # 监控日志
# 修改 save_checkpoint_stage2.py 中的 checkpoint 路径
python save_checkpoint_stage2.py             # 转换权重
```
