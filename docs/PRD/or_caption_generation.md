# OR Caption 生成脚本 — 需求文档 (PRD)

## 背景与目标

生成式 Video Inpainting 在 Object Removal 场景中存在一个核心痛点：
**模型会根据 mask 的形状"想象"出物体，而非生成正确的背景。**

为解决此问题，需要为 OR 场景设计专门的 text guidance：
- **prompt** 描述 mask 以外的背景 → 引导模型生成正确的背景纹理
- **n_prompt** 描述 mask 以内的物体 → 抑制模型在 mask 区域生成物体

## 核心设计

### 双阶段 VLM 推理

| Stage | 输入 | VLM 任务 | 输出字段 |
|-------|------|----------|----------|
| 1. 背景描述 | 原图 + mask 区域灰色填充 | 描述灰色区域以外的环境 | `prompt` |
| 2. 物体识别 | mask 区域 bbox 裁剪 | 识别 mask 内的物体 | `n_prompt` = 物体描述 + 质量负面词 |

### YAML 格式变更

从平面格式升级为 `{BR: {...}, OR: {...}}` 子键结构：

```yaml
bear:
  BR:
    prompt: ["A brown bear walking in a rocky enclosure..."]
    n_prompt: ["blurry, flickering..."]
  OR:
    prompt: ["A rocky enclosure with stone wall, scattered rocks..."]
    n_prompt: ["brown bear, blurry, flickering..."]
    object_description: "brown bear"
```

`run_OR.py` 优先读 `OR` 子键，`run_BR.py` 优先读 `BR` 子键，两者均向后兼容旧的平面格式。

## 使用方法

```bash
# 批量处理 DAVIS 数据集
CUDA_VISIBLE_DEVICES=1 python generate_captions_OR.py \
    --dataset_root /home/hj/Train_Diffueraser/dataset/DAVIS \
    --model_path /home/hj/DiffuEraser_new/weights/Qwen2.5-VL-7B-Instruct \
    --batch_output_dir prompt_cache \
    --device cuda --force

# 单视频模式
python generate_captions_OR.py \
    --video_path /path/to/DAVIS/JPEGImages/480p/bear \
    --mask_path /path/to/DAVIS/Annotations/480p/bear \
    --output_yaml prompt_cache/bear_OR.yaml \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct
```

## 变更文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `generate_captions_OR.py` | 新建 | OR 专用双阶段 caption 生成脚本 |
| `run_OR.py` | 修改 | `_resolve_prompt_for_video()` 支持 `{OR: ...}` 子键 |
| `run_BR.py` | 修改 | prompt 读取逻辑支持 `{BR: ...}` 子键 |
