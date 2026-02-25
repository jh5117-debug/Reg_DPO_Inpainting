# VBench 视频质量评估集成

## 概述

在 `run_OR.py` 推理完成后，使用 VBench 框架对 ProPainter 和 DiffuEraser 输出视频进行多维度质量评估，并输出对比结果。

## 新增 CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--eval_vbench` | flag | False | 启用 VBench 评估 |
| `--vbench_dimensions` | nargs+ | 8 个默认维度 | 自定义评估维度 |
| `--vbench_output` | str | `vbench_results.json` | 结果输出文件名 |

## 默认 8 个评估维度

| 维度 | 说明 | 模型 |
|------|------|------|
| `subject_consistency` | 主体一致性 | DINO ViT-B/16 |
| `background_consistency` | 背景一致性 | CLIP ViT-B/32 |
| `temporal_flickering` | 时序闪烁 | MAE (无模型) |
| `motion_smoothness` | 运动平滑度 | AMT-S |
| `aesthetic_quality` | 美学质量 | LAION Aesthetic + CLIP ViT-L/14 |
| `imaging_quality` | 成像质量 | MUSIQ-SPAQ |
| `dynamic_degree` | 动态程度 | RAFT |
| `overall_consistency` | 整体一致性 | ViCLIP (需 `--use_text`) |

## 使用示例

```bash
CUDA_VISIBLE_DEVICES=0 python run_OR.py \
  --dataset davis \
  --video_root /path/to/DAVIS/JPEGImages/Full-Resolution \
  --mask_root  /path/to/DAVIS/Annotations/Full-Resolution \
  --save_path  results_OR \
  --eval_vbench \
  --vbench_dimensions subject_consistency temporal_flickering aesthetic_quality \
  ... (其他参数)
```

## 输出格式

- **终端**：打印 ProPainter vs DiffuEraser 对比表格，含每维度分数和 Δ 差异
- **JSON**：保存到 `{save_path}/vbench_results.json`，含 per-video 和 average 分数

## 依赖

- VBench 源码：`/home/hj/VBench`（通过 `sys.path` 引入）
- `decord`：视频解码库（已安装）
- 预训练模型：首次运行时自动下载到 `~/.cache/vbench/`
