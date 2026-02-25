# OR Pipeline 精简 + 文本引导嵌入

## 变更概要

| 文件 | 行数变化 | 变更内容 |
|------|---------|---------|
| `diffueraser/diffueraser_OR.py` | 648→593 | 删除 anchor 注入(~55行)，加 `negative_prompt` |
| `run_OR.py` | 1304→580 | 删除 metrics(~200行) + anchor(~120行)，重构 prompt 加载 |

## 详细变更

### `diffueraser_OR.py`
- `forward()` 删除 `anchor_frame`, `anchor_frame_idx` 参数
- 新增 `validation_n_prompt` 解析
- 两处 `self.pipeline()` 调用加 `negative_prompt=validation_n_prompt`
- 删除整个 Anchor Frame Injection 块及 anchor strip 逻辑

### `run_OR.py`

**删除:**
- 全部 metrics 代码：`MetricsCalculator` 导入、`eval_one_video()`、`fmt()`、I3D 累积、VFID 计算、metrics 统计表
- 全部 anchor 代码：`anchor_inpainter` 导入/加载/调用/cleanup、`--use_anchor`/`--anchor_*` 参数
- `--gt_root`、`--gt_video` 参数

**新增:**
- `--unified_prompt_yaml`: 统一 YAML prompt 文件路径
- `--prompt_root`: per-video YAML 目录
- Per-video prompt 加载函数 `_resolve_prompt_for_video()`, 优先级: CLI → unified YAML → per-video YAML → 空

**保持不变:**
- mask 膨胀策略 (`mask_dilation_iter=4`)
- compositing (GaussianBlur blended=True)
- pre-inference 采样策略
- 对比视频生成
- ProPainter 推理逻辑

## 用法示例

```bash
CUDA_VISIBLE_DEVICES=0 python run_OR.py \
  --dataset davis \
  --video_root /path/to/JPEGImages/Full-Resolution \
  --mask_root  /path/to/Annotations/Full-Resolution \
  --save_path  results_OR \
  --video_length 60 \
  --ref_stride 6 --neighbor_length 25 --subvideo_length 80 \
  --mask_dilation_iter 4 \
  --save_comparison \
  --height 360 --width 640 \
  --use_text \
  --unified_prompt_yaml prompt_cache/all_captions.yaml \
  --text_guidance_scale 3.5
```

## 验证

- [x] `py_compile` 语法检查通过
- [x] `--help` 不含 `--no_metrics`、`--use_anchor`、`--anchor_*`
- [x] `--help` 包含 `--use_text`、`--unified_prompt_yaml`、`--prompt_root`、`--text_guidance_scale`
