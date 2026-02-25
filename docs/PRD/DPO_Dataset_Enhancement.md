# DPO 数据集：视频级三路对决评分策略

需求文档 — 2026-02-20 v2

## 策略

**正样本 = GT，负样本 = 每个视频跑 3 种管线 → 评分 → 取最差**

| 类型 | 管线 | 伪影 |
|------|------|------|
| blur | 仅 ProPainter | 天然模糊 |
| hallucination | 仅 DiffuEraser (全黑 dummy priori) | 幻觉 |
| flicker | DiffuEraser 分段多 seed (dummy priori) | 时序闪烁 |

评分：mask 区域 SSIM - 0.5 × temporal_flicker → 最低分胜出为负样本。

## 输出

```
dpo_data/
├── {video_name}/
│   ├── gt_frames/
│   ├── masks/
│   ├── neg_frames/    ← 三路中最差
│   ├── meta.json      ← neg_type + 三路分数
│   └── comparison.mp4
└── manifest.json
```

## 启动

```bash
bash run_dpo_dataset.sh
```
