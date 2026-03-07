# 12 实验体系 — 需求文档

## 需求背景

DiffuEraser 模型使用 PCM (Phased Consistency Model) 进行加速推理，
需要系统性地对比三种 PCM checkpoint 配置在不同数据集和处理模式下的表现：

| ckpt 名 | 权重类型 | 步数 | 内建 GS |
|---------|---------|------|---------|
| `2-Step` | smallcfg | 2 | 0.0 |
| `4-Step` | smallcfg | 4 | 0.0 |
| `Normal CFG 4-Step` | normalcfg | 4 | 7.5 |

## 实验矩阵

3 ckpt × 2 数据集(OR/BR) × 2 处理模式(noblend/blend+dil8) = **12 实验**

每个实验内部包含 Baseline(无prompt) vs Text-Guided(有prompt) 对比。

## 技术方案

### 代码修改
- `compare_all.py`: 添加 `--ckpt` CLI 参数（消除硬编码）

### 新增文件
- `run_12exp_all.sh`: 2波×6GPU并行执行脚本
- `generate_report.py`: Markdown报告生成器

### GPU 调度
- 6张 RTX 3090 (GPU 0,1,2,3,5,6)
- Wave 1: 6个noblend实验并行
- Wave 2: 6个blend+dil8实验并行

### guidance_scale 规则
- 4-Step → text_guidance_scale = 7.5
- 2-Step → text_guidance_scale = 3.0
- Baseline 始终 guidance_scale = 1.0

## 输出规范
- 目录命名: `{cfg_type}_{steps}_step_{dataset}_{blend}_{dilation}_gs{guidance_scale}`
- 最终报告: `experiment_report.md`
