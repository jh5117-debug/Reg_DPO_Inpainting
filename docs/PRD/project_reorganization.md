# DiffuEraser 项目文件统一整理方案

## 背景

项目在迭代过程中产生了 6 个分散的目录，功能重叠、文件重复、版本混乱。本方案将所有有价值的文件整理到一个统一目录 `/home/hj/DiffuEraser_Project` 中，不修改任何现有文件。

---

## 源目录分析

| 源目录 | 大小 | 角色 | 包含内容 |
|--------|------|------|---------|
| `DiffuEraser_new` | 94G | **推理主站** | 推理代码(run_OR/BR)、text-guided 设计文档、权重(30G+)、DAVIS Full-Res 测试集、ProPainter、prompt_cache |
| `Diffueraser_test` | 3.7G | **推理+VBench测试** | = DiffuEraser_new + VBench 集成的 `run_OR.py` 增强版，更多 PRD 文档 |
| `Train_Diffueraser` | 93G | **训练主站(无prompt)** | finetune 代码(stage1/2)、DPO 数据集生成、训练数据集(DAVIS+YTBV)、dpo_data、ProPainter 副本 |
| `Train_Diffueraser_prompt` | 7.9M | **训练主站(有prompt)** | prompt 版 finetune 脚本、FinetuneDatasetWithCaption、3564 条 captions、HF 上传脚本(符号链接到 Train_Diffueraser) |
| `DPO如何融入` | 16M | **DPO 研究文档** | Region-Reg-DPO 数学推导+实施路线图+可行性报告(PDF+MD+LaTeX) |
| `VBench` | 346M | **视频评估工具** | VBench 第三方 clone (git repo) |

### 关键差异分析

| 文件 | DiffuEraser_new | Diffueraser_test | 结论 |
|------|:---:|:---:|------|
| `run_OR.py` | 32K (基础版) | 42K (**+VBench 集成**) | ✅ 用 Diffueraser_test 版 |
| `run_BR.py` | 29K | 29K (完全相同) | 任一均可 |
| `metrics.py` | 32K | 32K (完全相同) | 任一均可 |
| `generate_captions_*.py` | 各 18K/28K | 完全相同 | 任一均可 |
| `diffueraser_OR.py` | 26K + `.bak` 备份 | 26K (**干净版**) | ✅ 用 Diffueraser_test 版 |
| `diffueraser.py` | 25K | 25K (空行差异) | ✅ 用 Diffueraser_test 版 |
| `libs/*` | 完全相同 × 3 目录 | — | 只保留一份 |

### 需要丢弃的文件

| 文件/目录 | 来源 | 原因 |
|-----------|------|------|
| `diffueraser_OR.py.bak` | DiffuEraser_new | 旧版备份 |
| `diffueraser_OR.py.bak_visual_*` | DiffuEraser_new | 旧版备份 |
| `diffueraser copy/` | Train_Diffueraser | 备份目录，内容与 DiffuEraser_new 一致 |
| `__pycache__/` | 所有目录 | 运行时缓存 |
| `.cache/` | 3个目录 | 运行时缓存 |
| `=4.49` | DiffuEraser_new | pip 安装残留 |
| `tian.txt` | DiffuEraser_new | 空文件 |
| `patch_mask_morph.py` | diffueraser/ | v2 已存在，v1 冗余 |

---

## 新目录结构

```
/home/hj/DiffuEraser_Project/
│
├── README.md                              # 项目总览导航文档
│
├── inference/                             # ======== 推理代码 ========
│   ├── run_OR.py                          ← Diffueraser_test (VBench集成版)
│   ├── run_BR.py                          ← DiffuEraser_new
│   ├── metrics.py                         ← DiffuEraser_new
│   ├── generate_captions_BR.py            ← DiffuEraser_new
│   ├── generate_captions_OR.py            ← DiffuEraser_new
│   ├── configs/
│   │   └── prompt_template.yaml           ← DiffuEraser_new
│   ├── prompt_cache/
│   │   ├── all_captions_BR.yaml           ← DiffuEraser_new
│   │   └── all_captions_OR.yaml           ← DiffuEraser_new
│   └── scripts/
│       └── instruction.txt                ← Diffueraser_test (运行命令速查)
│
├── diffueraser/                           # ======== DiffuEraser 核心模块 ========
│   ├── diffueraser.py                     ← Diffueraser_test (最新)
│   ├── diffueraser_OR.py                  ← Diffueraser_test (干净版)
│   ├── diffueraser_OR_DPO.py              ← DiffuEraser_new (DPO推理版)
│   ├── metrics.py                         ← Diffueraser_test
│   ├── patch_mask_morph_v2.py             ← Diffueraser_test (最新版)
│   ├── pipeline_diffueraser.py            ← DiffuEraser_new
│   ├── pipeline_diffueraser_stage1.py     ← DiffuEraser_new
│   └── pipeline_diffueraser_stage2.py     ← DiffuEraser_new
│
├── libs/                                  # ======== 共享模型库 ========
│   ├── brushnet_CA.py                     ← DiffuEraser_new
│   ├── transformer_temporal.py
│   ├── unet_2d_blocks.py
│   ├── unet_2d_condition.py
│   ├── unet_3d_blocks.py
│   └── unet_motion_model.py
│
├── propainter/ → (symlink)                # ======== ProPainter ========
│                                          ← 符号链接到 DiffuEraser_new/propainter
│
├── training/                              # ======== 训练代码 ========
│   ├── baseline/                          # --- 无 prompt 版 ---
│   │   ├── train_DiffuEraser_stage1.py    ← Train_Diffueraser
│   │   ├── train_DiffuEraser_stage2.py    ← Train_Diffueraser
│   │   ├── finetune_stage1.sh             ← Train_Diffueraser
│   │   ├── finetune_stage2.sh             ← Train_Diffueraser
│   │   ├── save_checkpoint_stage1.py      ← Train_Diffueraser
│   │   ├── save_checkpoint_stage2.py      ← Train_Diffueraser
│   │   ├── run_finetune_all.sbatch        ← Train_Diffueraser
│   │   ├── environment.yml                ← Train_Diffueraser
│   │   ├── requirements.txt               ← Train_Diffueraser
│   │   └── score_inpainting_quality.py    ← Train_Diffueraser
│   │
│   ├── prompt/                            # --- 有 prompt 版 ---
│   │   ├── train_DiffuEraser_stage1.py    ← Train_Diffueraser_prompt
│   │   ├── train_DiffuEraser_stage2.py    ← Train_Diffueraser_prompt
│   │   ├── finetune_stage1.sh             ← Train_Diffueraser_prompt
│   │   ├── finetune_stage2.sh             ← Train_Diffueraser_prompt
│   │   ├── save_checkpoint_stage1.py      ← Train_Diffueraser_prompt
│   │   ├── save_checkpoint_stage2.py      ← Train_Diffueraser_prompt
│   │   ├── run_finetune_all.sbatch        ← Train_Diffueraser_prompt
│   │   ├── generate_captions_ytvos.py     ← Train_Diffueraser_prompt
│   │   ├── merge_captions.py              ← Train_Diffueraser_prompt
│   │   ├── setup_project_prompt.sh        ← Train_Diffueraser_prompt
│   │   └── upload_to_hf.sh               ← Train_Diffueraser_prompt
│   │
│   ├── dpo/                               # --- DPO 数据集生成 ---
│   │   ├── generate_dpo_negatives.py      ← Train_Diffueraser
│   │   ├── run_dpo_dataset.sh             ← Train_Diffueraser
│   │   ├── visualize_comparison.py        ← Train_Diffueraser
│   │   └── train_stage1_memprofile.py     ← Train_Diffueraser
│   │
│   └── dataset/                           # --- 训练用 dataset 模块 ---
│       ├── __init__.py                    ← Train_Diffueraser
│       ├── utils.py                       ← Train_Diffueraser
│       ├── finetune_dataset.py            ← Train_Diffueraser
│       ├── finetune_dataset_caption.py    ← Train_Diffueraser_prompt
│       ├── dpo_dataset.py                 ← Train_Diffueraser
│       ├── region_mask_utils.py           ← Train_Diffueraser
│       ├── file_client.py                 ← Train_Diffueraser
│       └── img_util.py                    ← Train_Diffueraser
│
├── captions/                              # ======== Caption 数据 ========
│   └── (3564 yaml files)                  ← Train_Diffueraser_prompt/captions
│
├── evaluation/                            # ======== 评估工具 ========
│   └── VBench/ → (symlink)               ← 符号链接到 /home/hj/VBench
│
├── data/                                  # ======== 数据集 (符号链接) ========
│   ├── DAVIS_FullRes/ → (symlink)         ← DiffuEraser_new/DAVIS-2017-trainval-Full-Resolution
│   ├── DAVIS_480p/ → (symlink)            ← Train_Diffueraser/dataset/DAVIS
│   ├── YTBV/ → (symlink)                 ← Train_Diffueraser/dataset/YTBV
│   ├── davis_BR/ → (symlink)             ← DiffuEraser_new/dataset/davis
│   └── dpo_data/ → (symlink)             ← Train_Diffueraser/dpo_data
│
├── weights/ → (symlink)                   # ======== 模型权重 (符号链接) ========
│                                          ← 符号链接到 DiffuEraser_new/weights
│
├── docs/                                  # ======== 文档中心 ========
│   ├── design/                            # 设计文档
│   │   ├── text_guided_pipeline.txt       ← DiffuEraser_new/readme.txt
│   │   ├── 创新思路.txt                    ← DiffuEraser_new/创新.txt
│   │   └── 新思路_BR_vs_OR_分析.txt        ← DiffuEraser_new/新思路.txt
│   │
│   ├── training/                          # 训练文档
│   │   ├── train_idea.md                  ← Train_Diffueraser (完整需求文档)
│   │   ├── train_process.md               ← Train_Diffueraser (训练流程)
│   │   ├── instruction_prompt.md          ← Train_Diffueraser_prompt (合作者指南)
│   │   ├── instruction_non_prompt.txt     ← Train_Diffueraser_prompt
│   │   ├── instruction_combined.txt       ← Train_Diffueraser_prompt
│   │   ├── prepare_data.sh               ← Train_Diffueraser
│   │   └── setup_project.sh              ← Train_Diffueraser
│   │
│   ├── dpo/                               # DPO 研究文档
│   │   ├── Region-Reg-DPO_完整数学推导.md  ← DPO如何融入
│   │   ├── Region-Reg-DPO_实施路线图.md    ← DPO如何融入
│   │   ├── DPO_Dataset_Generation.md      ← DPO如何融入
│   │   ├── Region_Reg_DPO_Full_Report.tex ← DPO如何融入
│   │   ├── DiffuEraser_RegDPO融合可行性分析报告.pdf
│   │   ├── Reg-DPO_深度解析_完整推导.pdf
│   │   ├── Reg-DPO_compressed.pdf
│   │   ├── DiffuEraser.pdf
│   │   └── Diffueraser的目前来说的痛点.pdf
│   │
│   └── PRD/                               # PRD 合集（去重）
│       ├── Anchor_Insertion_Analysis.md
│       ├── FeatureAggregator_Integration.md
│       ├── fix_forward_signature_and_unified_yaml.md
│       ├── or_caption_generation.md
│       ├── or_pipeline_cleanup.md
│       ├── vbench_integration.md
│       ├── DiffuEraser_RegDPO_Report.md
│       ├── DiffuEraser_RegDPO_Report.tex
│       ├── Region-Reg-DPO_实施路线图.md
│       ├── DPO_Dataset_Enhancement.md
│       ├── DPO_Dataset_Generation.md
│       ├── Region-Reg-DPO_creat_dataset.md
│       ├── collaborator_guide.md
│       └── caption_finetune_prd.md
│
├── reference/                             # ======== 原始参考代码 ========
│   └── DiffuEraser_orign/ → (symlink)    ← DiffuEraser_new/DiffuEraser_orign
│
└── results/                               # ======== 推理结果输出目录 ========
    └── (空, 推理时生成)
```

---

## 关键设计决策

### 1. 大文件用符号链接

| 目录 | 原因 |
|------|------|
| `weights/` | ~30G 模型权重，拷贝浪费磁盘 |
| `data/DAVIS_FullRes/` | ~25G 完整分辨率视频 |
| `data/DAVIS_480p/` | ~3G 训练用480p帧 |
| `data/YTBV/` | ~60G YouTubeVOS 训练数据 |
| `data/dpo_data/` | ~10G DPO 负样本 |
| `propainter/` | ~200M ProPainter 代码 |
| `evaluation/VBench/` | 346M VBench 整个 repo |
| `reference/DiffuEraser_orign/` | 原始参考代码 |

### 2. 代码文件用物理拷贝
所有 `.py`、`.sh`、`.md`、`.txt`、`.yaml`、`.pdf`、`.tex` 文件用物理拷贝，确保新目录独立可用。

### 3. captions 目录
`Train_Diffueraser_prompt/captions/` 包含 3564 个 YAML 文件（约 800K），物理拷贝。

---

## 验证计划

### 自动验证
```bash
cd /home/hj/DiffuEraser_Project

echo "=== 1. 检查目录结构 ==="
find . -maxdepth 3 -type d | sort | head -50

echo "=== 2. 检查符号链接有效性 ==="
find . -type l -exec test ! -e {} \; -print

echo "=== 3. 检查关键文件存在 ==="
for f in inference/run_OR.py inference/run_BR.py \
         diffueraser/diffueraser_OR.py \
         libs/brushnet_CA.py \
         training/baseline/train_DiffuEraser_stage1.py \
         training/prompt/train_DiffuEraser_stage1.py \
         training/dpo/generate_dpo_negatives.py \
         weights/stable-diffusion-v1-5 \
         data/DAVIS_480p; do
    test -e "$f" && echo "OK: $f" || echo "MISSING: $f"
done

echo "=== 4. 确认无备份/缓存文件 ==="
find . -name "*.bak*" -o -name "__pycache__" -o -name ".cache" | head -10
```

### 手动验证
请在新目录中检查以下内容：
1. `inference/run_OR.py` 包含 VBench 评估代码（搜索 `eval_vbench`）
2. `training/prompt/` 和 `training/baseline/` 的训练脚本能区分开（prompt 版引用 `FinetuneDatasetWithCaption`）
3. `docs/dpo/` 中的 PDF 和 MD 文件完整可读
4. `weights/` 符号链接指向正确的权重目录
