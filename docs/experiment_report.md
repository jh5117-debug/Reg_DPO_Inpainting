# 12-Experiment Comparison Report

Generated: 2026-03-04 06:59:21

## Experiment Configuration

| # | Directory | ckpt | GS | Dataset | Blend | Dilation | Videos |
|---|-----------|------|----|---------|-------|----------|--------|
| 1 | `smallcfg_2step_OR_noblend_nodil_gs0.0` | 2-Step | 0.0 | OR | No | 0 | 50 |
| 2 | `smallcfg_2step_OR_noblend_nodil_gs3.0` | 2-Step | 3.0 | OR | No | 0 | 50 |
| 3 | `smallcfg_2step_BR_noblend_nodil_gs0.0` | 2-Step | 0.0 | BR | No | 0 | 50 |
| 4 | `smallcfg_2step_BR_noblend_nodil_gs3.0` | 2-Step | 3.0 | BR | No | 0 | 50 |
| 5 | `smallcfg_4step_OR_noblend_nodil_gs0.0` | 4-Step | 0.0 | OR | No | 0 | 50 |
| 6 | `smallcfg_4step_OR_noblend_nodil_gs7.5` | 4-Step | 7.5 | OR | No | 0 | 50 |
| 7 | `smallcfg_4step_BR_noblend_nodil_gs0.0` | 4-Step | 0.0 | BR | No | 0 | 50 |
| 8 | `smallcfg_4step_BR_noblend_nodil_gs7.5` | 4-Step | 7.5 | BR | No | 0 | 50 |
| 9 | `normalcfg_4step_OR_noblend_nodil_gs7.5` | Normal CFG 4-Step | 7.5 | OR | No | 0 | 50 |
| 10 | `normalcfg_4step_BR_noblend_nodil_gs7.5` | Normal CFG 4-Step | 7.5 | BR | No | 0 | 50 |
| 11 | `smallcfg_2step_OR_blend_dil8_gs0.0` | 2-Step | 0.0 | OR | Yes | 8 | 50 |
| 12 | `smallcfg_2step_OR_blend_dil8_gs3.0` | 2-Step | 3.0 | OR | Yes | 8 | 50 |
| 13 | `smallcfg_2step_BR_blend_dil8_gs0.0` | 2-Step | 0.0 | BR | Yes | 8 | 50 |
| 14 | `smallcfg_2step_BR_blend_dil8_gs3.0` | 2-Step | 3.0 | BR | Yes | 8 | 50 |
| 15 | `smallcfg_4step_OR_blend_dil8_gs0.0` | 4-Step | 0.0 | OR | Yes | 8 | 50 |
| 16 | `smallcfg_4step_OR_blend_dil8_gs7.5` | 4-Step | 7.5 | OR | Yes | 8 | 50 |
| 17 | `smallcfg_4step_BR_blend_dil8_gs0.0` | 4-Step | 0.0 | BR | Yes | 8 | 50 |
| 18 | `smallcfg_4step_BR_blend_dil8_gs7.5` | 4-Step | 7.5 | BR | Yes | 8 | 50 |
| 19 | `normalcfg_4step_OR_blend_dil8_gs7.5` | Normal CFG 4-Step | 7.5 | OR | Yes | 8 | 50 |
| 20 | `normalcfg_4step_BR_blend_dil8_gs7.5` | Normal CFG 4-Step | 7.5 | BR | Yes | 8 | 50 |

---

## Object Removal (OR) Results

### VBench Scores (OR)

| Experiment | Subj_Con↑ | BG_Con↑ | Temp_Flk↑ | Mot_Smo↑ | Aesth_Q↑ | Img_Q↑ | **Avg** |
|---|---|---|---|---|---|---|---|
| smallcfg_4step_blend_gs0.0 | 0.9156 | 0.9226 | 0.9415 | 0.9760 | 0.4301 | 0.5929 | **0.7965** |
| smallcfg_2step_blend_gs3.0 | 0.9137 | 0.9229 | 0.9421 | 0.9761 | 0.4300 | 0.5934 | **0.7964** |
| normalcfg_4step_blend_gs7.5 | 0.9143 | 0.9235 | 0.9420 | 0.9756 | 0.4308 | 0.5915 | **0.7963** |
| smallcfg_2step_blend_gs0.0 | 0.9147 | 0.9243 | 0.9433 | 0.9769 | 0.4288 | 0.5879 | **0.7960** |
| smallcfg_4step_blend_gs7.5 | 0.9111 | 0.9206 | 0.9386 | 0.9733 | 0.4334 | 0.5936 | **0.7951** |
| smallcfg_2step_noblend_gs0.0 | 0.8939 | 0.9150 | 0.9404 | 0.9753 | 0.4412 | 0.6045 | **0.7951** |
| smallcfg_2step_noblend_gs3.0 | 0.8862 | 0.9129 | 0.9387 | 0.9741 | 0.4430 | 0.6121 | **0.7945** |
| smallcfg_4step_noblend_gs0.0 | 0.8890 | 0.9117 | 0.9376 | 0.9736 | 0.4494 | 0.6055 | **0.7945** |
| normalcfg_4step_noblend_gs7.5 | 0.8847 | 0.9095 | 0.9379 | 0.9729 | 0.4384 | 0.6113 | **0.7924** |
| smallcfg_4step_noblend_gs7.5 | 0.8704 | 0.9072 | 0.9298 | 0.9667 | 0.4533 | 0.6040 | **0.7886** |

---

## Background Restoration (BR) Results

### VBench Scores (BR)

| Experiment | Subj_Con↑ | BG_Con↑ | Temp_Flk↑ | Mot_Smo↑ | Aesth_Q↑ | Img_Q↑ | **Avg** |
|---|---|---|---|---|---|---|---|
| smallcfg_4step_noblend_gs0.0 | 0.8973 | 0.9277 | 0.9339 | 0.9764 | 0.4658 | 0.6473 | **0.8081** |
| smallcfg_4step_blend_gs0.0 | 0.8967 | 0.9280 | 0.9338 | 0.9761 | 0.4659 | 0.6475 | **0.8080** |
| smallcfg_2step_noblend_gs3.0 | 0.8969 | 0.9265 | 0.9338 | 0.9763 | 0.4661 | 0.6473 | **0.8078** |
| smallcfg_2step_noblend_gs0.0 | 0.8974 | 0.9275 | 0.9344 | 0.9766 | 0.4652 | 0.6456 | **0.8078** |
| normalcfg_4step_noblend_gs7.5 | 0.8951 | 0.9264 | 0.9333 | 0.9756 | 0.4677 | 0.6482 | **0.8077** |
| smallcfg_2step_blend_gs3.0 | 0.8946 | 0.9267 | 0.9337 | 0.9761 | 0.4649 | 0.6496 | **0.8076** |
| smallcfg_2step_blend_gs0.0 | 0.8963 | 0.9273 | 0.9346 | 0.9766 | 0.4638 | 0.6458 | **0.8074** |
| smallcfg_4step_noblend_gs7.5 | 0.8930 | 0.9223 | 0.9319 | 0.9749 | 0.4684 | 0.6479 | **0.8064** |
| normalcfg_4step_blend_gs7.5 | 0.8924 | 0.9238 | 0.9328 | 0.9751 | 0.4639 | 0.6493 | **0.8062** |
| smallcfg_4step_blend_gs7.5 | 0.8900 | 0.9206 | 0.9313 | 0.9744 | 0.4651 | 0.6496 | **0.8052** |

### Pixel Metrics (BR, GT available)

| Experiment | PSNR↑ | SSIM↑ | LPIPS↓ | Ewarp↓ | AS↑ | IS↑ |
|---|---|---|---|---|---|---|
| smallcfg_2step_noblend_gs0.0 | 31.8125 | 0.9679 | 0.0173 | 8.1149 | 5.1573 | 1.2181 |
| smallcfg_4step_noblend_gs0.0 | 31.1548 | 0.9640 | 0.0168 | 8.5826 | 5.1715 | 1.2211 |
| smallcfg_2step_noblend_gs3.0 | 31.1222 | 0.9648 | 0.0175 | 8.5951 | 5.1871 | 1.2176 |
| smallcfg_2step_blend_gs0.0 | 30.6416 | 0.9596 | 0.0210 | 8.2670 | 5.1668 | 1.2259 |
| normalcfg_4step_noblend_gs7.5 | 30.3798 | 0.9587 | 0.0192 | 9.2580 | 5.1868 | 1.2217 |
| smallcfg_2step_blend_gs3.0 | 29.6244 | 0.9522 | 0.0224 | 8.8260 | 5.1879 | 1.2320 |
| smallcfg_4step_blend_gs0.0 | 29.6079 | 0.9521 | 0.0214 | 8.8896 | 5.1953 | 1.2190 |
| smallcfg_4step_noblend_gs7.5 | 28.9248 | 0.9484 | 0.0230 | 10.6625 | 5.2138 | 1.2322 |
| normalcfg_4step_blend_gs7.5 | 28.7556 | 0.9448 | 0.0256 | 10.2257 | 5.1915 | 1.2493 |
| smallcfg_4step_blend_gs7.5 | 27.1095 | 0.9320 | 0.0302 | 11.5975 | 5.1644 | 1.2400 |

---

## Cross-Experiment Comparison (VBench Average)

Average VBench score across all videos:

| Experiment | Dataset | Blend | Dil | Steps | GS | Subj_Con↑ | BG_Con↑ | Temp_Flk↑ | Mot_Smo↑ | Aesth_Q↑ | Img_Q↑ | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smallcfg_4step | BR | No | 0 | 4step | 0.0 | 0.8973 | 0.9277 | 0.9339 | 0.9764 | 0.4658 | 0.6473 | **0.8081** |
| smallcfg_4step | BR | Yes | 8 | 4step | 0.0 | 0.8967 | 0.9280 | 0.9338 | 0.9761 | 0.4659 | 0.6475 | **0.8080** |
| smallcfg_2step | BR | No | 0 | 2step | 3.0 | 0.8969 | 0.9265 | 0.9338 | 0.9763 | 0.4661 | 0.6473 | **0.8078** |
| smallcfg_2step | BR | No | 0 | 2step | 0.0 | 0.8974 | 0.9275 | 0.9344 | 0.9766 | 0.4652 | 0.6456 | **0.8078** |
| normalcfg_4step | BR | No | 0 | 4step | 7.5 | 0.8951 | 0.9264 | 0.9333 | 0.9756 | 0.4677 | 0.6482 | **0.8077** |
| smallcfg_2step | BR | Yes | 8 | 2step | 3.0 | 0.8946 | 0.9267 | 0.9337 | 0.9761 | 0.4649 | 0.6496 | **0.8076** |
| smallcfg_2step | BR | Yes | 8 | 2step | 0.0 | 0.8963 | 0.9273 | 0.9346 | 0.9766 | 0.4638 | 0.6458 | **0.8074** |
| smallcfg_4step | BR | No | 0 | 4step | 7.5 | 0.8930 | 0.9223 | 0.9319 | 0.9749 | 0.4684 | 0.6479 | **0.8064** |
| normalcfg_4step | BR | Yes | 8 | 4step | 7.5 | 0.8924 | 0.9238 | 0.9328 | 0.9751 | 0.4639 | 0.6493 | **0.8062** |
| smallcfg_4step | BR | Yes | 8 | 4step | 7.5 | 0.8900 | 0.9206 | 0.9313 | 0.9744 | 0.4651 | 0.6496 | **0.8052** |
| smallcfg_4step | OR | Yes | 8 | 4step | 0.0 | 0.9156 | 0.9226 | 0.9415 | 0.9760 | 0.4301 | 0.5929 | **0.7965** |
| smallcfg_2step | OR | Yes | 8 | 2step | 3.0 | 0.9137 | 0.9229 | 0.9421 | 0.9761 | 0.4300 | 0.5934 | **0.7964** |
| normalcfg_4step | OR | Yes | 8 | 4step | 7.5 | 0.9143 | 0.9235 | 0.9420 | 0.9756 | 0.4308 | 0.5915 | **0.7963** |
| smallcfg_2step | OR | Yes | 8 | 2step | 0.0 | 0.9147 | 0.9243 | 0.9433 | 0.9769 | 0.4288 | 0.5879 | **0.7960** |
| smallcfg_4step | OR | Yes | 8 | 4step | 7.5 | 0.9111 | 0.9206 | 0.9386 | 0.9733 | 0.4334 | 0.5936 | **0.7951** |
| smallcfg_2step | OR | No | 0 | 2step | 0.0 | 0.8939 | 0.9150 | 0.9404 | 0.9753 | 0.4412 | 0.6045 | **0.7951** |
| smallcfg_2step | OR | No | 0 | 2step | 3.0 | 0.8862 | 0.9129 | 0.9387 | 0.9741 | 0.4430 | 0.6121 | **0.7945** |
| smallcfg_4step | OR | No | 0 | 4step | 0.0 | 0.8890 | 0.9117 | 0.9376 | 0.9736 | 0.4494 | 0.6055 | **0.7945** |
| normalcfg_4step | OR | No | 0 | 4step | 7.5 | 0.8847 | 0.9095 | 0.9379 | 0.9729 | 0.4384 | 0.6113 | **0.7924** |
| smallcfg_4step | OR | No | 0 | 4step | 7.5 | 0.8704 | 0.9072 | 0.9298 | 0.9667 | 0.4533 | 0.6040 | **0.7886** |

---


---

## Data Analysis

### 1. BR vs OR：数据集差异

BR 实验的 VBench 平均分（0.8052–0.8081）**系统性高于** OR 实验（0.7886–0.7965），差距约 **+0.01**。

| 指标 | BR 最优 | OR 最优 | BR 优势 |
|---|---|---|---|
| **VBench Avg** | 0.8081 | 0.7965 | +0.0116 |
| **Aesth_Q** | 0.4684 | 0.4533 | +0.0151 |
| **Img_Q** | 0.6496 | 0.6121 | +0.0375 |
| **Subj_Con** | 0.8974 | 0.9156 | **OR 更优** |
| **BG_Con** | 0.9280 | 0.9243 | +0.0037 |

> **结论**: BR 任务因保留了原始背景，在美学和图像质量上天然占优；OR 任务在主体一致性上更强（可能因为被移除物体导致评分基准不同）。**两个任务不宜直接横向比较**，应分别分析。

---

### 2. Blend 效果分析

#### OR 数据集：Blend 显著提升一致性

| 配置 | No Blend | Blend+Dil8 | Δ Avg |
|---|---|---|---|
| 2step_gs0.0 | 0.7951 | 0.7960 | **+0.0009** |
| 2step_gs3.0 | 0.7945 | 0.7964 | **+0.0019** |
| 4step_gs0.0 | 0.7945 | 0.7965 | **+0.0020** |
| 4step_gs7.5 | 0.7886 | 0.7951 | **+0.0065** |
| normalcfg_gs7.5 | 0.7924 | 0.7963 | **+0.0039** |

- Blend 在 OR 上**一致性地提升所有配置**，尤其在高 GS 时提升最大（+0.0065）
- 核心增益来自 Subj_Con 和 BG_Con（分别提升约 +0.02~+0.04）

#### BR 数据集：Blend 几乎无影响

| 配置 | No Blend | Blend+Dil8 | Δ Avg |
|---|---|---|---|
| 2step_gs0.0 | 0.8078 | 0.8074 | **-0.0004** |
| 4step_gs0.0 | 0.8081 | 0.8080 | **-0.0001** |
| normalcfg_gs7.5 | 0.8077 | 0.8062 | **-0.0015** |

- BR 上 Blend 对 VBench 无正向贡献，甚至略有下降
- **但 Pixel Metrics 下降明显**: No Blend 的 PSNR=31.81 vs Blend 的 PSNR=30.64（差 1.17 dB），说明 Blend 的边界混合引入了像素偏差

> **结论**: OR 必须开 Blend；BR 不建议开 Blend（会损害像素精度）。

---

### 3. Guidance Scale (GS) 效果

#### OR 数据集

| 对比 | GS=0 | GS>0 | Δ Avg |
|---|---|---|---|
| 2step Blend | 0.7960 | 0.7964 (gs3) | +0.0004 |
| 4step Blend | 0.7965 | 0.7951 (gs7.5) | **-0.0014** |
| 2step NoBlend | 0.7951 | 0.7945 (gs3) | -0.0006 |
| 4step NoBlend | 0.7945 | 0.7886 (gs7.5) | **-0.0059** |

- GS=0（无引导）普遍优于有 GS 的配置
- **4step + gs7.5 + noblend 是最差的 OR 组合**（0.7886），高 GS 导致 Subj_Con 和 BG_Con 明显退化

#### BR 数据集

| 对比 | GS=0 | GS>0 | Δ Avg |
|---|---|---|---|
| 2step NoBlend | 0.8078 | 0.8078 (gs3) | ±0 |
| 4step NoBlend | 0.8081 | 0.8064 (gs7.5) | **-0.0017** |

- BR 上也是 GS=0 最优或持平
- GS 增大 → Pixel Metrics 恶化（PSNR 从 31.81 降到 28.92）

> **结论**: **GS=0 是最稳健的选择**。Text guidance 在当前设置下没有正向贡献，反而引入噪声。

---

### 4. 步数（2-Step vs 4-Step）

#### OR (Blend 组)

| 对比 | 2-Step | 4-Step | Δ Avg |
|---|---|---|---|
| gs=0.0 | 0.7960 | 0.7965 | +0.0005 |
| gs>0 | 0.7964 (gs3) | 0.7951 (gs7.5) | -0.0013 |

#### BR (NoBlend 组)

| 对比 | 2-Step | 4-Step | Δ Avg |
|---|---|---|---|
| gs=0.0 | 0.8078 | 0.8081 | +0.0003 |
| gs>0 | 0.8078 (gs3) | 0.8064 (gs7.5) | -0.0014 |

- 在 GS=0 时，4-Step 非常轻微地优于 2-Step（约 +0.0003~0.0005）
- 但 4-Step 的推理时间是 2-Step 的 **2 倍**
- **性价比角度看 2-Step 更优**

> **结论**: 2-Step 和 4-Step 几乎无差异。考虑到训练/推理成本，优先使用 2-Step。

---

### 5. CFG 类型：smallcfg vs normalcfg

仅在 4step+gs7.5 上可比较：

| 配置 | smallcfg | normalcfg | Δ Avg |
|---|---|---|---|
| OR NoBlend | 0.7886 | 0.7924 | **+0.0038** (normalcfg 更优) |
| OR Blend | 0.7951 | 0.7963 | **+0.0012** |
| BR NoBlend | 0.8064 | 0.8077 | **+0.0013** |
| BR Blend | 0.8052 | 0.8062 | **+0.0010** |

- normalcfg 在所有场景下均**略优**于 smallcfg
- 但差异很小（0.001~0.004），不构成决定性优势

> **结论**: normalcfg 表现略好，但优势微弱。如果训练资源有限，smallcfg 也完全可接受。

---

### 6. 最优配置推荐

| 任务 | 推荐配置 | VBench Avg | 关键理由 |
|---|---|---|---|
| **OR** | `smallcfg_4step_blend_dil8_gs0.0` | **0.7965** | Blend 对 OR 有稳定增益，GS=0 最稳健 |
| **BR** | `smallcfg_4step_noblend_nodil_gs0.0` | **0.8081** | NoBlend 保持像素精度，GS=0 最优 |
| **BR (像素精度优先)** | `smallcfg_2step_noblend_nodil_gs0.0` | PSNR=**31.81** | 最高 PSNR/SSIM，训练成本低 |

### 7. 关键发现总结

1. **GS=0 全局最优** — 文本引导在当前任务中无正向贡献（可能因为 inpainting 的 prompt 过于简单如 "clean background"，无法提供有效语义信息）
2. **Blend 对 OR 有效、对 BR 有害** — 因 OR 的 mask 边界更不规则，需要边界混合来平滑过渡；BR 的 mask 较规则，Blend 反而引入伪影
3. **2-Step ≈ 4-Step** — 差异可忽略（<0.001），性价比首选 2-Step
4. **所有实验的 VBench 差异极小**（OR 最大跨度仅 0.0079，BR 仅 0.0029）— 说明模型对这些超参数不敏感，基础模型质量是决定性因素
5. **BR 的 Pixel Metrics 远比 VBench 更敏感** — PSNR 跨度 4.7 dB，而 VBench Avg 跨度仅 0.003，**优化 BR 应优先看 Pixel Metrics**
