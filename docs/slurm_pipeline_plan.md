# DiffuEraser Finetune — 合作者 SLURM 全自动化部署改造

## 背景

合作者在 cluster 上操作，无法使用 `nohup`，需要通过 SLURM 脚本完成：
1. 数据下载 + 解压（数据量大，耗时长）
2. 环境搭建
3. 训练流程（Stage 1 → 转换 → Stage 2 → 转换）
4. 训练完成后自动保存权重

同时需要将所有路径从 `~/` 改为 `$PROJECT_HOME/xxx`，新增 DPO-dataset 到 README，并确保合作者 **只需修改 2 个变量（HF_TOKEN + 版本选择）然后 sbatch 即可**。

## HuggingFace 仓库验证结果 ✅

| # | 仓库 | 内容 | 大小 | 状态 |
|---|------|------|------|------|
| 1 | `DiffuEraser-finetune-code` | DAVIS.tar + YTBV.tar + code_base.tar.gz + environment.yml + setup_project.sh | ~10.1 GB | ✅ |
| 2 | `DiffuEraser-finetune-prompt-code` | captions.tar.gz + code_base_prompt.tar.gz + setup_project_prompt.sh | ~0.8 MB | ✅ |
| 3 | `DiffuEraser-finetune-weights` | weights_extras.tar (13.1 GB) + stable-diffusion-v1-5/ (含 unet/vae/text_encoder 等) | ~48.4 GB | ✅ |
| 4 | `DPO-dataset` | 60 davis_ + ~1964 ytbv_ 文件夹, 每个含 gt_frames/masks/neg_frames/meta.json | ~84 GB | ✅ |

> [!NOTE]
> 所有 4 个 HF 仓库文件均已验证完整。DPO-dataset 当前不参与 finetuning 流程，仅在 README 中记录供后续使用。

---

## Proposed Changes

### README 重写

#### [MODIFY] [README.md](file:///home/hj/DiffuEraser_Project/README.md)

完全重写 README，核心变更：

1. **所有路径使用 `$PROJECT_HOME` 环境变量**
   - `~/DiffuEraser_finetune` → `$PROJECT_HOME/DiffuEraser_finetune`
   - `~/DiffuEraser_downloads` → `$PROJECT_HOME/DiffuEraser_downloads`
2. **新增第 4 个 HF 仓库 `DPO-dataset`** 到仓库列表表格
3. **新增 SLURM 解压脚本** — 合作者一次 `sbatch` 完成下载 + 解压 + 环境安装
4. **前置配置部分** — 合作者编辑 config 文件填入 HF_TOKEN 和选择 Prompt / Non-Prompt 版
5. **训练参数修改指引** — 告诉合作者在哪个文件修改 learning_rate / max_train_steps / nframes 等
6. **美化目录结构** — 包含 DPO-dataset 的完整项目树

### SLURM 脚本

#### [NEW] setup_and_train.sbatch (写入到 README 中，合作者 copy 出来使用)

一个 **一键式** SLURM 脚本，流程：
1. 下载 4 个 HF repos
2. 解压所有 tar/tar.gz 文件
3. 安装 conda 环境（如果不存在）
4. 配置 accelerate（非交互式）
5. 自动替换路径中的硬编码
6. 运行 Stage 1 训练 → 转换权重 → 运行 Stage 2 训练 → 转换权重
7. 最终权重保存到 `$PROJECT_HOME/DiffuEraser_finetune/converted_weights/`

> [!IMPORTANT]
> SLURM partition 使用 `compute`, 资源参考 `srun -p compute -c 16 --mem=32G`。解压阶段只需 CPU + 内存，训练阶段需要 GPU，所以将拆为两个 sbatch：
> - `01_setup.sbatch` — CPU-only, 下载 + 解压 + 环境安装 (compute partition, 无 GPU)
> - `02_train.sbatch` — GPU, 训练 + 权重转换 (需要 GPU partition)

脚本中去除所有 DGX-specific 配置（NCCL_TOPO_FILE 等），使用通用 NCCL 设置。

---

### PRD 文档

#### [NEW] [PRD/slurm_pipeline_plan.md](file:///home/hj/DiffuEraser_Project/PRD/slurm_pipeline_plan.md)

保存本执行计划的副本到 PRD 目录。

---

## User Review Required

> [!IMPORTANT]
> **关于 SLURM 的 GPU partition 名称**：当前代码中使用 `pgpu`, 合作者 cluster 的 GPU partition 名是什么？
> 解压用 `compute` partition 已确认，但训练需要 GPU partition。如果不确定，我将使用 `gpu` 作为默认值并在 README 中标注需要修改。

> [!IMPORTANT]
> **关于 GPU 数量**：当前脚本默认 `--num_processes 8` (8卡)。合作者 cluster 有多少张 GPU？我将在脚本中预留变量 `NUM_GPUS=8` 让合作者修改。

---

## Verification Plan

### Automated Tests

1. **SLURM 脚本语法检查**
   ```bash
   bash -n 01_setup.sbatch
   bash -n 02_train.sbatch
   ```

2. **README 命令冒烟测试** — 逐行模拟 README 中的 bash 命令（非训练部分）：
   ```bash
   # 模拟目录创建
   PROJECT_HOME=/tmp/smoke_test
   mkdir -p $PROJECT_HOME/DiffuEraser_downloads
   # 验证所有 sed 替换命令的正确性（用 echo 模拟）
   # 验证 accelerate config 非交互式命令
   ```

3. **路径一致性检查** — grep 验证 README 和 sbatch 中不存在 `~/` 或 `/home/hj/` 硬编码路径

### Manual Verification

1. 合作者在 cluster 上运行 `sbatch 01_setup.sbatch`，等待完成后运行 `sbatch 02_train.sbatch`
2. 查看 logs 目录下的输出日志确认训练进度
3. 训练完成后检查 `$PROJECT_HOME/DiffuEraser_finetune/converted_weights/` 或 `$PROJECT_HOME/DiffuEraser_finetune_prompt/converted_weights/` 是否有权重文件
