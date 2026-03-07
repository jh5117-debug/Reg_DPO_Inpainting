# Non-prompt Finetune SLURM 脚本简化

## 背景
合作者反馈原有训练脚本过于复杂，要求简化为 Non-prompt only 的两个独立 stage 脚本。

## 修改内容

### README.md 变更清单
- [x] `set -euo pipefail` → `set -e`
- [x] `HF_TOKEN` 从硬编码改为环境变量读取（`export HF_TOKEN`）
- [x] `PROJECT_HOME` 去掉 `:-$HOME/project` fallback
- [x] `WORK_DIR` 改为 `${PROJECT_HOME}/dev`
- [x] GPU 数量从 8 改为 4（`#SBATCH --gres=gpu:4`）
- [x] 删除所有 `TRAIN_MODE` prompt/non-prompt 分支判断
- [x] 删除 `sed -i` 路径替换逻辑
- [x] 删除 Prompt 版搭建逻辑（setup 脚本）
- [x] 删除 Prompt 版 HF 仓库下载
- [x] 训练脚本从 `02_train.sbatch` 拆分为 `02_train_stage1.sbatch` + `02_train_stage2.sbatch`
- [x] 更新 HuggingFace 仓库表（移除 prompt-code 仓库）
- [x] 更新目录结构附录（`DiffuEraser_finetune` → `dev`）
- [x] 更新参数指引和 FAQ
