# 验证步骤调整与最大步数修改（需求与执行记录文档）

## 1. 需求背景 (Context)
合作者已从 Hugging Face 账号 `JiaHuang01/data_val` 下载了验证数据集。每次计算 validation 时，需要对 `data_val` 中的所有视频计算 PSNR 和 SSIM 的平均值图表，不再对每一个视频给出单独的 PSNR / SSIM WandB 图表（保留原有的平均值记录逻辑即可，原有的 `inference/metrics.py` 中的逻辑保留）。同时，将 Stage 1 和 Stage 2 训练脚本的 `max_train_steps` 更新为 26000 步。

## 2. 设计与执行计划 (Thought)
针对现有项目的代码结构分析，计划执行以下操作：
1. **分析当前逻辑**：当前项目中 `train_DiffuEraser_stage1.py` 与 `train_DiffuEraser_stage2.py` 中不仅记录了平均指标（如 `"val/psnr"` ），还在循环中记录了带有每个视频文件名的指标（如 `f"val/psnr_{r['video_name']}"`）。
2. **去除每个视频单独图表**：去除代码中通过 `batch_results['per_video']` 遍历记录各视频的 wandb log。将图表名称显式更新为 `val/psnr_mean` 和 `val/ssim_mean`，使其意义更加明确。保留平均值的 WandB 图表。
3. **设置最大步数**：更新所有关联启动脚本以及参数解析部分中的 `max_train_steps`/`MAX_STEPS` 使其默认为 `26000`。
4. **验证集数据源检查**：通过检查发现启动脚本（如 `run_train_stage1.py`，`run_train_stage2.py`）已经写死了评估数据集文件夹为 `eval_dir = os.path.join(project_root, "data_val")`，这满足每次验证测试指定验证集的需求，无需进一步修改。

## 3. 具体修改 (Implementation)
1. **修改点一：训练主文件**
   - 文件：`train_DiffuEraser_stage1.py` 和 `train_DiffuEraser_stage2.py`
   - 修改细节：去除了 190-210 行之间的 `for r in batch_results["per_video"]: wandb.log(...)` 逻辑。同时将全局平均 wandb log 的 key 更新为 `val/psnr_mean` 和 `val/ssim_mean`。
   
2. **修改点二：最大步数**
   - 文件：`scripts/02_train_stage1.sbatch`, `scripts/02_train_stage2.sbatch`, `scripts/02_train_all.sbatch`
   - 修改细节：将环境变量 `MAX_STEPS` 各处的默认值 `25000` 改为 `26000`。
   - 文件：`scripts/run_train_stage1.py`, `scripts/run_train_stage2.py`, `scripts/run_train_all.py`
   - 修改细节：将参数 `parser.add_argument("--max_train_steps", ...)` 的默认值 `25000` 改为 `26000`。

## 4. 验证方式 (Verification)
测试运行时可以通过类似如下命令检查参数解析正确性与运行正常性：
```bash
bash scripts/02_train_stage1.sbatch
# 检查 wandb 的输出图表，预期只会输出 val/psnr_mean 和 val/ssim_mean 两条曲线。
```
