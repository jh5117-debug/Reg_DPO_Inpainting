================================================================
   DiffuEraser Finetune_With_Prompt 合作者操作指南 (一键部署版)
================================================================

本项目是带 Caption/Prompt 支持的 DiffuEraser 微调版本。
区别于非 Prompt 版本(训练时使用硬编码 "clean background")，
本版本使用 VLM 预生成的真实场景描述作为 text conditioning。

数据集和权重与非 Prompt 版本完全相同，无需重新下载。

================================================================
一、前置条件
================================================================

确保已安装 HuggingFace CLI 并登录：

pip install -U huggingface_hub
huggingface-cli login

================================================================
二、下载所有文件
================================================================

在 Home 目录下创建目录：

mkdir -p ~/DiffuEraser_finetune_prompt
cd ~/DiffuEraser_finetune_prompt

2.1 下载 Prompt 版代码 + 预生成 Captions
----------------------------------------------------------------

huggingface-cli download JiaHuang01/DiffuEraser-finetune-prompt-code \
    --repo-type dataset \
    --local-dir ./

2.2 下载基础代码 (libs, diffueraser, dataset工具等，来自非Prompt版)
----------------------------------------------------------------

huggingface-cli download JiaHuang01/DiffuEraser-finetune-code \
    --repo-type dataset \
    --local-dir ./

2.3 下载权重文件
----------------------------------------------------------------

mkdir -p weights
huggingface-cli download JiaHuang01/DiffuEraser-finetune-weights \
    --repo-type dataset \
    --local-dir weights/

================================================================
三、一键还原项目结构
================================================================

cd ~/DiffuEraser_finetune_prompt
bash setup_project_prompt.sh

当看到 "🎉 Setup Complete!" 时，说明代码、Captions、数据集、权重
都已经自动解压并归位。

此时的目录结构应为：~

DiffuEraser_finetune_prompt/
├── train_DiffuEraser_stage1.py   # Stage1 训练 (带 --caption_yaml)
├── train_DiffuEraser_stage2.py   # Stage2 训练 (带 --caption_yaml)
├── finetune_stage1.sh            # Stage1 启动脚本
├── finetune_stage2.sh            # Stage2 启动脚本
├── run_finetune_all.sbatch       # SLURM 全流程作业
├── save_checkpoint_stage1.py     # Stage1 权重转换
├── save_checkpoint_stage2.py     # Stage2 权重转换
├── generate_captions_ytvos.py    # (可选) YTVOS caption 生成
├── merge_captions.py             # (可选) caption 合并工具
├── libs/                         # 模型库
├── diffueraser/                  # DiffuEraser 管道
├── propainter/                   # ProPainter
├── dataset/
│   ├── DAVIS/                    # DAVIS 数据集
│   ├── YTBV/                     # YouTubeVOS 数据集
│   ├── finetune_dataset_caption.py  # 带 caption 的 Dataset
│   ├── utils.py
│   └── ...
├── captions/
│   ├── all_captions_merged.yaml  # 合并后的 caption (训练用)
│   ├── all_captions_BR.yaml      # DAVIS captions
│   ├── all_captions_ytvos.yaml   # YTVOS captions
│   └── *.yaml                    # 每个视频的单独 caption
└── weights/
    ├── stable-diffusion-v1-5/
    ├── diffuEraser/
    ├── sd-vae-ft-mse/
    └── animatediff-motion-adapter-v1-5-2/

================================================================
四、安装 Python 依赖
================================================================

conda create -n diffueraser python=3.10 -y
conda activate diffueraser

pip install -r requirements.txt

# 或从 environment.yml 创建:
# conda env create -f environment.yml

================================================================
五、配置 Accelerate (多卡训练)
================================================================

accelerate config

# 推荐配置:
#   - multi-GPU
#   - num_processes: GPU 数量 (如 8)
#   - mixed_precision: bf16

================================================================
六、修改脚本路径 (一键替换)
================================================================

脚本中的路径默认指向原作者的路径，需要替换成你的路径。
复制以下命令并在终端运行：

cd ~/DiffuEraser_finetune_prompt

PROJECT_DIR=$(pwd)
WEIGHTS_DIR="${PROJECT_DIR}/weights"

for f in finetune_stage1.sh finetune_stage2.sh run_finetune_all.sbatch save_checkpoint_stage1.py save_checkpoint_stage2.py; do
    sed -i "s|/home/hj/Train_Diffueraser_prompt|${PROJECT_DIR}|g" $f
    sed -i "s|/home/hj/Train_Diffueraser/dataset|${PROJECT_DIR}/dataset|g" $f
    sed -i "s|/home/hj/DiffuEraser_new/weights|${WEIGHTS_DIR}|g" $f
    echo "Fixed path in $f"
done

================================================================
七、开始训练
================================================================

7.1 单独运行 (推荐调试用)
----------------------------------------------------------------

Stage 1 (训练 UNet2D + BrushNet 所有参数):

mkdir -p logs converted_weights
bash finetune_stage1.sh

Stage 2 (在 Stage1 基础上训练时序层):
(需要先完成 Stage1 并转换权重)

bash finetune_stage2.sh

7.2 SLURM 一键提交 (推荐生产用)
----------------------------------------------------------------

自动执行: Stage1训练 → 权重转换 → Stage2训练 → 权重转换

mkdir -p logs converted_weights
sbatch run_finetune_all.sbatch

查看日志:

tail -f logs/DiffuEraser_Prompt-*.out

================================================================
八、监控训练
================================================================

# 查看 Stage1 日志
tail -f finetune-stage1.log

# 查看 Stage2 日志
tail -f finetune-stage2.log

# 查看 GPU 使用
nvidia-smi

================================================================
九、训练完成后：转换权重
================================================================

如果使用 sbatch，权重会自动转换。

如果手动训练，需要手动转换：

Stage 1:
    1. 修改 save_checkpoint_stage1.py 中 checkpoint-xxxx 为实际步数
    2. python save_checkpoint_stage1.py

Stage 2:
    1. 修改 save_checkpoint_stage2.py 中 checkpoint-xxxx 为实际步数
    2. python save_checkpoint_stage2.py

转换后的权重保存在 converted_weights/ 目录下。

================================================================
十、关键区别：Prompt 版 vs 非 Prompt 版
================================================================

| 项目               | 非 Prompt 版                | Prompt 版                        |
|--------------------|----------------------------|----------------------------------|
| 训练 Caption       | 硬编码 "clean background"  | 使用 VLM 生成的真实场景描述      |
| Dataset 类         | FinetuneDataset            | FinetuneDatasetWithCaption       |
| 新增参数           | 无                         | --caption_yaml                   |
| Caption 数据       | 无                         | captions/all_captions_merged.yaml|
| Caption 数量       | 0                          | 3561 (90 DAVIS + 3471 YTVOS)     |
| 权重/数据集        | 相同                       | 相同                             |

================================================================
十一、常见问题
================================================================

Q: CUDA OOM 怎么办？
A: 尝试 --gradient_checkpointing 或减小 --nframes / --resolution

Q: 合作者不在同一个机器，如何传输权重？
A: 把 converted_weights/ 打包上传到共享存储或 HuggingFace

Q: 如何验证 Caption 是否生效？
A: 查看训练日志，FinetuneDatasetWithCaption 初始化时会打印
   "X captions loaded"。X 应为 3561。

Q: 我想重新生成 Captions 怎么做？
A: 需要 Qwen2.5-VL 模型和 qwen_env 环境，参考 generate_captions_ytvos.py
