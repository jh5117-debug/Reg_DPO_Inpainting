#!/bin/bash
# DPO 数据集生成启动脚本（视频级三路对决策略）
# 使用 nohup 运行，SSH 断开后继续处理
#
# 用法: bash run_dpo_dataset.sh
# 监控: tail -f dpo_data/run.log

PYTHON=/home/hj/miniconda/envs/diffueraser/bin/python
LOG_DIR=dpo_data
LOG_FILE="${LOG_DIR}/run.log"

mkdir -p "${LOG_DIR}"

echo "========================================"
echo " DPO Dataset (video-level 3-way)"
echo " Log: ${LOG_FILE}"
echo "========================================"

nohup env CUDA_VISIBLE_DEVICES=1 "${PYTHON}" generate_dpo_negatives.py \
    --davis_root dataset/DAVIS \
    --ytvos_root dataset/YTBV \
    --output_dir "${LOG_DIR}" \
    --min_video_frames 22 \
    --comparison_fps 8 \
    --height 512 \
    --width 512 \
    --base_model_path /home/hj/DiffuEraser_new/weights/stable-diffusion-v1-5 \
    --vae_path /home/hj/DiffuEraser_new/weights/sd-vae-ft-mse \
    --diffueraser_path /home/hj/DiffuEraser_new/weights/diffuEraser \
    --propainter_model_dir /home/hj/DiffuEraser_new/weights/propainter \
    --pcm_weights_path /home/hj/DiffuEraser_new/weights/PCM_Weights \
    --resume \
    >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${LOG_DIR}/run.pid"
echo "Started! PID=${PID}"
echo ""
echo "Commands:"
echo "  Monitor: tail -f ${LOG_FILE}"
echo "  Stop:    kill \$(cat ${LOG_DIR}/run.pid)"
echo "  Status:  ps aux | grep generate_dpo"
