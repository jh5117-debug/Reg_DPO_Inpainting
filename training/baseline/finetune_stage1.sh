#!/bin/bash
# DiffuEraser Stage 1 Finetune
# Trains UNet2D + BrushNet (all parameters)
cd /home/hj/Train_Diffueraser
WEIGHTS="/home/hj/DiffuEraser_new/weights"
DAVIS="/home/hj/Train_Diffueraser/dataset/DAVIS"
validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"
accelerate launch  \
  train_DiffuEraser_stage1.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1_path="${WEIGHTS}/diffuEraser" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --davis_root="/home/hj/Train_Diffueraser/dataset/DAVIS" \
  --ytvos_root="/home/hj/Train_Diffueraser/dataset/YTBV" \
  --resolution=8 \
  --nframes=2 \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --dataloader_num_workers=2 \
  --learning_rate=5e-06 \
  --resume_from_checkpoint="latest" \
  --validation_steps=2000 \
  --output_dir="finetune-stage1" \
  --logging_dir="logs-finetune-stage1" \
  --validation_image="$validation_image" \
  --validation_mask="$validation_mask" \
  --validation_prompt="$validation_prompt" \
  --checkpointing_steps=2000 \
  --gradient_checkpointing \
  >finetune-stage1.log 2>&1
