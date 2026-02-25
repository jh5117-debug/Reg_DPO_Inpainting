#!/bin/bash
# DiffuEraser Stage 2 Finetune
# Trains only MotionAdapter temporal layers (UNet2D spatial + BrushNet frozen)
cd /home/hj/Train_Diffueraser

WEIGHTS="/home/hj/DiffuEraser_new/weights"
DAVIS="/home/hj/Train_Diffueraser/dataset/DAVIS"

# ⚠️ Change to your finetuned Stage 1 converted weights path
FINETUNED_STAGE1="/home/hj/Train_Diffueraser/converted_weights/finetuned-stage1"

validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"

accelerate launch  \
  train_DiffuEraser_stage2.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1="${FINETUNED_STAGE1}" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --motion_adapter_path="${WEIGHTS}/animatediff-motion-adapter-v1-5-2" \
  --davis_root="/home/hj/Train_Diffueraser/dataset/DAVIS" \
  --ytvos_root="/home/hj/Train_Diffueraser/dataset/YTBV" \
  --resolution=512 \
  --nframes=22 \
  --mixed_precision="fp16" \
  --train_batch_size=1 \
  --dataloader_num_workers=2 \
  --learning_rate=5e-06 \
  --resume_from_checkpoint="latest" \
  --validation_steps=2000 \
  --output_dir="finetune-stage2" \
  --logging_dir="logs-finetune-stage2" \
  --validation_image="$validation_image" \
  --validation_mask="$validation_mask" \
  --validation_prompt="$validation_prompt" \
  --checkpointing_steps=2000 \
  >finetune-stage2.log 2>&1
