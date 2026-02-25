"""
Save checkpoint for Stage 1 finetune.

Usage:
  1. Change `input_dir` to the actual checkpoint directory (e.g., finetune-stage1/checkpoint-2000)
  2. Run: python save_checkpoint_stage1.py
"""
from accelerate import Accelerator
import os
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel

## ===== Modify these paths =====
# Base model for UNet2D structure
base_model_path = "/home/hj/DiffuEraser_new/weights/stable-diffusion-v1-5"
# BrushNet pretrained path
brushnet_pretrained_path = "/home/hj/DiffuEraser_new/weights/diffuEraser"
# Accelerator checkpoint directory from finetune training
input_dir = "/home/hj/Train_Diffueraser/finetune-stage1/checkpoint-xxxx"  # <- CHANGE to actual step
# Output directory for converted weights
output_dir = "/home/hj/Train_Diffueraser/converted_weights/finetuned-stage1"
## ==============================

# Initialize models with same structure as training
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

# Load model structure (must match training script initialization)
unet_main = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
brushnet = BrushNetModel.from_pretrained(brushnet_pretrained_path, subfolder="brushnet")

# Prepare and load state (must match the order in training script: unet_main, brushnet)
unet_main, brushnet = accelerator.prepare(unet_main, brushnet)
accelerator.load_state(input_dir)

# Unwrap and save
unet_main = accelerator.unwrap_model(unet_main)
brushnet = accelerator.unwrap_model(brushnet)

unet_main_path = os.path.join(output_dir, "unet_main")
os.makedirs(unet_main_path, exist_ok=True)
unet_main.save_pretrained(unet_main_path)

brushnet_path = os.path.join(output_dir, "brushnet")
os.makedirs(brushnet_path, exist_ok=True)
brushnet.save_pretrained(brushnet_path)

print(f'Stage 1 checkpoint saved to {output_dir}')
