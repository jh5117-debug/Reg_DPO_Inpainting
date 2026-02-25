"""
Save checkpoint for Stage 2 finetune (Caption version).

Usage:
  1. Change `input_dir` to the actual checkpoint directory (e.g., finetune-stage2/checkpoint-2000)
  2. Ensure `pretrained_brushnet_path` points to finetuned stage1 brushnet
  3. Run: python save_checkpoint_stage2.py
"""
from accelerate import Accelerator
import os
from libs.brushnet_CA import BrushNetModel
from libs.unet_2d_condition import UNet2DConditionModel
from libs.unet_motion_model import MotionAdapter, UNetMotionModel

## ===== Modify these paths =====
base_model_name_or_path = "/home/hj/DiffuEraser_new/weights/stable-diffusion-v1-5"
pretrained_brushnet_path = "/home/hj/Train_Diffueraser_prompt/converted_weights/finetuned-stage1/brushnet"
motion_path = "/home/hj/DiffuEraser_new/weights/animatediff-motion-adapter-v1-5-2"
input_dir = "/home/hj/Train_Diffueraser_prompt/finetune-stage2/checkpoint-xxxx"  # <- CHANGE
output_dir = "/home/hj/Train_Diffueraser_prompt/converted_weights/finetuned-stage2"
## ==============================

accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision='fp16',
)

unet = UNet2DConditionModel.from_pretrained(base_model_name_or_path, subfolder="unet")
motion_adapter = MotionAdapter.from_pretrained(motion_path)
unet_main = UNetMotionModel.from_unet2d(unet, motion_adapter)

brushnet = BrushNetModel.from_pretrained(pretrained_brushnet_path)

unet_main, brushnet = accelerator.prepare(unet_main, brushnet)
accelerator.load_state(input_dir)

unet_main = accelerator.unwrap_model(unet_main)
brushnet = accelerator.unwrap_model(brushnet)

unet_main_path = os.path.join(output_dir, "unet_main")
os.makedirs(unet_main_path, exist_ok=True)
unet_main.save_pretrained(unet_main_path)

brushnet_path = os.path.join(output_dir, "brushnet")
os.makedirs(brushnet_path, exist_ok=True)
brushnet.save_pretrained(brushnet_path)

print(f'Stage 2 checkpoint saved to {output_dir}')
