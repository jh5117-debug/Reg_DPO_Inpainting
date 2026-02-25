#!/bin/bash
#SBATCH --job-name=DiffuEraser_Total
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

set -e

echo "[SLURM] Job ID: $SLURM_JOB_ID"
echo "[SLURM] Node: $(hostname)"
echo "[SLURM] GPUs requested: 8"

##############################
# Activate mamba environment
##############################
source ~/.bashrc
# using the env name found in your logs
mamba activate diffueraser || source activate diffueraser

export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

##############################
# DGX A100 Optimizations
##############################
export NCCL_TOPO_FILE=/opt/nccl_topo/dgx-a100-topo.xml
export NCCL_NET_GDR_LEVEL=3
export NCCL_IB_DISABLE=0
export NCCL_P2P_LEVEL=SYS
export CUDA_DEVICE_ORDER=PCI_BUS_ID

##############################
# Move to project directory
##############################
cd "/home/hj/Train_Diffueraser"
echo "[SLURM] Working dir: $(pwd)"
WEIGHTS="/home/hj/DiffuEraser_new/weights"
DAVIS="/home/hj/Train_Diffueraser/dataset/DAVIS"

# Common Args
validation_image="['${DAVIS}/JPEGImages/480p/bear','${DAVIS}/JPEGImages/480p/boat']"
validation_mask="['${DAVIS}/Annotations/480p/bear','${DAVIS}/Annotations/480p/boat']"
validation_prompt="['clean background','clean background']"

############################################################
# STAGE 1: FINETUNE
############################################################
echo "========================================================"
echo "[SLURM] Starting Stage 1 Training..."
echo "========================================================"

# Note: Added --max_train_steps to prevent it from running for 10000 epochs
accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --mixed_precision bf16 \
  train_DiffuEraser_stage1.py \
  --base_model_name_or_path="${WEIGHTS}/stable-diffusion-v1-5" \
  --pretrained_stage1_path="${WEIGHTS}/diffuEraser" \
  --vae_path="${WEIGHTS}/sd-vae-ft-mse" \
  --davis_root="/home/hj/Train_Diffueraser/dataset/DAVIS" \
  --ytvos_root="/home/hj/Train_Diffueraser/dataset/YTBV" \
  --resolution=512 \
  --nframes=10 \
  --mixed_precision="fp16" \
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
  --max_train_steps=50000  # <--- ADJUST THIS IF NEEDED

echo "[SLURM] Stage 1 Training finished."

############################################################
# CONVERT STAGE 1 CHECKPOINT
############################################################
echo "[SLURM] Identifying latest checkpoint..."
# Find the latest checkpoint directory (sort by version number)
LATEST_CKPT_DIR=$(ls -d finetune-stage1/checkpoint-* | sort -V | tail -n 1)
# Extract the checkpoint name (e.g., checkpoint-50000)
CKPT_NAME=$(basename "$LATEST_CKPT_DIR")

if [ -z "$LATEST_CKPT_DIR" ]; then
    echo "Error: No checkpoint found in finetune-stage1/"
    exit 1
fi
echo "[SLURM] Found checkpoint: $LATEST_CKPT_DIR"

echo "[SLURM] Converting Stage 1 weights..."

# Create a temporary script to avoid modifying the original
cp save_checkpoint_stage1.py save_checkpoint_stage1_temp.py

# Use sed to replace the placeholder path with the actual path
# Replaces 'checkpoint-xxxx' with the actual checkpoint name found
sed -i "s|checkpoint-xxxx|$CKPT_NAME|g" save_checkpoint_stage1_temp.py

# Run the conversion
python save_checkpoint_stage1_temp.py

# Cleanup
rm save_checkpoint_stage1_temp.py
echo "[SLURM] Stage 1 Conversion finished."

############################################################
# STAGE 2: FINETUNE
############################################################
echo "========================================================"
echo "[SLURM] Starting Stage 2 Training..."
echo "========================================================"

FINETUNED_STAGE1="/home/hj/Train_Diffueraser/converted_weights/finetuned-stage1"

accelerate launch \
  --multi_gpu \
  --num_processes 8 \
  --mixed_precision fp16 \
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
  --max_train_steps=50000 

echo "[SLURM] Stage 2 Training finished."

############################################################
# CONVERT STAGE 2 CHECKPOINT
############################################################
echo "[SLURM] Identifying latest checkpoint for Stage 2..."
LATEST_CKPT_DIR_S2=$(ls -d finetune-stage2/checkpoint-* | sort -V | tail -n 1)
CKPT_NAME_S2=$(basename "$LATEST_CKPT_DIR_S2")

if [ -z "$LATEST_CKPT_DIR_S2" ]; then
    echo "Error: No checkpoint found in finetune-stage2/"
    exit 1
fi
echo "[SLURM] Found checkpoint: $LATEST_CKPT_DIR_S2"

echo "[SLURM] Converting Stage 2 weights..."
cp save_checkpoint_stage2.py save_checkpoint_stage2_temp.py
sed -i "s|checkpoint-xxxx|$CKPT_NAME_S2|g" save_checkpoint_stage2_temp.py
python save_checkpoint_stage2_temp.py
rm save_checkpoint_stage2_temp.py

echo "[SLURM] All done!"