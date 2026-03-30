#!/bin/bash
#SBATCH --job-name=DPO_S2
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dpo-stage2-%j.out

set -euo pipefail

if [[ -z "${PROJECT_HOME:-}" ]]; then
  echo "PROJECT_HOME is not set. Please export PROJECT_HOME before running sbatch." >&2
  exit 1
fi

PROJECT_ROOT="${PROJECT_HOME}/dev/Reg_DPO_Inpainting"

NUM_GPUS=${NUM_GPUS:-8}
WEIGHTS_DIR=${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}
DPO_DATA_ROOT=${DPO_DATA_ROOT:-${PROJECT_ROOT}/data/DPO_Finetune_data}
PRETRAINED_DPO_S1=${PRETRAINED_DPO_S1:-${PROJECT_ROOT}/dpo-finetune-stage1/best_weights}
REF_MODEL_PATH=${REF_MODEL_PATH:-${PROJECT_ROOT}/finetune-stage2/converted_weights_step34000}
BASELINE_UNET_PATH=${BASELINE_UNET_PATH:-${PROJECT_ROOT}/weights/diffuEraser}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LR=${LR:-1e-6}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
LR_WARMUP=${LR_WARMUP:-500}
MAX_STEPS=${MAX_STEPS:-30000}
CKPT_STEPS=${CKPT_STEPS:-2000}
CKPT_LIMIT=${CKPT_LIMIT:-3}
VAL_STEPS=${VAL_STEPS:-2000}
NFRAMES=${NFRAMES:-16}
SEED=${SEED:-42}
MIXED_PRECISION=${MIXED_PRECISION:-fp16}
WANDB_PROJECT=${WANDB_PROJECT:-DPO_Diffueraser}
BETA_DPO=${BETA_DPO:-500.0}
DAVIS_OVERSAMPLE=${DAVIS_OVERSAMPLE:-10}

source ~/.bashrc
conda activate diffueraser
cd "$PROJECT_ROOT"

# === 所有缓存/日志/artifacts 必须在项目目录内，禁止写入 home ===
export WANDB_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_CACHE_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_DATA_DIR="${PROJECT_ROOT}/.wandb_cache"
export WANDB_CONFIG_DIR="${PROJECT_ROOT}/.wandb_cache/config"
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  WANDB_API_KEY_FROM_NETRC=$(grep -A2 'api.wandb.ai' "${HOME}/.netrc" | grep password | awk '{print $2}' | head -n 1 || true)
  if [[ -n "${WANDB_API_KEY_FROM_NETRC}" ]]; then
    export WANDB_API_KEY="${WANDB_API_KEY_FROM_NETRC}"
  fi
fi
export HF_HOME="${PROJECT_ROOT}/.hf_cache"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/.hf_cache"
mkdir -p "${WANDB_DIR}" "${WANDB_CONFIG_DIR}" "${HF_HOME}"


python DPO_finetune/scripts/run_dpo_stage2.py \
  --num_gpus ${NUM_GPUS} \
  --weights_dir ${WEIGHTS_DIR} \
  --dpo_data_root ${DPO_DATA_ROOT} \
  --pretrained_dpo_stage1 ${PRETRAINED_DPO_S1} \
  --ref_model_path ${REF_MODEL_PATH} \
  --baseline_unet_path ${BASELINE_UNET_PATH} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM} \
  --learning_rate ${LR} \
  --lr_scheduler ${LR_SCHEDULER} \
  --lr_warmup_steps ${LR_WARMUP} \
  --max_train_steps ${MAX_STEPS} \
  --checkpointing_steps ${CKPT_STEPS} \
  --checkpoints_total_limit ${CKPT_LIMIT} \
  --validation_steps ${VAL_STEPS} \
  --nframes ${NFRAMES} \
  --seed ${SEED} \
  --mixed_precision ${MIXED_PRECISION} \
  --wandb_project ${WANDB_PROJECT} \
  --beta_dpo ${BETA_DPO} \
  --davis_oversample ${DAVIS_OVERSAMPLE} \
  --chunk_aligned
