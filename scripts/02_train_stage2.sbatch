#!/bin/bash
#SBATCH --job-name=DE_Stage2
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train-stage2-%j.out

NUM_GPUS=${NUM_GPUS:-1}
DATA_DIR=${DATA_DIR:-}
WEIGHTS_DIR=${WEIGHTS_DIR:-}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
LR=${LR:-5e-6}
LR_SCHEDULER=${LR_SCHEDULER:-constant}
LR_WARMUP=${LR_WARMUP:-500}
MAX_STEPS=${MAX_STEPS:-50000}
CKPT_STEPS=${CKPT_STEPS:-2000}
CKPT_LIMIT=${CKPT_LIMIT:-3}
VAL_STEPS=${VAL_STEPS:-2000}
NFRAMES=${NFRAMES:-22}
SEED=${SEED:-42}
MIXED_PRECISION=${MIXED_PRECISION:-fp16}
WANDB_PROJECT=${WANDB_PROJECT:-DPO_Diffueraser}

source ~/.bashrc
conda activate diffueraser

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

CMD="python scripts/run_train_stage2.py \
  --num_gpus ${NUM_GPUS} \
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
  --wandb_project ${WANDB_PROJECT}"

[ -n "$DATA_DIR" ] && CMD="$CMD --data_dir ${DATA_DIR}"
[ -n "$WEIGHTS_DIR" ] && CMD="$CMD --weights_dir ${WEIGHTS_DIR}"

eval $CMD
