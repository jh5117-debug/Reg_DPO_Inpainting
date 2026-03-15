#!/bin/bash
#SBATCH --job-name=DE_Full
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train-all-%j.out

PROJECT_ROOT="${PROJECT_HOME}/dev/Reg_DPO_Inpainting"

NUM_GPUS=${NUM_GPUS:-8}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data}
WEIGHTS_DIR=${WEIGHTS_DIR:-${PROJECT_ROOT}/weights}
MIXED_PRECISION=${MIXED_PRECISION:-fp16}
SEED=${SEED:-42}
WANDB_PROJECT=${WANDB_PROJECT:-DPO_Diffueraser}

S1_BATCH_SIZE=${S1_BATCH_SIZE:-1}
S1_GRAD_ACCUM=${S1_GRAD_ACCUM:-1}
S1_LR=${S1_LR:-5e-6}
S1_LR_SCHEDULER=${S1_LR_SCHEDULER:-constant}
S1_LR_WARMUP=${S1_LR_WARMUP:-500}
S1_MAX_STEPS=${S1_MAX_STEPS:-26000}
S1_CKPT_STEPS=${S1_CKPT_STEPS:-2000}
S1_CKPT_LIMIT=${S1_CKPT_LIMIT:-3}
S1_VAL_STEPS=${S1_VAL_STEPS:-2000}
S1_NFRAMES=${S1_NFRAMES:-10}

S2_BATCH_SIZE=${S2_BATCH_SIZE:-1}
S2_GRAD_ACCUM=${S2_GRAD_ACCUM:-1}
S2_LR=${S2_LR:-5e-6}
S2_LR_SCHEDULER=${S2_LR_SCHEDULER:-constant}
S2_LR_WARMUP=${S2_LR_WARMUP:-500}
S2_MAX_STEPS=${S2_MAX_STEPS:-26000}
S2_CKPT_STEPS=${S2_CKPT_STEPS:-2000}
S2_CKPT_LIMIT=${S2_CKPT_LIMIT:-3}
S2_VAL_STEPS=${S2_VAL_STEPS:-2000}
S2_NFRAMES=${S2_NFRAMES:-22}

source ~/.bashrc
conda activate diffueraser
cd "$PROJECT_ROOT"

python scripts/run_train_all.py \
  --num_gpus ${NUM_GPUS} \
  --data_dir ${DATA_DIR} \
  --weights_dir ${WEIGHTS_DIR} \
  --mixed_precision ${MIXED_PRECISION} \
  --seed ${SEED} \
  --wandb_project ${WANDB_PROJECT} \
  --s1_batch_size ${S1_BATCH_SIZE} \
  --s1_gradient_accumulation_steps ${S1_GRAD_ACCUM} \
  --s1_learning_rate ${S1_LR} \
  --s1_lr_scheduler ${S1_LR_SCHEDULER} \
  --s1_lr_warmup_steps ${S1_LR_WARMUP} \
  --s1_max_train_steps ${S1_MAX_STEPS} \
  --s1_checkpointing_steps ${S1_CKPT_STEPS} \
  --s1_checkpoints_total_limit ${S1_CKPT_LIMIT} \
  --s1_validation_steps ${S1_VAL_STEPS} \
  --s1_nframes ${S1_NFRAMES} \
  --s2_batch_size ${S2_BATCH_SIZE} \
  --s2_gradient_accumulation_steps ${S2_GRAD_ACCUM} \
  --s2_learning_rate ${S2_LR} \
  --s2_lr_scheduler ${S2_LR_SCHEDULER} \
  --s2_lr_warmup_steps ${S2_LR_WARMUP} \
  --s2_max_train_steps ${S2_MAX_STEPS} \
  --s2_checkpointing_steps ${S2_CKPT_STEPS} \
  --s2_checkpoints_total_limit ${S2_CKPT_LIMIT} \
  --s2_validation_steps ${S2_VAL_STEPS} \
  --s2_nframes ${S2_NFRAMES}
