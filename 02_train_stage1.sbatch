#!/bin/bash
#SBATCH --job-name=DE_Stage1
#SBATCH --partition=pgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train-stage1-%j.out

source ~/.bashrc
conda activate diffueraser

cd "${PROJECT_HOME}/dev/DiffuEraser_finetune"

python run_train_stage1.py --num_gpus "${1:-1}"
