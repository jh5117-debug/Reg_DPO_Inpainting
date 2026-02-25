#!/bin/bash
set -e
# DiffuEraser Prompt Finetune - Upload to HuggingFace
# Run this script from ~/Train_Diffueraser_prompt/

echo "=== Uploading to HuggingFace ==="

REPO_NAME="JiaHuang01/DiffuEraser-finetune-prompt-code"

echo "[1/4] Creating HuggingFace dataset repo..."
huggingface-cli repo create "$REPO_NAME" --type dataset 2>/dev/null || echo "Repo already exists."

echo "[2/4] Uploading setup_project_prompt.sh..."
huggingface-cli upload "$REPO_NAME" \
    setup_project_prompt.sh setup_project_prompt.sh \
    --repo-type dataset

echo "[3/4] Uploading code_base_prompt.tar.gz..."
huggingface-cli upload "$REPO_NAME" \
    code_base_prompt.tar.gz code_base_prompt.tar.gz \
    --repo-type dataset

echo "[4/4] Uploading captions.tar.gz..."
huggingface-cli upload "$REPO_NAME" \
    captions.tar.gz captions.tar.gz \
    --repo-type dataset

echo ""
echo "✅ All files uploaded to: https://huggingface.co/datasets/$REPO_NAME"
echo ""
echo "Note: Collaborator also needs to download from:"
echo "  - JiaHuang01/DiffuEraser-finetune-code     (for DAVIS.tar, YTBV.tar, code_base.tar.gz)"
echo "  - JiaHuang01/DiffuEraser-finetune-weights   (for weights)"
