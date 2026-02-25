#!/bin/bash
set -e

# DiffuEraser (Prompt-Enabled) Setup: One-click Environment Restoration
# Usage: bash setup_project_prompt.sh
#
# Prerequisites:
#   1. Non-prompt code repo already downloaded (for DAVIS.tar, YTBV.tar, code_base.tar.gz)
#      OR these archives are available in the current directory
#   2. Weights repo already downloaded to weights/

echo "============================================="
echo "  DiffuEraser (Prompt): Setting up Project"
echo "============================================="

unpack_tar() {
    ARCHIVE=$1
    DEST=$2
    if [ -f "$ARCHIVE" ]; then
        echo "  -> Found $ARCHIVE. Unpacking to $DEST ..."
        mkdir -p "$DEST"
        tar -xf "$ARCHIVE" -C "$DEST"
        echo "  -> Done."
    else
        echo "  ⚠️ Archive $ARCHIVE not found. Skipping."
    fi
}

echo ""
echo "[Step 1/6] Unpacking Base Code (libs, diffueraser, propainter, dataset)..."
echo "  (From non-prompt repo: JiaHuang01/DiffuEraser-finetune-code)"
if [ -f "code_base.tar.gz" ]; then
    echo "  -> Found code_base.tar.gz. Unpacking..."
    tar -xzf code_base.tar.gz
    echo "  -> Done."
else
    echo "  ⚠️ code_base.tar.gz not found!"
    echo "     Please download from JiaHuang01/DiffuEraser-finetune-code first."
    exit 1
fi

echo ""
echo "[Step 2/6] Unpacking Prompt Code & Scripts (overwriting base versions)..."
if [ -f "code_base_prompt.tar.gz" ]; then
    echo "  -> Found code_base_prompt.tar.gz. Unpacking..."
    tar -xzf code_base_prompt.tar.gz
    echo "  -> Done."
else
    echo "  ⚠️ code_base_prompt.tar.gz not found!"
    exit 1
fi

echo ""
echo "[Step 3/6] Unpacking Pre-generated Captions..."
if [ -f "captions.tar.gz" ]; then
    echo "  -> Found captions.tar.gz. Unpacking..."
    tar -xzf captions.tar.gz
    echo "  -> Done."
else
    echo "  ⚠️ captions.tar.gz not found!"
fi

echo ""
echo "[Step 4/6] Unpacking Weights..."
WEIGHTS_ARCHIVE="weights/weights_extras.tar"
if [ -f "$WEIGHTS_ARCHIVE" ]; then
    unpack_tar "$WEIGHTS_ARCHIVE" "weights"
elif [ -f "weights_extras.tar" ]; then
    unpack_tar "weights_extras.tar" "weights"
else
    echo "  ⚠️ weights_extras.tar not found in weights/!"
fi

echo "  -> Checking Stable Diffusion v1.5..."
if [ -d "weights/stable-diffusion-v1-5" ]; then
    echo "  ✅ Found Stable Diffusion v1.5 directory."
else
    echo "  ❌ MISSING Stable Diffusion v1.5! Please ensure weights download completed."
fi

echo ""
echo "[Step 5/6] Unpacking Datasets..."
unpack_tar "DAVIS.tar" "."
unpack_tar "YTBV.tar" "."

echo ""
echo "[Step 6/6] Verifying Structure..."
CHECK_DIRS=(
    "libs"
    "diffueraser"
    "dataset/DAVIS/JPEGImages"
    "dataset/finetune_dataset_caption.py"
    "captions/all_captions_merged.yaml"
    "weights/diffuEraser"
    "weights/sd-vae-ft-mse"
    "weights/stable-diffusion-v1-5"
    "weights/animatediff-motion-adapter-v1-5-2"
)
MISSING=0
for item in "${CHECK_DIRS[@]}"; do
    if [ -e "$item" ]; then
        echo "  ✅ Found: $item"
    else
        echo "  ❌ Missing: $item"
        MISSING=1
    fi
done

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "  🎉 Setup Complete! Ready for training."
    echo "============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Fix paths:  (see instructions below)"
    echo "  2. Configure:  accelerate config"
    echo "  3. Train:      sbatch run_finetune_all.sbatch"
else
    echo ""
    echo "============================================="
    echo "  ⚠️ Setup finished with missing components."
    echo "============================================="
fi
