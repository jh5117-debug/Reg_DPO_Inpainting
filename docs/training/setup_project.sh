#!/bin/bash
set -e

# DiffuEraser Setup: One-click Environment Restoration
# Usage: bash setup_project.sh

echo "============================================="
echo "  DiffuEraser: Setting up Project Environment"
echo "============================================="

# Helper function
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
echo "[Step 1/4] Unpacking Code & Scripts..."
if [ -f "code_base.tar.gz" ]; then
    echo "  -> Found code_base.tar.gz. Unpacking to current directory..."
    tar -xzf code_base.tar.gz
    echo "  -> Done."
else
    echo "  ⚠️ code_base.tar.gz not found!"
fi

echo ""
echo "[Step 2/4] Unpacking Weights..."
# The weights archive is inside the 'weights/' folder because it was downloaded there
# And it contains folders like 'diffuEraser/', 'sd-vae-ft-mse/' (no 'weights/' prefix)
# So we unpack it INTO 'weights/' directory.
WEIGHTS_ARCHIVE="weights/weights_extras.tar"

if [ -f "$WEIGHTS_ARCHIVE" ]; then
    unpack_tar "$WEIGHTS_ARCHIVE" "weights"
else
    # Fallback: maybe user moved it to root?
    if [ -f "weights_extras.tar" ]; then
        unpack_tar "weights_extras.tar" "weights"
    else
        echo "  ⚠️ weights_extras.tar not found in weights/!"
    fi
fi

echo "  -> Checking Stable Diffusion v1.5..."
if [ -d "weights/stable-diffusion-v1-5" ]; then
    echo "  ✅ Found Stable Diffusion v1.5 directory."
else
    echo "  ❌ MISSING Stable Diffusion v1.5 directory! Please ensure download completed."
fi

echo ""
echo "[Step 3/4] Unpacking Datasets..."
# Packaged as 'dataset/DAVIS' inside tar, so unpack to current dir '.' puts them in 'dataset/'
unpack_tar "DAVIS.tar" "."
unpack_tar "YTBV.tar" "."

echo ""
echo "[Step 4/4] Verifying Structure..."
CHECK_DIRS=(
    "libs" "diffueraser" "dataset/DAVIS/JPEGImages" 
    "weights/diffuEraser" "weights/sd-vae-ft-mse"
)
MISSING=0
for d in "${CHECK_DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "  ❌ Missing directory: $d"
        MISSING=1
    else
        echo "  ✅ Found: $d"
    fi
done

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "  🎉 Setup Complete! You are ready to train."
    echo "============================================="
else
    echo ""
    echo "============================================="
    echo "  ⚠️ Setup finished with missing components."
    echo "============================================="
fi
