#!/bin/bash
set -e

# DiffuEraser Helper Script: Prepare Dataset
# Usage: bash prepare_data.sh

echo "========================================"
echo "  DiffuEraser: Unpacking Datasets"
echo "========================================"

BASE_DIR=$(pwd)
DAVIS_DIR="${BASE_DIR}/dataset/DAVIS"
YTBV_DIR="${BASE_DIR}/dataset/YTBV"

# 1. DAVIS
echo "[1/2] Checking DAVIS..."
if [ -f "${DAVIS_DIR}/DAVIS_JPEGImages_480p.tar.gz" ]; then
    echo "  -> Found archive. Extracting to ${DAVIS_DIR}/JPEGImages ..."
    mkdir -p "${DAVIS_DIR}/JPEGImages"
    tar -xzf "${DAVIS_DIR}/DAVIS_JPEGImages_480p.tar.gz" -C "${DAVIS_DIR}/JPEGImages"
    echo "  -> Done."
    # Optional: cleanup
    # rm "${DAVIS_DIR}/DAVIS_JPEGImages_480p.tar.gz"
else
    echo "  -> Archive not found (or already extracted?). Skipping."
fi

# 2. YouTubeVOS
echo "[2/2] Checking YouTubeVOS..."
if [ -f "${YTBV_DIR}/YTBV_JPEGImages.tar" ]; then
    echo "  -> Found archive. Extracting to ${YTBV_DIR}/JPEGImages ..."
    mkdir -p "${YTBV_DIR}/JPEGImages"
    tar -xf "${YTBV_DIR}/YTBV_JPEGImages.tar" -C "${YTBV_DIR}"
    echo "  -> Done."
    # Optional: cleanup
    # rm "${YTBV_DIR}/YTBV_JPEGImages.tar"
else
    echo "  -> Archive not found (or already extracted?). Skipping."
fi

echo "========================================"
echo "  Dataset Preparation Completed!"
echo "========================================"
echo "Verify structures:"
ls -d "${DAVIS_DIR}/JPEGImages/480p" 2>/dev/null || echo "  ❌ DAVIS 480p missing"
ls -d "${YTBV_DIR}/JPEGImages" 2>/dev/null || echo "  ❌ YTBV JPEGImages missing"

echo ""
echo "You can now run finetuning."
