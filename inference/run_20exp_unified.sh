#!/bin/bash
set -uo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffueraser

trap 'echo ""; echo "[!] Ctrl+C detected, killing all child processes..."; kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; rm -f /tmp/gpu_lock_*.lock; echo "[!] All experiments stopped."; exit 1' INT TERM

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_ROOT="/home/hj/DiffuEraser_Project/exp_result"
LOG_DIR="${EXP_ROOT}/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

BASE_MODEL=/home/hj/DiffuEraser_new/weights/stable-diffusion-v1-5
VAE=/home/hj/DiffuEraser_new/weights/sd-vae-ft-mse
DIFFUERASER=/home/hj/DiffuEraser_new/weights/diffuEraser
PROPAINTER=/home/hj/DiffuEraser_new/weights/propainter
PCM=/home/hj/DiffuEraser_new/weights/PCM_Weights
I3D=/home/hj/DiffuEraser_new/weights/i3d_rgb_imagenet.pt
RAFT=/home/hj/DiffuEraser_new/weights/propainter/raft-things.pth

OR_VIDEO=/home/hj/DiffuEraser_new/DAVIS-2017-trainval-Full-Resolution/DAVIS/JPEGImages/Full-Resolution
OR_MASK=/home/hj/DiffuEraser_new/DAVIS-2017-trainval-Full-Resolution/DAVIS/Annotations/Full-Resolution
OR_CAPTION=/home/hj/DiffuEraser_Project/prompt_caption/all_captions_OR.yaml

BR_VIDEO=/home/hj/DiffuEraser_new/dataset/davis/JPEGImages_432_240/
BR_MASK=/home/hj/DiffuEraser_new/dataset/davis/test_masks/
BR_GT=/home/hj/DiffuEraser_new/dataset/davis/JPEGImages_432_240/
BR_CAPTION=/home/hj/DiffuEraser_Project/prompt_caption/all_captions_BR.yaml

INFERENCE_DIR=/home/hj/DiffuEraser_Project/inference
MAX_VIDEOS=50
ALL_GPUS=(1 2 3 4 5 6)
MIN_FREE_MB=18000
POLL_INTERVAL=30

acquire_gpu() {
    while true; do
        for gpu_id in "${ALL_GPUS[@]}"; do
            local lockfile="/tmp/gpu_lock_${gpu_id}.lock"
            if ( set -o noclobber; echo $$ > "$lockfile" ) 2>/dev/null; then
                local free_mb
                free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d '[:space:]')
                if [[ -n "$free_mb" ]] && (( free_mb >= MIN_FREE_MB )); then
                    echo "$gpu_id"
                    return 0
                else
                    rm -f "$lockfile"
                fi
            fi
        done
        echo "[WAIT] No GPU with >=${MIN_FREE_MB}MB free. Polling in ${POLL_INTERVAL}s..." >&2
        sleep "$POLL_INTERVAL"
    done
}

release_gpu() {
    local gpu_id=$1
    rm -f "/tmp/gpu_lock_${gpu_id}.lock"
}

run_experiment() {
    local gpu=$1 exp_type=$2 ckpt=$3 gs=$4 blend=$5 dil=$6 outdir=$7 logname=$8

    local blend_flag=""
    [[ "$blend" == "true" ]] && blend_flag="--blended"

    echo "[START] $outdir on GPU $gpu ($(date))"

    local vid_root mask_root gt_root caption_yaml extra_args
    if [[ "$exp_type" == "OR" ]]; then
        vid_root="$OR_VIDEO"
        mask_root="$OR_MASK"
        gt_root=""
        caption_yaml="$OR_CAPTION"
        extra_args="--height 360 --width 640 --video_length 60 --ref_stride 6"
    else
        vid_root="$BR_VIDEO"
        mask_root="$BR_MASK"
        gt_root="$BR_GT"
        caption_yaml="$BR_CAPTION"
        extra_args="--video_length 100 --ref_stride 3"
    fi

    local gt_flag=""
    [[ -n "$gt_root" ]] && gt_flag="--gt_root $gt_root"

    local i3d_flag=""
    [[ -n "$I3D" && -f "$I3D" ]] && i3d_flag="--i3d_model_path $I3D"
    local raft_flag=""
    [[ -n "$RAFT" && -f "$RAFT" ]] && raft_flag="--raft_model_path $RAFT"

    local exit_code=0
    CUDA_VISIBLE_DEVICES=$gpu python "${INFERENCE_DIR}/compare_all.py" \
      --dataset davis \
      --video_root "$vid_root" \
      --mask_root "$mask_root" \
      $gt_flag \
      --caption_yaml "$caption_yaml" \
      --ckpt "$ckpt" \
      --text_guidance_scale "$gs" \
      $extra_args \
      --neighbor_length 25 --subvideo_length 80 \
      --mask_dilation_iter "$dil" \
      $blend_flag \
      --base_model_path "$BASE_MODEL" \
      --vae_path "$VAE" \
      --diffueraser_path "$DIFFUERASER" \
      --propainter_model_dir "$PROPAINTER" \
      --pcm_weights_path "$PCM" \
      $i3d_flag \
      $raft_flag \
      --eval \
      --output_dir "${EXP_ROOT}/$outdir" \
      --max_videos $MAX_VIDEOS \
      2>&1 | tee "${LOG_DIR}/${logname}.log" || exit_code=$?

    if (( exit_code != 0 )); then
        echo "[FAIL] $outdir exited with code $exit_code on GPU $gpu"
        local target_dir="${EXP_ROOT}/$outdir"
        if [[ -d "$target_dir" ]]; then
            local file_count
            file_count=$(find "$target_dir" -type f 2>/dev/null | wc -l)
            if (( file_count == 0 )); then
                rm -rf "$target_dir"
                echo "[CLEANUP] Removed empty dir: $target_dir"
            fi
        fi
    else
        echo "[DONE] $outdir on GPU $gpu ($(date))"
    fi

    release_gpu "$gpu"
    return $exit_code
}

echo "============================================================"
echo "  20-Experiment Unified Runner (Single Inference)  |  $(date)"
echo "  Output -> ${EXP_ROOT}/"
echo "  Logs   -> ${LOG_DIR}/"
echo "  GPU pool: ${ALL_GPUS[*]}  (min free: ${MIN_FREE_MB}MB)"
echo "  Experiments: 3 ckpt configs x OR/BR x noblend/blend = 20"
echo "============================================================"

rm -f /tmp/gpu_lock_*.lock

declare -a EXPERIMENTS=(
    "OR|2-Step|0.0|false|0|smallcfg_2step_OR_noblend_nodil_gs0.0|s2_OR_gs0"
    "BR|2-Step|0.0|false|0|smallcfg_2step_BR_noblend_nodil_gs0.0|s2_BR_gs0"
    "OR|2-Step|3.0|false|0|smallcfg_2step_OR_noblend_nodil_gs3.0|s2_OR_gs3"
    "BR|2-Step|3.0|false|0|smallcfg_2step_BR_noblend_nodil_gs3.0|s2_BR_gs3"
    "OR|4-Step|0.0|false|0|smallcfg_4step_OR_noblend_nodil_gs0.0|s4_OR_gs0"
    "BR|4-Step|0.0|false|0|smallcfg_4step_BR_noblend_nodil_gs0.0|s4_BR_gs0"
    "OR|4-Step|7.5|false|0|smallcfg_4step_OR_noblend_nodil_gs7.5|s4_OR_gs7"
    "BR|4-Step|7.5|false|0|smallcfg_4step_BR_noblend_nodil_gs7.5|s4_BR_gs7"
    "OR|Normal CFG 4-Step|7.5|false|0|normalcfg_4step_OR_noblend_nodil_gs7.5|n4_OR_gs7"
    "BR|Normal CFG 4-Step|7.5|false|0|normalcfg_4step_BR_noblend_nodil_gs7.5|n4_BR_gs7"
    "OR|2-Step|0.0|true|8|smallcfg_2step_OR_blend_dil8_gs0.0|s2_OR_b_gs0"
    "BR|2-Step|0.0|true|8|smallcfg_2step_BR_blend_dil8_gs0.0|s2_BR_b_gs0"
    "OR|2-Step|3.0|true|8|smallcfg_2step_OR_blend_dil8_gs3.0|s2_OR_b_gs3"
    "BR|2-Step|3.0|true|8|smallcfg_2step_BR_blend_dil8_gs3.0|s2_BR_b_gs3"
    "OR|4-Step|0.0|true|8|smallcfg_4step_OR_blend_dil8_gs0.0|s4_OR_b_gs0"
    "BR|4-Step|0.0|true|8|smallcfg_4step_BR_blend_dil8_gs0.0|s4_BR_b_gs0"
    "OR|4-Step|7.5|true|8|smallcfg_4step_OR_blend_dil8_gs7.5|s4_OR_b_gs7"
    "BR|4-Step|7.5|true|8|smallcfg_4step_BR_blend_dil8_gs7.5|s4_BR_b_gs7"
    "OR|Normal CFG 4-Step|7.5|true|8|normalcfg_4step_OR_blend_dil8_gs7.5|n4_OR_b_gs7"
    "BR|Normal CFG 4-Step|7.5|true|8|normalcfg_4step_BR_blend_dil8_gs7.5|n4_BR_b_gs7"
)

TOTAL=${#EXPERIMENTS[@]}
QUEUED=0
declare -A JOB_PIDS

for exp_line in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_type ckpt gs blend dil outdir logname <<< "$exp_line"

    if [[ -d "${EXP_ROOT}/${outdir}" ]]; then
        local_files=$(find "${EXP_ROOT}/${outdir}" -name "summary.json" 2>/dev/null | wc -l)
        if (( local_files > 0 )); then
            echo "[SKIP] ${outdir} already complete (summary.json exists)"
            QUEUED=$((QUEUED + 1))
            continue
        fi
        rm -rf "${EXP_ROOT}/${outdir}"
        echo "[CLEANUP] Removed incomplete dir: ${EXP_ROOT}/${outdir}"
    fi

    gpu=$(acquire_gpu)
    QUEUED=$((QUEUED + 1))
    echo "[ASSIGN] ${QUEUED}/${TOTAL}: ${outdir} -> GPU ${gpu}"

    run_experiment "$gpu" "$exp_type" "$ckpt" "$gs" "$blend" "$dil" "$outdir" "$logname" &
    JOB_PIDS["$outdir"]=$!

    sleep 5
done

echo ""
echo "[INFO] All ${TOTAL} experiments assigned. Waiting for completion..."
FAIL_COUNT=0
for outdir in "${!JOB_PIDS[@]}"; do
    pid=${JOB_PIDS[$outdir]}
    if ! wait "$pid"; then
        echo "[WARN] $outdir (PID $pid) failed"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

rm -f /tmp/gpu_lock_*.lock

echo ""
echo "============================================================"
echo "  All 20 experiments finished at $(date)"
echo "  Failures: ${FAIL_COUNT}"
echo "============================================================"

if (( FAIL_COUNT > 0 )); then
    echo "[INFO] Re-running failed experiments..."
    for exp_line in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_type ckpt gs blend dil outdir logname <<< "$exp_line"
        if [[ -d "${EXP_ROOT}/${outdir}" ]]; then
            local_files=$(find "${EXP_ROOT}/${outdir}" -name "summary.json" 2>/dev/null | wc -l)
            if (( local_files > 0 )); then
                continue
            fi
            rm -rf "${EXP_ROOT}/${outdir}"
        fi

        echo "[RETRY] $outdir"
        gpu=$(acquire_gpu)
        run_experiment "$gpu" "$exp_type" "$ckpt" "$gs" "$blend" "$dil" "$outdir" "${logname}_retry" &
        JOB_PIDS["${outdir}_retry"]=$!
        sleep 5
    done
    wait
    rm -f /tmp/gpu_lock_*.lock
fi

echo ""
echo "Generating final report..."
python "${INFERENCE_DIR}/generate_report.py" \
  "${EXP_ROOT}/smallcfg_2step_OR_noblend_nodil_gs0.0" \
  "${EXP_ROOT}/smallcfg_2step_OR_noblend_nodil_gs3.0" \
  "${EXP_ROOT}/smallcfg_2step_BR_noblend_nodil_gs0.0" \
  "${EXP_ROOT}/smallcfg_2step_BR_noblend_nodil_gs3.0" \
  "${EXP_ROOT}/smallcfg_4step_OR_noblend_nodil_gs0.0" \
  "${EXP_ROOT}/smallcfg_4step_OR_noblend_nodil_gs7.5" \
  "${EXP_ROOT}/smallcfg_4step_BR_noblend_nodil_gs0.0" \
  "${EXP_ROOT}/smallcfg_4step_BR_noblend_nodil_gs7.5" \
  "${EXP_ROOT}/normalcfg_4step_OR_noblend_nodil_gs7.5" \
  "${EXP_ROOT}/normalcfg_4step_BR_noblend_nodil_gs7.5" \
  "${EXP_ROOT}/smallcfg_2step_OR_blend_dil8_gs0.0" \
  "${EXP_ROOT}/smallcfg_2step_OR_blend_dil8_gs3.0" \
  "${EXP_ROOT}/smallcfg_2step_BR_blend_dil8_gs0.0" \
  "${EXP_ROOT}/smallcfg_2step_BR_blend_dil8_gs3.0" \
  "${EXP_ROOT}/smallcfg_4step_OR_blend_dil8_gs0.0" \
  "${EXP_ROOT}/smallcfg_4step_OR_blend_dil8_gs7.5" \
  "${EXP_ROOT}/smallcfg_4step_BR_blend_dil8_gs0.0" \
  "${EXP_ROOT}/smallcfg_4step_BR_blend_dil8_gs7.5" \
  "${EXP_ROOT}/normalcfg_4step_OR_blend_dil8_gs7.5" \
  "${EXP_ROOT}/normalcfg_4step_BR_blend_dil8_gs7.5"

mv experiment_report.md "${EXP_ROOT}/experiment_report.md" 2>/dev/null

echo ""
echo "============================================================"
echo "  DONE!"
echo "  Results: ${EXP_ROOT}/"
echo "  Report:  ${EXP_ROOT}/experiment_report.md"
echo "  Logs:    ${LOG_DIR}/"
echo "============================================================"
