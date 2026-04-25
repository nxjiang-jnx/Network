#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data/imagenet1k_imagefolder}"
OUTPUT_DIR="${2:-outputs}"
GPU_RESNET="${GPU_RESNET:-cuda:0}"
GPU_SD="${GPU_SD:-cuda:1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
WORKERS="${WORKERS:-20}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-6}"
TARGET_TOP1="${TARGET_TOP1:-75.0}"

python explore_deletion.py \
  --data-root "${DATA_ROOT}" \
  --model resnet152 \
  --checkpoint "${OUTPUT_DIR}/resnet152/best.pt" \
  --batch-size "${VAL_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --device "${GPU_RESNET}" \
  --output-dir "${OUTPUT_DIR}/explore" &
PID_EXP_RESNET=$!

python explore_deletion.py \
  --data-root "${DATA_ROOT}" \
  --model resnet152_sd \
  --sd-p-last 0.5 \
  --checkpoint "${OUTPUT_DIR}/resnet152_sd/best.pt" \
  --batch-size "${VAL_BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --prefetch-factor "${PREFETCH_FACTOR}" \
  --device "${GPU_SD}" \
  --output-dir "${OUTPUT_DIR}/explore" &
PID_EXP_SD=$!

wait "${PID_EXP_RESNET}"
wait "${PID_EXP_SD}"

python analyze_results.py \
  --resnet-csv "${OUTPUT_DIR}/explore/resnet152/deletion_curve.csv" \
  --sd-csv "${OUTPUT_DIR}/explore/resnet152_sd/deletion_curve.csv" \
  --output "${OUTPUT_DIR}/analysis_report.md"

python speedup_inference.py \
  --checkpoint "${OUTPUT_DIR}/resnet152_sd/best.pt" \
  --deletion-csv "${OUTPUT_DIR}/explore/resnet152_sd/deletion_curve.csv" \
  --target-top1 "${TARGET_TOP1}" \
  --sd-p-last 0.5 \
  --batch-size "${VAL_BATCH_SIZE}" \
  --device "${GPU_SD}" \
  --output-dir "${OUTPUT_DIR}/accel/resnet152_sd"
