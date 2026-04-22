#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data/imagenet1k_imagefolder}"
OUTPUT_DIR="${2:-outputs}"
DEVICE="${3:-cuda}"

python explore_deletion.py \
  --data-root "${DATA_ROOT}" \
  --model resnet152 \
  --checkpoint "${OUTPUT_DIR}/resnet152/best.pt" \
  --batch-size 256 \
  --workers 8 \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}/explore"

python explore_deletion.py \
  --data-root "${DATA_ROOT}" \
  --model resnet152_sd \
  --sd-p-last 0.5 \
  --checkpoint "${OUTPUT_DIR}/resnet152_sd/best.pt" \
  --batch-size 256 \
  --workers 8 \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}/explore"

python analyze_results.py \
  --resnet-csv "${OUTPUT_DIR}/explore/resnet152/deletion_curve.csv" \
  --sd-csv "${OUTPUT_DIR}/explore/resnet152_sd/deletion_curve.csv" \
  --output "${OUTPUT_DIR}/analysis_report.md"

python speedup_inference.py \
  --data-root "${DATA_ROOT}" \
  --model resnet152 \
  --checkpoint "${OUTPUT_DIR}/resnet152/best.pt" \
  --error-budget 0.5 \
  --batch-size 256 \
  --workers 8 \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}/accel"

python speedup_inference.py \
  --data-root "${DATA_ROOT}" \
  --model resnet152_sd \
  --sd-p-last 0.5 \
  --checkpoint "${OUTPUT_DIR}/resnet152_sd/best.pt" \
  --error-budget 0.5 \
  --batch-size 256 \
  --workers 8 \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}/accel"
