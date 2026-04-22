#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/path/to/imagenet}"
OUTPUT_DIR="${2:-outputs}"
DEVICE="${3:-cuda}"

python train_imagenet.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model resnet152 \
  --epochs 120 \
  --batch-size 256 \
  --val-batch-size 256 \
  --workers 8 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --label-smoothing 0.1 \
  --warmup-epochs 5 \
  --device "${DEVICE}"

python train_imagenet.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model resnet152_sd \
  --sd-p-last 0.5 \
  --epochs 120 \
  --batch-size 256 \
  --val-batch-size 256 \
  --workers 8 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --label-smoothing 0.1 \
  --warmup-epochs 5 \
  --device "${DEVICE}"
