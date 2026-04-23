#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-./data/imagenet1k_imagefolder}"
OUTPUT_DIR="${2:-outputs}"
NPROC="${NPROC:-4}"
BASE_PORT="${MASTER_PORT:-29501}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-112}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# Reduce allocator fragmentation risk near memory limit.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export MASTER_PORT="${BASE_PORT}"
torchrun --nproc_per_node="${NPROC}" train_imagenet.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model resnet152 \
  --epochs 120 \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --workers 8 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --label-smoothing 0.1 \
  --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"

export MASTER_PORT=$((BASE_PORT + 1))
torchrun --nproc_per_node="${NPROC}" train_imagenet.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model resnet152_sd \
  --sd-p-last 0.5 \
  --epochs 120 \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --val-batch-size "${VAL_BATCH_SIZE}" \
  --workers 8 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --label-smoothing 0.1 \
  --warmup-epochs 5 \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
