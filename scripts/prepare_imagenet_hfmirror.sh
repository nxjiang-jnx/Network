#!/usr/bin/env bash
set -euo pipefail

# Zero-argument pipeline:
# 1) download imagenet-1k via hf-mirror
# 2) convert to ImageFolder layout expected by training scripts

bash scripts/download_imagenet_from_hfmirror.sh

python scripts/convert_hf_imagenet_to_imagefolder.py \
  --skip-existing

echo "Prepared dataset: ./data/imagenet1k_imagefolder"
