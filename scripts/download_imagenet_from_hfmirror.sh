#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/download_imagenet_from_hfmirror.sh \
#     /path/to/hf_raw \
#     https://hf-mirror.com \
#     /path/to/hf_token.txt
#
# Notes:
# - You must first accept the dataset terms on the website for ILSVRC/imagenet-1k.
# - token file should contain plain token text on one line.

RAW_DIR="${1:-$HOME/datasets/hf_imagenet1k_raw}"
HF_ENDPOINT_VALUE="${2:-https://hf-mirror.com}"
TOKEN_FILE="${3:-}"
CACHE_DIR="${RAW_DIR}/.cache"
export RAW_DIR
export CACHE_DIR

mkdir -p "${RAW_DIR}" "${CACHE_DIR}"

if [[ -n "${TOKEN_FILE}" && -f "${TOKEN_FILE}" ]]; then
  export HF_TOKEN="$(tr -d '\r\n' < "${TOKEN_FILE}")"
  echo "Loaded HF token from ${TOKEN_FILE}"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Using HF_TOKEN from environment."
else
  echo "No token found. Set HF_TOKEN or provide token file path."
  exit 1
fi

export HF_ENDPOINT="${HF_ENDPOINT_VALUE}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=120

echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "RAW_DIR=${RAW_DIR}"
echo "CACHE_DIR=${CACHE_DIR}"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

raw_dir = os.path.expanduser(os.environ["RAW_DIR"])
cache_dir = os.path.expanduser(os.environ["CACHE_DIR"])

print("Start snapshot_download...")
path = snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    local_dir=raw_dir,
    cache_dir=cache_dir,
    token=os.environ.get("HF_TOKEN"),
    max_workers=8,
)
print("Download finished:", path)
PY
