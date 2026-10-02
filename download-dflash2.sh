#!/usr/bin/env bash
# Download the DFlash2 draft model (pinned revision) into ./hf as a standard
# HF cache, using the built recipe image (same library versions as serving).
# Usage: ./download-dflash2.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

TF_VERSION="${TF_VERSION:-v0.6.0}"
IMAGE="${IMAGE:-tensorfold-glm53:${TF_VERSION}}"
DFLASH2_ID="${DFLASH2_ID:-incoai/GLM-5.3-Flash-DFlash2}"
DFLASH2_REVISION="${DFLASH2_REVISION:-bf582e4eacc1810f76656d1811693ff6c6737d2a}"
HF_DIR="${HF_DIR:-$PWD/hf}"

mkdir -p "$HF_DIR"
docker run --rm --network host \
  -v "$HF_DIR:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE=0 \
  ${HF_TOKEN:+-e HF_TOKEN} \
  "$IMAGE" python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$DFLASH2_ID', revision='$DFLASH2_REVISION')
print('DFlash2 at:', p)
"

echo "[download] done. serve/serve-tf.sh mounts this cache automatically."
