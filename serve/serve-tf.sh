#!/usr/bin/env bash
# Launch TensorFold TP2 on a single dual-GPU box: one container, both GPUs
# visible, one rank process per GPU (per-process CUDA_VISIBLE_DEVICES).
# Adapted from MiaAI-Lab's two-Spark launcher for x86_64 / COMM=nccl.
# Every knob via env — see .env.example.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

TF_VERSION="${TF_VERSION:-v0.6.0}"
IMAGE="${IMAGE:-tensorfold-glm53:${TF_VERSION}}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the local TR3-4bpw checkpoint directory (flat snapshot layout)}"
HF_DIR="${HF_DIR:-$PWD/hf}"
CACHE_DIR="${CACHE_DIR:-$PWD/cache}"
NAME="${NAME:-glm53-flash-tf-box}"
PORT="${PORT:-8888}"
MASTER_PORT="${MASTER_PORT:-29551}"
PARALLEL="${PARALLEL:-4}"
CONTEXT="${CONTEXT:-0}"                # 0 = largest native window that fits
MAX_TOKENS="${MAX_TOKENS:-32768}"
KV="${KV:-fp8}"
DENSE="${DENSE:-q4}"
COMM="${COMM:-nccl}"
MAX_IMAGES="${TENSORFOLD_GLM_MAX_IMAGES:-128}"
CACHE_GIB="${TF_GLM_CACHE_GIB:-0}"
RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-4}"
RANK0_GPU="${RANK0_GPU:-0}"
RANK1_GPU="${RANK1_GPU:-1}"

DRAFT=$(echo "$HF_DIR"/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/*)
[ -d "$DRAFT" ] || { echo "ERROR: DFlash2 not found under $HF_DIR — run ./download-dflash2.sh first"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: $MODEL_DIR/config.json missing — set MODEL_DIR to the checkpoint"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }

mkdir -p "$CACHE_DIR"
docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" --gpus all \
  --network host --ipc=host --shm-size 16g --init \
  -v "$MODEL_DIR:/model:ro" \
  -v "$HF_DIR:/root/.cache/huggingface:ro" \
  -v "$CACHE_DIR:/cache" \
  -v "$PWD/serve/start-ranks.sh:/workspace/start-ranks.sh:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e TF_GLM_KV="$KV" -e TF_GLM_DENSE="$DENSE" -e TF_GLM_COMM="$COMM" \
  -e TENSORFOLD_GLM_MAX_IMAGES="$MAX_IMAGES" \
  -e TF_GLM_CACHE_GIB="$CACHE_GIB" \
  -e TENSORFOLD_MEMORY_RESERVE_GIB="$RESERVE_GIB" \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e TRITON_CACHE_DIR=/cache/triton \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_DISABLE=1 \
  -e RANK0_GPU="$RANK0_GPU" -e RANK1_GPU="$RANK1_GPU" \
  -e PORT="$PORT" -e MASTER_PORT="$MASTER_PORT" \
  -e PARALLEL="$PARALLEL" -e CONTEXT="$CONTEXT" -e MAX_TOKENS="$MAX_TOKENS" \
  "$IMAGE" bash /workspace/start-ranks.sh

echo "[tf] container $NAME up; logs: docker logs -f $NAME"
echo "[tf] API:    http://127.0.0.1:$PORT/v1   (health: /health)"
