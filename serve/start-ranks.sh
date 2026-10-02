#!/usr/bin/env bash
# Both TP2 ranks in one container: rank 1 (background) + rank 0 (API, foreground).
# Per-process CUDA_VISIBLE_DEVICES: each rank must see exactly one GPU, or NCCL
# refuses with "Multiple Ranks are using the same GPU" (and per-container GPU
# pinning breaks its SHM buffer import with 'invalid device ordinal').
set -euo pipefail

ARGS=(--tp 2 --parallel "${PARALLEL:-4}" --context "${CONTEXT:-0}"
      --max-tokens "${MAX_TOKENS:-32768}"
      --drafter /root/.cache/huggingface/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a
      --thinking --vision)

CUDA_VISIBLE_DEVICES="${RANK1_GPU:-1}" tensorfold serve /model --tp 2 \
  --rank 1 --master 127.0.0.1 --master-port "${MASTER_PORT:-29551}" "${ARGS[@]}" &
CUDA_VISIBLE_DEVICES="${RANK0_GPU:-0}" tensorfold serve /model --tp 2 \
  --rank 0 --master 127.0.0.1 --master-port "${MASTER_PORT:-29551}" "${ARGS[@]}" \
  --name "${SERVED_NAME:-GLM-5.3-Flash-EXL3-4bpw}" --host 0.0.0.0 --port "${PORT:-8888}"
wait -n
exit 1
