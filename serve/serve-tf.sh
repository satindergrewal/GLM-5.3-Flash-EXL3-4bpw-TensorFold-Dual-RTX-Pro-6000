#!/usr/bin/env bash
# Launch TensorFold TP2 on a single dual-GPU box: one container, both GPUs
# visible, one rank process per GPU (per-process CUDA_VISIBLE_DEVICES).
# Adapted from MiaAI-Lab's two-Spark launcher for x86_64 / COMM=nccl.
# Every knob via env — see .env.example.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

# per-box overrides: ./.env (sourced as bash, see .env.example).
# Precedence: an exported variable from the caller's environment wins over .env
# (so `QUANT=4bpw ./serve/serve-tf.sh` overrides a .env that says 3.5bpw).
declare -A _env_before=()
while IFS= read -r _n; do _env_before[$_n]=${!_n}; done < <(compgen -e)
[[ -f .env ]] && source .env
for _n in "${!_env_before[@]}"; do
  [[ "${!_n-}" == "${_env_before[$_n]}" ]] || export "$_n=${_env_before[$_n]}"
done
unset _env_before

# TF_VERSION is defaulted per quant arm in the case below (the 3.5bpw arm needs
# the mixed-k34 image; do not pre-set it here or the arm default never applies)

# --- quant switch: stock 4bpw mirror, or the 3.5bpw mixed encode -------------
QUANT="${QUANT:-4bpw}"                 # 4bpw | 3.5bpw
MODEL_DIR_4BPW="${MODEL_DIR_4BPW:-}"
MODEL_DIR_35BPW="${MODEL_DIR_35BPW:-}"
case "$QUANT" in
  4bpw)
    [[ -n "$MODEL_DIR_4BPW" ]] || { echo "ERROR: QUANT=4bpw needs MODEL_DIR_4BPW"; exit 1; }
    MODEL_DIR="$MODEL_DIR_4BPW"
    TF_VERSION="${TF_VERSION:-v0.6.5-mixed-k34}"
    SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-EXL3-4bpw}"
    VISION="${VISION:-1}"
    ;;
  3.5bpw)
    [[ -n "$MODEL_DIR_35BPW" ]] || { echo "ERROR: QUANT=3.5bpw needs MODEL_DIR_35BPW"; exit 1; }
    MODEL_DIR="$MODEL_DIR_35BPW"
    # the merged fork image: TensorFold 0.6.5 + the mixed-k34 support (per-expert-width
    # MoE, the pick-stride fix) - serves BOTH arms
    TF_VERSION="${TF_VERSION:-v0.6.5-mixed-k34}"
    SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash-EXL3-3.5bpw-mixed}"
    # vision works on this arm since the artifact ships the 4bpw chat template
    # (chat_template.jinja = the 4bpw one; TF's --vision extends its media branch;
    # the k35-native template is kept as chat_template.jinja.k35-orig but has no
    # branch TensorFold knows how to extend). Verified 2026-10-04: 3-image color ID.
    VISION="${VISION:-1}"
    # the auto-fit stops one 64k block short of native for this arm (983,040); explicit
    # context serves the full window with ~12.6 GiB/GPU headroom (measured 2026-10-04)
    CONTEXT="${CONTEXT:-1048576}"
    ;;
  *)
    echo "ERROR: QUANT must be 4bpw or 3.5bpw (got: $QUANT)"; exit 1 ;;
esac

# --- ABLIT mode (TODO, disabled): abliterated-checkpoint lane ----------------
# ABLIT=0 (default): serve the checkpoint selected above, untouched.
# ABLIT=1: serve the abliterated variant of the selected arm - MODEL_DIR_ABLIT_4BPW /
# MODEL_DIR_ABLIT_35BPW from .env. No abliterated checkpoint is wired yet (see
# .env.example); until one is validated this fails fast rather than serving silently.
ABLIT="${ABLIT:-0}"
case "$ABLIT" in
  0) ;;
  1)
    case "$QUANT" in
      4bpw)   ABLIT_DIR="${MODEL_DIR_ABLIT_4BPW:-}" ;;
      3.5bpw) ABLIT_DIR="${MODEL_DIR_ABLIT_35BPW:-}" ;;
    esac
    if [[ -z "$ABLIT_DIR" || ! -d "$ABLIT_DIR" ]]; then
      echo "ERROR: ABLIT=1 needs MODEL_DIR_ABLIT_4BPW or MODEL_DIR_ABLIT_35BPW set to a validated abliterated checkpoint (TODO: not wired yet)"; exit 1
    fi
    MODEL_DIR="$ABLIT_DIR"
    SERVED_NAME="${SERVED_NAME%-ablit}-ablit"
    ;;
  *) echo "ERROR: ABLIT must be 0 or 1 (got: $ABLIT)"; exit 1 ;;
esac
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
IMAGE="${IMAGE:-tensorfold-glm53:${TF_VERSION}}"

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
  -e TENSORFOLD_GLM_MAX_IMAGES="$MAX_IMAGES" -e VISION="$VISION" \
  -e SERVED_NAME="$SERVED_NAME" -e QUANT="$QUANT" \
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
