#!/usr/bin/env bash
# Stop the TensorFold serve (removes the container; caches stay in ./cache).
set -euo pipefail
NAME="${NAME:-glm53-flash-tf-box}"
docker rm -f "$NAME" 2>/dev/null && echo "[tf] stopped ($NAME)" || echo "[tf] $NAME was not running"
