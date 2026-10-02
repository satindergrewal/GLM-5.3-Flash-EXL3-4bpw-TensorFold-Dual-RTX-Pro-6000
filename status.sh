#!/usr/bin/env bash
# One-glance state: container, health, advertised model info, GPU memory.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
NAME="${NAME:-glm53-flash-tf-box}"
PORT="${PORT:-8888}"

docker ps -a --filter "name=$NAME" --format '{{.Names}}  {{.Status}}' || true
if curl -fsS -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "API: healthy — http://127.0.0.1:$PORT/v1"
  curl -s -m 5 "http://127.0.0.1:$PORT/health" | python3 -c "
import json, sys
h = json.load(sys.stdin)
print(f'  pool: {h[\"pool_tokens\"]} tokens ({h[\"pool_free_tokens\"]} free) | kept prompts: {h[\"kept_prompts\"]}')"
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import json, sys
d = json.load(sys.stdin)['data'][0]
print(f'  model: {d[\"id\"]} | max_model_len: {d[\"max_model_len\"]}')"
else
  echo "API: not responding"
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
