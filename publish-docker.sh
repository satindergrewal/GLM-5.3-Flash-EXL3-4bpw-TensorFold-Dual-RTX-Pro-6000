#!/usr/bin/env bash
# Push the built image to Docker Hub as $DOCKER_USER/$DOCKER_REPO:<version>-<hash>
# and :latest, with OCI labels pointing at this repository.
# Requires a prior `docker login` (the push uses your existing credentials).
# ~24.5 GB uncompressed; only changed layers upload on re-push.
# Usage: DOCKER_USER=myuser ./publish-docker.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

TF_VERSION="${TF_VERSION:-v0.6.0}"
IMAGE="${IMAGE:-tensorfold-glm53:${TF_VERSION}}"
DOCKER_USER="${DOCKER_USER:?set DOCKER_USER to your Docker Hub username}"
DOCKER_REPO="${DOCKER_REPO:-glm-5.3-flash-exl3-4bpw-tensorfold}"
REPO_URL="${REPO_URL:-https://github.com/satindergrewal/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Dual-RTX-Pro-6000}"

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "ERROR: image $IMAGE missing — run ./build-image.sh first"; exit 1; }
hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE")
[[ -n "$hash" ]] || { echo "ERROR: $IMAGE has no tf.patches label — rebuild with ./build-image.sh"; exit 1; }
tag="${TF_VERSION}-${hash}"
target="$DOCKER_USER/$DOCKER_REPO"

echo "[publish] labelling $IMAGE as $target:$tag (+ :latest)"
docker build -q -t "$target:$tag" -t "$target:latest" \
  --label org.opencontainers.image.source="$REPO_URL" \
  --label org.opencontainers.image.licenses=Apache-2.0 \
  --label org.opencontainers.image.description="GLM-5.3-Flash EXL3 4bpw, 1M context, TensorFold $TF_VERSION + recipe patches $hash, 2x RTX PRO 6000 (x86_64, COMM=nccl)" \
  - <<<"FROM $IMAGE" >/dev/null
# the labels above sit on top of the built image: its tf.patches label must survive
[[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$target:$tag")" == "$hash" ]] ||
  { echo "ERROR: $target:$tag lost its tf.patches label"; exit 1; }

echo "[publish] pushing $target:$tag and :latest"
docker push "$target:$tag"
docker push "$target:latest"
digest=$(docker image inspect -f '{{range .RepoDigests}}{{println .}}{{end}}' "$target:$tag" | grep -m1 "^$target@" | cut -d@ -f2)
echo "[publish] done: docker pull $target:$tag"
echo "[publish] digest: $digest  (pin this in .env.example / README if you want immutability)"
