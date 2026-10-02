#!/usr/bin/env bash
# Build the serving image for this recipe: base NVIDIA PyTorch container +
# TensorFold v0.6.0 (pip, from the upstream git tag) + patches/*.patch.
# The patches hash is baked into the image label `tf.patches`; serve scripts
# and publish-docker.sh read it back from there.
# Usage: ./build-image.sh [--no-cache]
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

TF_VERSION="${TF_VERSION:-v0.6.0}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE="${IMAGE:-tensorfold-glm53:${TF_VERSION}}"
IMAGE_EXTRAS="${IMAGE_EXTRAS:-av==18.1.0 xgrammar>=0.2.8,<0.3}"

hash=$( (cat patches/*.patch; echo "$IMAGE_EXTRAS") | sha256sum | cut -c1-12)
echo "[build] $IMAGE  base=$BASE_IMAGE  tensorfold=$TF_VERSION  patches=$hash ($(compgen -G 'patches/*.patch' | wc -l) patches)"

nocache=(); [[ "${1:-}" == "--no-cache" ]] && nocache=(--no-cache)
docker build "${nocache[@]}" -t "$IMAGE" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg TF_SPEC="git+${TF_REPO}@${TF_VERSION}" \
  --build-arg PATCHES_HASH="$hash" \
  --build-arg EXTRAS="$IMAGE_EXTRAS" \
  -f Dockerfile patches

echo "[build] done: $IMAGE (tf.patches=$hash)"
