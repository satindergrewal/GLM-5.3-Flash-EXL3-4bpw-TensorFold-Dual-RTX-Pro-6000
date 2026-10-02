# TensorFold v0.6 + GLM-5.3-Flash recipe patches, x86_64.
# Base: NVIDIA PyTorch container (CUDA 13.x userspace; runs under driver 580+ via
# forward compatibility). The build applies patches/*.patch (unified diffs against
# site-packages, applied with patch -p0) and sanity-imports the patched engine.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE}
ARG TF_SPEC
ARG EXTRAS
RUN pip install --no-cache-dir --upgrade "${TF_SPEC}" && pip install --no-cache-dir ${EXTRAS} && tensorfold --version
COPY . /opt/tf-patches
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done && \
    python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, tensorfold.vision.glm, av, xgrammar"
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH}
ENV HF_HOME=/root/.cache/huggingface TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton
WORKDIR /workspace
