# syntax=docker/dockerfile:1.7
#
# Build vLLM from source for the NVIDIA DGX Spark (GB10 Grace Blackwell, sm_121a).
#
# Strategy: start FROM the official CUDA 13 nightly image, which already ships a
# known-good GPU toolchain for this exact GPU (PyTorch 2.11 + cu130, FlashInfer,
# Triton, full CUDA 13.0 toolkit incl. nvcc + ninja). We then recompile vLLM's
# C++/CUDA kernels from a pinned source revision against that toolchain, so the
# kernels are built for compute_121a instead of a compatibility fallback -- while
# never touching the fragile torch/flashinfer stack the base image got right.

ARG BASE_IMAGE=vllm/vllm-openai:cu130-nightly
FROM ${BASE_IMAGE}

# --- build configuration (override via --build-arg / build.sh) -------------
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=main
# GB10 Grace Blackwell compute capability. "12.1a" -> sm_121a.
ARG TORCH_CUDA_ARCH_LIST=12.1a
# Compile parallelism (Spark has 20 cores / 121 GB RAM).
ARG MAX_JOBS=16
ARG NVCC_THREADS=2

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    NVCC_THREADS=${NVCC_THREADS} \
    CCACHE_DIR=/ccache \
    CMAKE_BUILD_TYPE=Release \
    DEBIAN_FRONTEND=noninteractive

# Build tools the runtime base image lacks (nvcc, ninja, gcc/g++ are present),
# plus the CUDA math-library DEV headers/symlinks. The runtime base ships the
# CUDA .so runtimes but not the dev side (no cusparse.h, no unversioned
# libnvrtc.so, etc.), which torch's ATen headers and vLLM's CMake require to
# compile/link. cuda-libraries-dev-13-0 pulls cublas/cusparse/cusolver/cufft/
# curand/nvjitlink/nvrtc dev in one shot, matched to the base's CUDA 13.0.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git cmake ccache build-essential curl cuda-libraries-dev-13-0 \
 && rm -rf /var/lib/apt/lists/*

# vLLM main builds Rust artifacts via setuptools-rust, which needs a cargo
# toolchain (absent from the base image). Install current stable via rustup.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Pin the torch family the base image built for cu130/sm_121 so dependency
# resolution can never swap it for a generic PyPI wheel. flashinfer is left
# UNPINNED on purpose: vLLM main often pins a newer flashinfer than the nightly
# base ships, and flashinfer JIT-compiles its kernels at runtime using the
# in-image CUDA 13 toolchain (which supports sm_121a). use_existing_torch.py
# below also strips torch from vLLM's own requirements as belt-and-suspenders.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' \
      > /opt/spark-constraints.txt \
 && echo "--- pinned toolchain ---" && cat /opt/spark-constraints.txt
ENV PIP_CONSTRAINT=/opt/spark-constraints.txt

# Drop the base's prebuilt flashinfer kernel caches; they are tied to the base's
# flashinfer version and would mismatch the (possibly newer) one vLLM pins.
# flashinfer-python is reinstalled below to vLLM's pin and JITs at runtime.
RUN pip uninstall -y flashinfer-jit-cache flashinfer-cubin || true

# Fetch the requested vLLM source revision (branch, tag, or commit SHA).
# An explicit fetch of the ref keeps this robust when main advances between
# builds (the cached clone layer may predate the requested commit).
WORKDIR /opt
RUN git clone --filter=blob:none ${VLLM_REPO} vllm
WORKDIR /opt/vllm
RUN git fetch --depth 1 origin ${VLLM_REF} \
 && git checkout ${VLLM_REF} \
 && git rev-parse HEAD | tee /opt/vllm-commit.txt

# Strip torch from vLLM's requirement files so the pinned cu130 build is kept.
RUN python3 use_existing_torch.py

# Compile vLLM (C++/CUDA extensions) from source against the existing torch.
# ccache is mounted as a BuildKit cache so repeat builds reuse object files.
RUN --mount=type=cache,target=/ccache \
    pip install --no-build-isolation -r requirements/build/cuda.txt \
 && pip install --no-build-isolation -v .

# Very-new model architectures land in vLLM main before they reach a tagged
# transformers release. vLLM main recognizes e.g. model_type `gemma4_unified`,
# but transformers <= 5.6.0 does not, so HF AutoConfig rejects the checkpoint
# before vLLM's own model class ever loads it (vLLM only requires
# transformers >= 5.5.3, which is satisfied but insufficient). Install
# transformers from a newer source revision. PIP_CONSTRAINT still pins the torch
# family, so this cannot drag in a generic torch wheel; flashinfer stays
# untouched. Done AFTER the kernel compile so that cached layer is reused.
# Override the ref (a branch/tag/SHA) or set it to a release like "5.6.0" to opt
# out: --build-arg TRANSFORMERS_REF=...
ARG TRANSFORMERS_REF=main
RUN pip install --no-cache-dir \
      "transformers @ git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_REF}"

# Record provenance inside the image and sanity-check the result.
RUN cp /opt/vllm-commit.txt /etc/vllm-source-commit \
 && python3 -c "import vllm, torch, flashinfer, transformers; \
print('vllm', vllm.__version__, '| torch', torch.__version__, \
'| flashinfer', flashinfer.__version__, '| transformers', transformers.__version__)"

WORKDIR /vllm-workspace
# Inherit the base image ENTRYPOINT ["vllm","serve"].
