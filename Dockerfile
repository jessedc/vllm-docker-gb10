# syntax=docker/dockerfile:1.7
#
# Build vLLM from source for the NVIDIA DGX Spark (GB10 Grace Blackwell, sm_121a).
#
# Strategy: start FROM the official vLLM image built from the SAME source ref we
# are compiling (build.sh derives it), which already ships the exact GPU stack
# that ref pins -- torch + cu130, FlashInfer and its prebuilt cubins, Triton, the
# full CUDA 13 toolkit incl. nvcc + ninja. The one thing that image lacks is
# kernels for this GPU: every official image is built for 8.0-12.0, and GB10 is
# 12.1a. So we recompile only vLLM's own C++/CUDA extensions for compute_121a
# against that toolchain, never touching the fragile torch/flashinfer stack.

ARG BASE_IMAGE=vllm/vllm-openai:v0.27.1-aarch64
FROM ${BASE_IMAGE}

# --- build configuration (override via --build-arg / build.sh) -------------
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.27.1
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
# compile/link. cuda-libraries-dev-<maj>-<min> pulls cublas/cusparse/cusolver/
# cufft/curand/nvjitlink/nvrtc dev in one shot. The version is derived from the
# base's own CUDA_VERSION env (13.0.x on v0.27.1, 13.2.x on the late-2026
# nightly) -- a hardcoded -13-0 would mismatch a newer base's runtimes.
# libdw-dev (elfutils): late-2026 main's bundled DeepGEMM JIT (_deep_gemm_C)
# includes <elfutils/libdwfl.h>; upstream's build stage installs elfutils-devel.
RUN cuda_mm="$(echo "${CUDA_VERSION}" | cut -d. -f1,2 | tr . -)" \
 && echo ">> base CUDA ${CUDA_VERSION} -> cuda-libraries-dev-${cuda_mm}" \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      git cmake ccache build-essential curl "cuda-libraries-dev-${cuda_mm}" \
      libdw-dev \
 && rm -rf /var/lib/apt/lists/*

# vLLM main builds Rust artifacts via setuptools-rust, which needs a cargo
# toolchain (absent from the base image). Install current stable via rustup.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# Pin the torch family the base image was built against so dependency resolution
# can never swap it for a generic PyPI wheel. When base and source ref match (the
# default) this pin equals what vLLM's requirements ask for anyway, so it is pure
# insurance; it still matters when building main on the nightly base, where the
# two can drift. flashinfer stays UNPINNED on purpose -- it JIT-compiles its
# kernels at runtime via the in-image CUDA 13 toolchain (which supports sm_121a),
# so a newer pin is safe. use_existing_torch.py below also strips torch from
# vLLM's own requirements as belt-and-suspenders.
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio)==' \
      > /opt/spark-constraints.txt \
 && echo "--- pinned toolchain ---" && cat /opt/spark-constraints.txt
ENV PIP_CONSTRAINT=/opt/spark-constraints.txt

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
# The script moved from the repo root to tools/ in late 2026 (after v0.27.1);
# accept either so both pinned releases and current main build.
RUN if [ -f tools/use_existing_torch.py ]; then python3 tools/use_existing_torch.py; \
    else python3 use_existing_torch.py; fi

# Compile vLLM (C++/CUDA extensions) from source against the existing torch.
# ccache is mounted as a BuildKit cache so repeat builds reuse object files.
RUN --mount=type=cache,target=/ccache \
    pip install --no-build-isolation -r requirements/build/cuda.txt \
 && pip install --no-build-isolation -v .

# flashinfer ships two companion packages of prebuilt kernels (flashinfer-cubin,
# flashinfer-jit-cache) that are only valid for their exact flashinfer-python
# version. Keep them when they still match after the install above -- which is
# the normal case now that the base image is built from this same vLLM ref, and
# it matters on this machine: without cubins flashinfer JIT-compiles at first
# boot, and that compile has been observed to drive the Spark's unified memory
# into a global OOM. Drop them only if the versions actually diverged (e.g. when
# building main on the nightly base), so flashinfer JITs rather than loading
# kernels built for a different version.
# Compare only the public version: flashinfer-jit-cache carries a local build
# suffix (0.6.16.post3+cu130) that flashinfer-python does not, and a naive string
# compare would read that as a mismatch and throw the cubins away.
RUN set -eu; \
    ver() { pip show "$1" 2>/dev/null | awk '/^Version:/{print $2}' | cut -d+ -f1; }; \
    fi_ver="$(ver flashinfer-python)"; \
    for pkg in flashinfer-cubin flashinfer-jit-cache; do \
      pkg_ver="$(ver "$pkg")"; \
      if [ -z "$pkg_ver" ]; then \
        echo ">> $pkg: not installed"; \
      elif [ "$pkg_ver" = "$fi_ver" ]; then \
        echo ">> $pkg $pkg_ver: kept (matches flashinfer-python $fi_ver)"; \
      else \
        echo ">> $pkg $pkg_ver: STALE vs flashinfer-python $fi_ver -- dropping"; \
        pip uninstall -y "$pkg"; \
      fi; \
    done

# Very-new model architectures can land in vLLM before they reach a tagged
# transformers release: vLLM may recognize a model_type (e.g. `gemma4_unified`)
# that HF AutoConfig still rejects, so the checkpoint fails before vLLM's own
# model class ever loads (vLLM only requires transformers >= 5.5.3, which is
# satisfied but insufficient). This escape hatch installs transformers from an
# arbitrary source revision.
#
# EMPTY BY DEFAULT: when the base image is the official image for the vLLM ref
# we are building (the norm), it already carries the transformers upstream
# validated against that release, and overwriting it with a much newer main is
# the bigger risk. Set --build-arg TRANSFORMERS_REF=main (or a tag/SHA) to opt
# in when a checkpoint needs an arch newer than the release ships.
#
# PIP_CONSTRAINT still pins the torch family so this cannot drag in a generic
# torch wheel; flashinfer stays untouched. Done AFTER the kernel compile so that
# expensive cached layer is reused.
ARG TRANSFORMERS_REF=
RUN if [ -n "${TRANSFORMERS_REF}" ]; then \
      pip install --no-cache-dir \
        "transformers @ git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_REF}"; \
    else \
      echo ">> keeping base image transformers $(pip show transformers | awk '/^Version:/{print $2}')"; \
    fi

# Record provenance inside the image and sanity-check the result.
RUN cp /opt/vllm-commit.txt /etc/vllm-source-commit \
 && python3 -c "import vllm, torch, flashinfer, transformers; \
print('vllm', vllm.__version__, '| torch', torch.__version__, \
'| flashinfer', flashinfer.__version__, '| transformers', transformers.__version__)"

WORKDIR /vllm-workspace
# Inherit the base image ENTRYPOINT ["vllm","serve"].
