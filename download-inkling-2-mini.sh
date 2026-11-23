#!/usr/bin/env bash
# Pre-fetch the Inkling-2-Mini checkpoint (NVFP4 + MXFP8 experts, bundled DSpark draft) into the HF cache, so
# run-inkling-2-mini.sh can serve it without a cold download at startup.
#
# Runs `hf download` *inside* the vllm-spark image, so the host needs nothing
# installed (the image already ships the `hf` CLI + huggingface_hub).
#
# Usage:
#   ./download-inkling-2-mini.sh
#
# Env:
#   IMAGE       image to use            (default: vllm-spark:latest)
#   HF_HOME     host HF cache dir       (default: ~/.cache/huggingface)
#   HF_TOKEN    token for gated repos   (this repo is public, usually unneeded)
#   MODEL_REPO  override the model repo
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# Thinking Machines' ModelOpt NVFP4 checkpoint (~25 GiB incl. the 0.6 GiB
# mtp.safetensors DSpark draft). Single repo; load as released, never re-quantise.
MODEL_REPO="${MODEL_REPO:-thinkingmachines/Inkling-2-Mini}"

echo ">>> Downloading ${MODEL_REPO} into ${HF_HOME} (~25 GiB)"
# Allocate a TTY only when we have one (progress bars), so this also works
# headless / piped / in the background.
tty_flags=(-i); [[ -t 1 ]] && tty_flags+=(-t)
# Xet is DISABLED deliberately: huggingface_hub's chunked Xet backend repeatedly
# wedges on this box (many established-but-idle connections, zero byte growth,
# no timeout). Plain HTTPS with a read timeout runs ~2x faster here and resumes
# from .incomplete blobs, so a killed download never loses progress.
exec docker run --rm "${tty_flags[@]}" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=30 \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  --entrypoint bash "$IMAGE" -c '
    set -e
    repo="$1"
    echo ">>> hf download ${repo}"
    hf download "$repo"
    echo ">>> done. cached repos:"
    du -sh /root/.cache/huggingface/hub/models--* 2>/dev/null || true
  ' _ "$MODEL_REPO"
