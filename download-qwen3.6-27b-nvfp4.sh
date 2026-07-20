#!/usr/bin/env bash
# Pre-fetch the unsloth Qwen3.6-27B-NVFP4 checkpoint into the HF cache, so
# run-qwen3.6-27b-nvfp4.sh can serve it without a cold download at startup.
#
# Runs `hf download` *inside* the vllm-spark image, so the host needs nothing
# installed (the image already ships the `hf` CLI + huggingface_hub).
#
# Usage:
#   ./download-qwen3.6-27b-nvfp4.sh
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

# unsloth's compressed-tensors NVFP4 quant of the dense, multimodal qwen3_5
# checkpoint (~16 GB). Unlike the DFlash preset there is no separate drafter —
# spec-decode uses the model's built-in MTP head. Override to pin a different remix.
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.6-27B-NVFP4}"

echo ">>> Downloading ${MODEL_REPO} into ${HF_HOME}"
# Allocate a TTY only when we have one (progress bars), so this also works
# headless / piped / in the background.
tty_flags=(-i); [[ -t 1 ]] && tty_flags+=(-t)
exec docker run --rm "${tty_flags[@]}" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  --entrypoint bash "$IMAGE" -c '
    set -e
    repo="$1"
    echo ">>> hf download ${repo}"
    hf download "$repo"
    echo ">>> done. cached repos:"
    du -sh /root/.cache/huggingface/hub/models--* 2>/dev/null || true
  ' _ "$MODEL_REPO"
