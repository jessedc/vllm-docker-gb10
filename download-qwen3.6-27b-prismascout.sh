#!/usr/bin/env bash
# Pre-fetch the Qwen3.6-27B *dense* checkpoint + the DFlash drafter into the HF
# cache, so run-qwen3.6-27b-prismascout.sh can serve them without a cold download at startup.
#
# Runs `hf download` *inside* the vllm-spark image, so the host needs nothing
# installed (the image already ships the `hf` CLI + huggingface_hub).
#
# Usage:
#   ./download-qwen3.6-27b-prismascout.sh
#
# Env:
#   IMAGE         image to use            (default: vllm-spark:latest)
#   HF_HOME       host HF cache dir       (default: ~/.cache/huggingface)
#   HF_TOKEN      token for gated repos   (these are public, usually unneeded)
#   MODEL_REPO    override the body repo
#   DRAFTER_REPO  override the drafter repo
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# PrismaSCOUT mixed NVFP4+BF16 body (~20 GB) and the z-lab DFlash block-diffusion
# drafter (~3.5 GB). They are co-designed: the small body buys the VRAM the
# drafter spends. Override either var to pin a different community remix.
MODEL_REPO="${MODEL_REPO:-rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm}"
DRAFTER_REPO="${DRAFTER_REPO:-z-lab/Qwen3.6-27B-DFlash}"

echo ">>> Downloading into ${HF_HOME} (model + drafter, ~24 GB total)"
# Allocate a TTY only when we have one (progress bars), so this also works
# headless / piped / in the background.
tty_flags=(-i); [[ -t 1 ]] && tty_flags+=(-t)
exec docker run --rm "${tty_flags[@]}" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  --entrypoint bash "$IMAGE" -c '
    set -e
    for repo in "$@"; do
      echo ">>> hf download ${repo}"
      hf download "$repo"
    done
    echo ">>> done. cached repos:"
    du -sh /root/.cache/huggingface/hub/models--* 2>/dev/null || true
  ' _ "$MODEL_REPO" "$DRAFTER_REPO"
