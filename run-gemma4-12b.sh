#!/usr/bin/env bash
# Serve RedHatAI/gemma-4-12B-it-NVFP4 on the DGX Spark (GB10 / sm_121a) using the
# locally built from-source vLLM image (vllm-spark:latest).
#
# Gemma 4 12B is the "unified" (encoder-free) multimodal model: text + image +
# audio in, projected straight into the LM. This RedHatAI checkpoint quantizes
# the Linear weights+activations to NVFP4 (compressed-tensors "nvfp4-pack-
# quantized"); the vision/audio embedders are left unquantized.
#
# Usage:
#   ./run-gemma4-12b.sh                       # foreground (Ctrl-C to stop)
#   ./run-gemma4-12b.sh --no-tools            # disable tool-calling / reasoning parsing
#   DETACH=1 ./run-gemma4-12b.sh              # background server, restarts on boot
#   ./run-gemma4-12b.sh --max-num-seqs 8      # append/override any vllm serve flag
#
# Env: IMAGE, PORT (host), HF_TOKEN, HF_HOME, GPU_MEM_UTIL, MAX_MODEL_LEN.
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
MODEL="RedHatAI/gemma-4-12B-it-NVFP4"
PORT="${PORT:-8000}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- arg parsing -----------------------------------------------------------
# Pull our own flags out of the argument list; everything else passes through
# to `vllm serve` unchanged.
TOOLS=1
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-tools) TOOLS=0; shift ;;
    *)          passthrough+=("$1"); shift ;;
  esac
done

# --- vllm serve options ----------------------------------------------------
# Notes (Gemma 4 12B NVFP4 on this Spark image):
#   * NO --quantization flag. config.json carries a compressed-tensors
#     "nvfp4-pack-quantized" block, so vLLM auto-detects the NVFP4 path. Forcing
#     a quant method here would pick the wrong kernel (same trap as the qwen3.6
#     preset).
#   * --kv-cache-dtype fp8 — Blackwell has the kernels; halves KV footprint so
#     the 256K context window actually fits.
#   * NO forced --attention-backend. The unified model uses bidirectional
#     multimodal ("partial multimodal token full") attention, which FlashInfer's
#     full-attention path rejects; let vLLM pick the backend per layer.
#   * max_position_embeddings is 262144; the model uses 1024-token sliding-window
#     attention on most layers, so the long-context KV cost stays bounded.
#   * --limit-mm-per-prompt caps multimodal items so vLLM can size the encoder
#     cache; this checkpoint accepts images and audio.
vllm_args=(
  "$MODEL"
  --port 8000
  --kv-cache-dtype fp8
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.65}"
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --limit-mm-per-prompt '{"image":4,"audio":1}'
  --enable-chunked-prefill
  --enable-prefix-caching
  --async-scheduling
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":64,"min_p":0.0}'
)

# Tool calling + reasoning are on by default. They need the gemma4 parsers
# (registered in this build) plus the bundled tool chat template, which lives in
# the cloned vLLM source inside the image at /opt/vllm/examples/. --no-tools
# skips all three (e.g. for a pure multimodal / plain-chat server).
if [[ "$TOOLS" == 1 ]]; then
  vllm_args+=(
    --enable-auto-tool-choice
    --reasoning-parser gemma4
    --tool-call-parser gemma4
    --chat-template /opt/vllm/examples/tool_chat_template_gemma4.jinja
  )
fi

# --- docker run ------------------------------------------------------------
docker_flags=(--gpus all --ipc=host -p "${PORT}:8000"
              -e "HF_TOKEN=${HF_TOKEN:-}"
              -v "${HF_HOME}:/root/.cache/huggingface")

if [[ "${DETACH:-0}" == 1 ]]; then
  docker_flags+=(-d --name gemma4 --restart unless-stopped)
else
  docker_flags+=(--rm -it)
fi

set -x
exec docker run "${docker_flags[@]}" "$IMAGE" "${vllm_args[@]}" "${passthrough[@]+"${passthrough[@]}"}"
