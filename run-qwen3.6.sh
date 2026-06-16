#!/usr/bin/env bash
# Serve nvidia/Qwen3.6-35B-A3B-NVFP4 on the DGX Spark (GB10 / sm_121a) using the
# locally built from-source vLLM image (vllm-spark:latest).
#
# Usage:
#   ./run-qwen3.6.sh                 # foreground (Ctrl-C to stop)
#   ./run-qwen3.6.sh --mtp           # enable MTP speculative decoding (off by default)
#   DETACH=1 ./run-qwen3.6.sh        # background server, restarts on boot
#   ./run-qwen3.6.sh --max-num-seqs 8   # append/override any vllm serve flag
#
# Env: IMAGE, PORT (host), HF_TOKEN, HF_HOME.
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
MODEL="nvidia/Qwen3.6-35B-A3B-NVFP4"
PORT="${PORT:-8000}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

# --- arg parsing -----------------------------------------------------------
# Pull our own flags out of the argument list; everything else passes through
# to `vllm serve` unchanged.
MTP=0
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mtp) MTP=1; shift ;;
    *)     passthrough+=("$1"); shift ;;
  esac
done

# --- vllm serve options ----------------------------------------------------
# Deduped from the requested set:
#   * removed  --dtype auto         -> "auto" is already the default
#   * removed  --quantization modelopt
#       The checkpoint's config.json carries a quantization_config (mixed
#       W4A16_NVFP4 + FP8), so vLLM auto-detects it. Forcing "modelopt" selects
#       the FP8 path, which is wrong for this NVFP4 model. Let auto-detect pick
#       the NVFP4 (modelopt_fp4) path. To force it explicitly instead, you'd use
#       --quantization modelopt_fp4 (not modelopt).
vllm_args=(
  "$MODEL"
  --port 8000
  --trust-remote-code
  --kv-cache-dtype fp8
  --attention-backend flashinfer
  --moe-backend marlin
  --gpu-memory-utilization 0.65
  --max-model-len 262144
  --max-num-seqs 4
  --max-num-batched-tokens 32768
  --enable-auto-tool-choice
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --enable-chunked-prefill
  --async-scheduling
  --enable-prefix-caching
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}'
)

# MTP speculative decoding is opt-in via --mtp.
if [[ "$MTP" == 1 ]]; then
  vllm_args+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}')
fi

# --- docker run ------------------------------------------------------------
docker_flags=(--gpus all --ipc=host -p "${PORT}:8000"
              -e "HF_TOKEN=${HF_TOKEN:-}"
              -v "${HF_HOME}:/root/.cache/huggingface")

if [[ "${DETACH:-0}" == 1 ]]; then
  docker_flags+=(-d --name qwen36 --restart unless-stopped)
else
  docker_flags+=(--rm -it)
fi

set -x
exec docker run "${docker_flags[@]}" "$IMAGE" "${vllm_args[@]}" "${passthrough[@]+"${passthrough[@]}"}"
