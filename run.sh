#!/usr/bin/env bash
# Serve a model with the locally built vLLM image on the DGX Spark.
#
# Usage:
#   ./run.sh                                  # serves a small default model
#   ./run.sh Qwen/Qwen3-8B                    # serve a specific HF model
#   ./run.sh <model> --max-model-len 32768    # extra flags pass through to `vllm serve`
#
# Env:
#   IMAGE          image to run            (default: vllm-spark:latest)
#   HF_TOKEN       Hugging Face token for gated/private models
#   HF_HOME        host HF cache dir       (default: ~/.cache/huggingface)
#   PORT           host port               (default: 8000)
#   GPU_MEM_UTIL   --gpu-memory-utilization (default: 0.85, suits unified memory)
#   MAX_NUM_SEQS   --max-num-seqs          (default: 8)
#   DETACH=1       run detached + restart (server mode) instead of interactive
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
MODEL="${1:-Qwen/Qwen3-4B}"; shift || true
PORT="${PORT:-8000}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"

run_flags=(--gpus all --ipc=host -p "${PORT}:8000"
           -e "HF_TOKEN=${HF_TOKEN:-}"
           -v "${HF_HOME}:/root/.cache/huggingface")

if [[ "${DETACH:-0}" == 1 ]]; then
  run_flags+=(-d --name vllm --restart unless-stopped)
else
  run_flags+=(--rm -it)
fi

set -x
exec docker run "${run_flags[@]}" \
  "$IMAGE" \
  "$MODEL" \
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}" \
  --max-num-seqs "${MAX_NUM_SEQS:-8}" \
  "$@"
