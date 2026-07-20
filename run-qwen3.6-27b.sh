#!/usr/bin/env bash
# Serve the Qwen3.6-27B *dense* model on the DGX Spark (GB10 / sm_121a) from the
# locally built from-source vLLM image (vllm-spark:latest), with DFlash
# speculative decoding by default.
#
# This is the 27B-dense sibling of run-qwen3.6.sh (which serves the 35B-A3B MoE).
# Translated from a community docker-compose (qwen36-27b-notes.md), with two
# deliberate changes for *our* mainline-from-source image:
#   * aeon-vllm-ultimate-only env vars dropped — VLLM_NVFP4_GEMM_BACKEND,
#     ENABLE_NVFP4_SM100, *TURBOQUANT*, VLLM_USE_FLASHINFER_MOE_FP4 are not read
#     by mainline vLLM (they are patches in that 3rd-party build), so they are
#     no-ops for us and only invite false confidence.
#   * multimodal flags dropped — the rdtand PrismaSCOUT checkpoint is text-only,
#     so --limit-mm-per-prompt / --mm-* would error.
# DFlash + the PrismaSCOUT body are co-designed: the small NVFP4 body reclaims the
# VRAM that the block-diffusion drafter spends on block verification.
#
# Run ./download-qwen3.6-27b.sh once first to populate the HF cache.
#
# !! UNIFIED-MEMORY SAFETY (learned the hard way — see logs/crash-*.vllm.log) !!
# The Spark's 121 GB is shared by GPU *and* host. On a model's FIRST boot, flashinfer
# JIT-compiles its FP4 GEMM kernels, and by default spawns one `cicc` per core (20),
# ~5.8 GB each. Stacked on vLLM's ~0.65*121=79 GB reservation that blows past 121 GB
# and triggers a *global* OOM that kills the desktop — the machine has to be power-
# cycled. We defend against this three ways, all on by default:
#   1. COMPILE_JOBS (MAX_JOBS) caps concurrent compilers to 2 (~12 GB spike).
#   2. CACHE_HOME persists /root/.cache so the compile happens ONCE, then is reused.
#   3. MEM_LIMIT hard-caps the container so any runaway is killed inside the cgroup
#      instead of taking down the host.
# After the first successful (cache-warming) boot you can safely raise MAX_MODEL_LEN.
#
# Usage:
#   ./run-qwen3.6-27b.sh                 # foreground, DFlash spec decode (default)
#   ./run-qwen3.6-27b.sh --mtp           # use the model's built-in MTP head instead
#   ./run-qwen3.6-27b.sh --no-spec       # plain autoregressive decode, no drafter
#   DETACH=1 ./run-qwen3.6-27b.sh        # background server (RESTART=no by default)
#   ./run-qwen3.6-27b.sh --max-num-seqs 8   # append/override any vllm serve flag
#
# Env: IMAGE, PORT, HF_TOKEN, HF_HOME, GPU_MEM_UTIL, MAX_NUM_SEQS, MAX_MODEL_LEN,
#      SPEC_TOKENS, CHAT_TEMPLATE, CACHE_HOME, LOG_DIR, COMPILE_JOBS, MEM_LIMIT,
#      RESTART, VLLM_LOG_LEVEL.
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
MODEL="${MODEL_REPO:-rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm}"
DRAFTER="${DRAFTER_REPO:-z-lab/Qwen3.6-27B-DFlash}"
PORT="${PORT:-8000}"
NAME="${NAME:-qwen36-27b}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Persist flashinfer/torch.compile caches across containers so the expensive (and
# OOM-prone) first-boot JIT compile only happens once. HF cache nests inside it.
CACHE_HOME="${CACHE_HOME:-$HOME/.cache/vllm-spark}"
mkdir -p "$HF_HOME" "$CACHE_HOME"

# --- arg parsing -----------------------------------------------------------
# Spec-decode mode: dflash (default) | mtp | none. Everything we don't recognise
# passes straight through to `vllm serve`.
# --no-reasoning-parser: drop `--reasoning-parser qwen3` so vLLM returns the raw
# thinking verbatim in `content` (for a non-tool-calling client that splits it
# itself, e.g. opencode). Mirrors run-qwen3.6-27b-nvfp4.sh.
# !! DO NOT use this to "fix" the runaway-think truncation for tool calling: it
#    makes tool calling WORSE, not better. Measured (benchmarks/qwen3.6-27b-tool-
#    calling.md): without the reasoning channel the model rambles its plan in prose
#    and often never emits the structured tool call — tool-selection accuracy fell
#    93% -> 47% and tool calls halved. Keep the parser ON for agentic use; raise
#    the per-turn token budget instead if a long think is truncating.
SPEC=dflash
NO_REASONING=0
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dflash)              SPEC=dflash; shift ;;
    --mtp)                 SPEC=mtp;    shift ;;
    --no-spec)             SPEC=none;   shift ;;
    --no-reasoning-parser) NO_REASONING=1; shift ;;
    *)                     passthrough+=("$1"); shift ;;
  esac
done

# DFlash inflates the KV cache at long context (it verifies whole token blocks),
# so cap utilisation well below the 0.85 we use elsewhere. The built-in MTP head
# is lighter, but the source notes still run it conservatively at long context.
case "$SPEC" in
  dflash) MEM_UTIL="${GPU_MEM_UTIL:-0.65}" ;;
  mtp)    MEM_UTIL="${GPU_MEM_UTIL:-0.52}" ;;
  none)   MEM_UTIL="${GPU_MEM_UTIL:-0.72}" ;;
esac

# --- vllm serve options ----------------------------------------------------
vllm_args=(
  "$MODEL"
  --served-model-name qwen/qwen3.6-27b
  --port 8000
  --trust-remote-code
  --quantization compressed-tensors      # PrismaSCOUT body is compressed-tensors NVFP4
  --kv-cache-dtype auto
  --attention-backend flashinfer
  --mamba-block-size 256                  # Qwen3.6 has GDN/linear-attention layers
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-6}"
  --max-num-batched-tokens 32768
  --gpu-memory-utilization "$MEM_UTIL"
  --enable-chunked-prefill
  --enable-prefix-caching
  --load-format fastsafetensors           # confirmed installed in the image
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --generation-config vllm
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":40,"presence_penalty":0.0,"repetition_penalty":1.0}'
)

# flashinfer's fp4_gemm autotuner allocates large GEMM workspaces that are NOT
# counted in --gpu-memory-utilization; on the Spark's unified memory they stack on
# top of the KV reservation and OOM the host (~38 GiB transient spike — it crashed
# the box twice). Disabling it costs a little GEMM throughput but lets us safely run
# at ~0.65 util. Re-enable with AUTOTUNE=1 (only at low util, e.g. GPU_MEM_UTIL<=0.4).
if [[ "${AUTOTUNE:-0}" != 1 ]]; then
  vllm_args+=(--no-enable-flashinfer-autotune)
fi

# Reasoning parser is ON by default (strips <think>...</think> from `content`).
# --no-reasoning-parser omits it so the raw thinking is returned verbatim in content.
if [[ "$NO_REASONING" != 1 ]]; then
  vllm_args+=(--reasoning-parser qwen3)
fi

# Spec-decode config. dflash pulls the external drafter; mtp uses the in-model head.
# SPEC_TOKENS=10 matches the source notes; forum post #32 found 8 the sweet spot
# and 15 too memory-hungry ("Spark died under it").
case "$SPEC" in
  dflash) vllm_args+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFTER}\",\"num_speculative_tokens\":${SPEC_TOKENS:-10}}") ;;
  mtp)    vllm_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-3}}") ;;
  none)   : ;;
esac

# --- docker run ------------------------------------------------------------
# Cache mounts: bind /root/.cache to a persistent host dir, then nest the HF cache
# inside it (nested bind takes precedence for that subpath). This keeps the
# flashinfer + vllm torch_compile caches between runs so the JIT compile is paid once.
docker_flags=(--gpus all --ipc=host -p "${PORT}:8000"
              -e "HF_TOKEN=${HF_TOKEN:-}"
              -v "${CACHE_HOME}:/root/.cache"
              -v "${HF_HOME}:/root/.cache/huggingface"
              # Bound the first-boot JIT compiler so it can't OOM the unified pool.
              -e "MAX_JOBS=${COMPILE_JOBS:-2}"
              -e "FLASHINFER_NVCC_THREADS=1"
              # Persist flashinfer's fp4_gemm autotune tactics. Unset, vLLM re-runs the
              # autotuner every boot, and its GEMM workspace (NOT counted in
              # gpu-memory-utilization) stacks on top of the KV reservation and OOMs the
              # unified pool. Pointed at the mounted cache, the autotune runs once.
              -e "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/flashinfer_autotune"
              -e "VLLM_LOGGING_LEVEL=${VLLM_LOG_LEVEL:-INFO}")

# Hard memory backstop so a runaway is killed in the container cgroup, not as a
# host-wide global OOM that crashes the desktop. Set MEM_LIMIT= (empty) to disable.
MEM_LIMIT="${MEM_LIMIT:-112g}"
if [[ -n "$MEM_LIMIT" ]]; then
  docker_flags+=(--memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT")
fi

# Optional community chat-template fix: CHAT_TEMPLATE=/host/path/to/template.jinja
if [[ -n "${CHAT_TEMPLATE:-}" ]]; then
  docker_flags+=(-v "${CHAT_TEMPLATE}:/chat-template.jinja:ro")
  vllm_args+=(--chat-template /chat-template.jinja)
fi

# --- logging ---------------------------------------------------------------
LOG_DIR="${LOG_DIR:-$(cd "$(dirname "$0")" && pwd)/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
VLLM_LOG="$LOG_DIR/${NAME}-${STAMP}.log"
MEM_LOG="$LOG_DIR/${NAME}-${STAMP}.mem.csv"

# Host-memory sampler — the single most useful artefact if the box OOMs during the
# first-boot compile. Records MemAvailable / SwapFree / summed compiler RSS every 2s,
# fsync'd each tick so the trace survives a hard crash. Runs until the container exits.
start_memtrace() {
  nohup bash -c '
    name="$1"; out="$2"
    echo "time,MemAvailable_kB,SwapFree_kB,compiler_rss_kB" > "$out"
    for _ in $(seq 60); do docker ps --format "{{.Names}}" | grep -qx "$name" && break; sleep 1; done
    while docker ps --format "{{.Names}}" | grep -qx "$name"; do
      ma=$(awk "/^MemAvailable:/{print \$2}" /proc/meminfo)
      sf=$(awk "/^SwapFree:/{print \$2}" /proc/meminfo)
      cc=$(ps -eo rss,comm 2>/dev/null | awk "/cicc|nvcc|ptxas|cudafe|c\\+\\+filt/{s+=\$1} END{print s+0}")
      printf "%s,%s,%s,%s\n" "$(date +%H:%M:%S)" "$ma" "$sf" "$cc" >> "$out"
      sync "$out" 2>/dev/null || true
      sleep 2
    done
  ' _ "$NAME" "$MEM_LOG" >/dev/null 2>&1 &
  disown
}

echo ">>> vLLM log:   $VLLM_LOG"
echo ">>> mem trace:  $MEM_LOG"
echo ">>> compile:    MAX_JOBS=${COMPILE_JOBS:-2}  mem-cap=${MEM_LIMIT:-none}  cache=$CACHE_HOME"

if [[ "${DETACH:-0}" == 1 ]]; then
  # RESTART defaults to "no": a model that can OOM the host on a cold compile must
  # not auto-restart into a crash loop after a reboot. Opt back in with RESTART=unless-stopped.
  docker_flags+=(-d --name "$NAME" --restart "${RESTART:-no}")
  run=(docker run "${docker_flags[@]}" "$IMAGE" "${vllm_args[@]}" "${passthrough[@]+"${passthrough[@]}"}")
  set -x
  "${run[@]}"
  { set +x; } 2>/dev/null
  start_memtrace
  nohup docker logs -f --since 0m "$NAME" >>"$VLLM_LOG" 2>&1 & disown
  echo ">>> detached as '$NAME'. Follow: docker logs -f $NAME   (also streaming to $VLLM_LOG)"
else
  docker_flags+=(--rm -i --name "$NAME")     # no -t: stdout is piped to tee
  run=(docker run "${docker_flags[@]}" "$IMAGE" "${vllm_args[@]}" "${passthrough[@]+"${passthrough[@]}"}")
  start_memtrace
  set -x
  "${run[@]}" 2>&1 | tee "$VLLM_LOG"
fi
