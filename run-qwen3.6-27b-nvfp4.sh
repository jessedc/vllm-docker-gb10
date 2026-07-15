#!/usr/bin/env bash
# Serve unsloth/Qwen3.6-27B-NVFP4 on the DGX Spark (GB10 / sm_121a) from the
# locally built from-source vLLM image (vllm-spark:latest), following unsloth's
# DGX Spark guide:
#   https://unsloth.ai/docs/models/qwen3.6#dgx-spark-with-nvfp4-quants
#
# This is the unsloth-NVFP4 sibling of run-qwen3.6-27b.sh (which serves the
# rdtand PrismaSCOUT body + z-lab DFlash drafter). The two are DIFFERENT 27B
# models: this one is unsloth's own compressed-tensors NVFP4 quant of the dense,
# *multimodal* qwen3_5 checkpoint, and it uses the model's built-in MTP head
# instead of an external DFlash drafter.
#
# NVFP4 is auto-detected from config.json (quant_method=compressed-tensors), so we
# pass NO --quantization -- matching the guide command and the 35B/gemma4 presets.
# The attention backend is left to vLLM's auto-pick: this model is multimodal, and
# forcing --attention-backend flashinfer triggers the same rejection gemma4 hits
# ("partial multimodal token full attention not supported").
#
# b12x REALITY-CHECK (measured 2026-07-15, not just the guide's wording):
#   * This checkpoint is DENSE (no MoE layers), so the guide's
#     `--moe-backend flashinfer_b12x` is a NO-OP here. It is kept below only for
#     guide parity / a future MoE variant -- it boots fine but changes nothing.
#   * The dense NVFP4 GEMM auto-selects FlashInferCutlassNvFp4LinearKernel
#     (cutlass). That is the best available dense path -- NOT the marlin W4A16
#     worst case the guide warns about. There is no faster option to force:
#     `--linear-backend flashinfer_b12x` HARD-FAILS on boot in this build
#     ("no 'flashinfer_b12x' kernel exists for this layer type").
#   * env CUTE_DSL_ARCH=sm_121a is set per the guide (harmless; only bites the
#     cute-DSL kernels where they actually apply).
#   * Single-stream ~11 tok/s (no MTP) is the Spark's memory-bandwidth ceiling
#     for a 27B, not a misconfig. The built-in MTP head measured +79%
#     (11.3 -> 20.2 tok/s) with tool calling intact, so it is the DEFAULT here;
#     pass --no-spec to disable it.
# Confirm the image has the b12x (MoE) kernels at all:
#   docker run --rm --gpus all --entrypoint python3 vllm-spark:latest -c \
#     "import torch; from vllm.utils.flashinfer import has_flashinfer_b12x_gemm as g, \
#      has_flashinfer_b12x_moe as m; print(torch.cuda.get_device_capability(), g(), m())"
#
# Run ./download-qwen3.6-27b-nvfp4.sh once first to populate the HF cache.
#
# !! UNIFIED-MEMORY SAFETY (learned the hard way — see logs/crash-*.vllm.log) !!
# The Spark's 121 GB is shared by GPU *and* host. On a model's FIRST boot, flashinfer
# JIT-compiles its FP4 GEMM kernels, and by default spawns one `cicc` per core (20),
# ~5.8 GB each. Stacked on vLLM's memory reservation that can blow past 121 GB and
# trigger a *global* OOM that kills the desktop — the machine has to be power-cycled.
# We defend against this three ways, all on by default:
#   1. COMPILE_JOBS (MAX_JOBS) caps concurrent compilers to 2 (~12 GB spike).
#   2. CACHE_HOME persists /root/.cache so the compile happens ONCE, then is reused.
#   3. MEM_LIMIT hard-caps the container so any runaway is killed inside the cgroup
#      instead of taking down the host.
# After the first successful (cache-warming) boot you can safely raise MAX_MODEL_LEN.
#
# Usage:
#   ./run-qwen3.6-27b-nvfp4.sh              # foreground, MTP spec decode (DEFAULT; measured +79%)
#   ./run-qwen3.6-27b-nvfp4.sh --no-spec    # disable MTP -> plain autoregressive decode
#   ./run-qwen3.6-27b-nvfp4.sh --mtp        # explicit MTP (same as default)
#   DETACH=1 ./run-qwen3.6-27b-nvfp4.sh     # background server (RESTART=no by default)
#   ./run-qwen3.6-27b-nvfp4.sh --max-num-seqs 8   # append/override any vllm serve flag
#
# Env: IMAGE, PORT, HF_TOKEN, HF_HOME, GPU_MEM_UTIL, MAX_NUM_SEQS, MAX_MODEL_LEN,
#      SPEC_TOKENS, CHAT_TEMPLATE, CACHE_HOME, LOG_DIR, COMPILE_JOBS, MEM_LIMIT,
#      RESTART, VLLM_LOG_LEVEL, AUTOTUNE.
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
MODEL="${MODEL_REPO:-unsloth/Qwen3.6-27B-NVFP4}"
PORT="${PORT:-8000}"
NAME="${NAME:-qwen36-27b-nvfp4}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Persist flashinfer/torch.compile caches across containers so the expensive (and
# OOM-prone) first-boot JIT compile only happens once. HF cache nests inside it.
CACHE_HOME="${CACHE_HOME:-$HOME/.cache/vllm-spark}"
mkdir -p "$HF_HOME" "$CACHE_HOME"

# --- arg parsing -----------------------------------------------------------
# Spec-decode mode: mtp (DEFAULT) | none. MTP is on by default because it measured
# +79% decode throughput (11.3->20.2 tok/s) with tool calling intact; --no-spec
# turns it off. Everything we don't recognise passes straight through to `vllm serve`.
SPEC=mtp
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mtp)     SPEC=mtp;   shift ;;
    --no-spec) SPEC=none;  shift ;;
    *)         passthrough+=("$1"); shift ;;
  esac
done

# The built-in MTP head adds a draft pass per step, so cap utilisation lower to
# leave headroom on the unified pool; plain decode can run a little higher.
case "$SPEC" in
  mtp)  MEM_UTIL="${GPU_MEM_UTIL:-0.52}" ;;
  none) MEM_UTIL="${GPU_MEM_UTIL:-0.72}" ;;
esac

# --- vllm serve options ----------------------------------------------------
# NOTE: no --quantization (compressed-tensors NVFP4 auto-detected) and no
# --attention-backend (auto-pick; see header for why forcing flashinfer breaks
# this multimodal model).
vllm_args=(
  "$MODEL"
  --served-model-name qwen/qwen3.6-27b-nvfp4
  --port 8000
  --trust-remote-code
  --moe-backend flashinfer_b12x           # guide parity only: NO-OP on this dense model (no MoE layers)
  --kv-cache-dtype auto
  --mamba-block-size 256                  # Qwen3.6 has GDN/linear-attention layers
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-6}"
  --max-num-batched-tokens 32768
  --gpu-memory-utilization "$MEM_UTIL"
  --enable-chunked-prefill
  --enable-prefix-caching
  --load-format fastsafetensors           # confirmed installed in the image
  --limit-mm-per-prompt '{"image":4,"video":2}'   # checkpoint carries image+video tokens
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --generation-config vllm
  --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":40,"presence_penalty":0.0,"repetition_penalty":1.0}'
)

# flashinfer's fp4_gemm autotuner allocates large GEMM workspaces that are NOT
# counted in --gpu-memory-utilization; on the Spark's unified memory they stack on
# top of the KV reservation and OOM the host. Disabling it costs a little GEMM
# throughput but lets us safely run at ~0.72 util. Re-enable with AUTOTUNE=1 (only
# at low util, e.g. GPU_MEM_UTIL<=0.4).
if [[ "${AUTOTUNE:-0}" != 1 ]]; then
  vllm_args+=(--no-enable-flashinfer-autotune)
fi

# Spec-decode config. mtp uses the model's in-checkpoint MTP head
# (mtp_num_hidden_layers=1). num_speculative_tokens=2 matches the unsloth card.
case "$SPEC" in
  mtp)  vllm_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-2}}") ;;
  none) : ;;
esac

# --- docker run ------------------------------------------------------------
# Cache mounts: bind /root/.cache to a persistent host dir, then nest the HF cache
# inside it (nested bind takes precedence for that subpath). This keeps the
# flashinfer + vllm torch_compile caches between runs so the JIT compile is paid once.
docker_flags=(--gpus all --ipc=host -p "${PORT}:8000"
              -e "HF_TOKEN=${HF_TOKEN:-}"
              # Guide-critical: select the Blackwell cute-DSL kernels (else ~2x slower).
              -e "CUTE_DSL_ARCH=sm_121a"
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
