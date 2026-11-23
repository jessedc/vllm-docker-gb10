#!/usr/bin/env bash
# Serve thinkingmachines/Inkling-2-Mini on the DGX Spark (GB10 / sm_121a).
#   https://huggingface.co/thinkingmachines/Inkling-2-Mini
#   upstream recipe: recipes/models/thinkingmachines/Inkling-2-Mini.yaml (vLLM main)
#
# WHAT THIS CHECKPOINT IS
#   Looped multimodal MoE (text+image in, text out), arch
#   Inkling2LoopForConditionalGeneration / model_type inkling2_loop: 26.4B total,
#   ~3.5B active, hidden 2304, 24 layers (layers 6-17 run 3 passes, each pass with
#   its own KV + conv-state cache), 128 routed experts top-8, 20 sliding-window
#   (1024) + 4 full-attention layers, 1M native context. ModelOpt NVFP4 experts
#   (layers 0-19) + MXFP8 (20-23); attention/routers/lm_head BF16. Bundles a DSpark
#   speculator (mtp.safetensors). Load AS RELEASED -- never re-quantise.
#
# MEASURED ON THIS BOX (image 647a7686, defaults below)
#   load 18.24 GiB, 703,218 KV tokens (10.7x at 64k), ready in ~4 min (warm cache)
#   decode 39.5 tok/s single stream; 48,665-token prompt: needle at 40% depth
#   retrieved exactly, ~1.5k tok/s prefill, host MemAvailable flat at ~31 GiB.
#   Non-streamed tool call parses cleanly. MoE takes the native moe_sm120 path
#   (GB10 is capability family 120) -- it logs no line for that, so don't hunt.
#   Reasoning arrives on the `reasoning` field (inkling parser).
#
# IMAGE / VERSION
#   No stable vLLM contains it (PR #69028 merged 2026-11-21, after v0.34.0). This
#   preset expects the from-source nightly build:
#     ./build.sh --vllm-ref 647a76866b4559a74e27d1c6b55fbffe85c07ce4
#   (= the commit vllm/vllm-openai:nightly-aarch64 was built from -> exact
#   base/source lockstep; includes #69168 sm120 small-M GEMM + #69021 SM12x TMA).
#   IMAGE=vllm/vllm-openai:inkling-2-mini also works (pre-merge snapshot).
#
# FLAGS DELIBERATELY *NOT* PASSED
#   --quantization       : hf_quant_config.json / config.json (modelopt NVFP4 +
#                          per-layer MXFP8) is auto-detected; forcing one scheme
#                          would flatten the mixed allocation.
#   --attention-backend  : the model ships its own relative-bias attention ops.
#
# KNOWN OPEN ISSUES (as of 2026-11-22)
#   #69181  ~3 GB of over-reserved bf16 conv state per request group on SM12x;
#           keep --max-num-seqs low (default 8).
#   #69267 / #69241  streaming tool calls: the inkling tool parser re-parses the
#           whole argument per delta and/or streams arguments as ''. Use
#           stream=false for tool-call turns.
#
# Run ./download-inkling-2-mini.sh once first to populate the HF cache.
#
# Usage:
#   ./run-inkling-2-mini.sh                   # foreground, reasoning + tools
#   ./run-inkling-2-mini.sh --dspark          # + DSpark spec decode (measured slower, see below)
#   ./run-inkling-2-mini.sh --no-tools        # drop tool-call parser
#   ./run-inkling-2-mini.sh --enforce-eager   # smoke test (any vllm serve flag passes through)
#   DETACH=1 ./run-inkling-2-mini.sh          # background server
#
# Env: IMAGE, PORT, HF_TOKEN, HF_HOME, GPU_MEM_UTIL, MAX_NUM_SEQS, MAX_MODEL_LEN,
#      CACHE_HOME, LOG_DIR, COMPILE_JOBS, MEM_LIMIT, RESTART,
#      VLLM_LOG_LEVEL, AUTOTUNE.
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
MODEL="${MODEL_REPO:-thinkingmachines/Inkling-2-Mini}"
PORT="${PORT:-8000}"
NAME="${NAME:-inkling-2-mini}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
CACHE_HOME="${CACHE_HOME:-$HOME/.cache/vllm-spark}"
mkdir -p "$HF_HOME" "$CACHE_HOME"

# --- arg parsing -----------------------------------------------------------
SPEC=none
TOOLS=1
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dspark)   SPEC=dspark; shift ;;
    --no-spec)  SPEC=none;   shift ;;
    --no-tools) TOOLS=0;     shift ;;
    *)          passthrough+=("$1"); shift ;;
  esac
done

# --- vllm serve options ----------------------------------------------------
# gpu-memory-utilization MUST be < 1.0 here: on CUDA the unset default is now 1.0
# (extensible KV cache), and the docs single out unified-memory devices (GB10)
# as the case to lower it -- the CPU/OS share the pool.
vllm_args=(
  "$MODEL"
  --port 8000
  # 262144 = upstream's Spark cap (native is 1M). Cheap here: 20/24 layers are
  # 1024-token SWA, so only 4 global layers hold full-length KV -- at 0.70 util
  # vLLM reports 1,384,695 KV tokens = 5.28x concurrency at 262144 (measured).
  # 1M is not attempted: unified-memory OOM risk. Override via MAX_MODEL_LEN.
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  # Decode-bound MoE: big batches don't help and #69181's conv-state over-reserve
  # scales with them.
  --max-num-seqs "${MAX_NUM_SEQS:-8}"
  --gpu-memory-utilization "${GPU_MEM_UTIL:-0.70}"
  --enable-prefix-caching
  --limit-mm-per-prompt '{"image":4}'
  --reasoning-parser inkling
)
if [[ "$TOOLS" == 1 ]]; then
  vllm_args+=(--enable-auto-tool-choice --tool-call-parser inkling)
fi

# See UNIFIED-MEMORY SAFETY below. Re-enable the flashinfer autotuner only at low
# util, where its uncounted GEMM workspace fits.
if [[ "${AUTOTUNE:-0}" != 1 ]]; then
  vllm_args+=(--no-enable-flashinfer-autotune)
fi

# DSpark: the 3-layer parallel block drafter + recurrent head bundled in the
# checkpoint; upstream recipe values. OPT-IN because it MEASURED SLOWER here
# (image 647a7686, 300-token single stream, essay prompt, warm):
#   no spec          39.5 tok/s  (temp 0)   703,218 KV tokens
#   DSpark n=7       36.0 tok/s  (temp 0)   598,148 KV tokens (-15%)
#                    32-37 tok/s (temp 1)
# Mean acceptance length only 2.1-2.7 of 7 (per-pos 63/37/24/17/11/9/7%): with
# just ~3.5B active params the target step is already cheap, so the drafter's
# cost isn't repaid. n is NOT tunable -- vLLM rejects anything but 7 ("requires
# num_speculative_tokens = 7 (the checkpoint's block size - 1)"). Worth
# re-measuring under concurrency, where the recipe's GB200 numbers come from.
case "$SPEC" in
  dspark) vllm_args+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}') ;;
  none)   : ;;
esac

# !! UNIFIED-MEMORY SAFETY -- shared verbatim with the other presets: first-boot
# JIT compiles can drive the 121 GB shared pool into a host-wide OOM. Defences:
# capped compiler jobs, persistent /root/.cache, container memory cap, autotune off.
# --- docker run ------------------------------------------------------------
# Cache mounts: bind /root/.cache to a persistent host dir, then nest the HF cache
# inside it (nested bind takes precedence for that subpath). This keeps the
# flashinfer + vllm torch_compile caches between runs so the JIT compile is paid once.
docker_flags=(--gpus all --ipc=host -p "${PORT}:8000"
              -e "HF_TOKEN=${HF_TOKEN:-}"
              # Select the Blackwell cute-DSL kernels (else ~2x slower).
              -e "CUTE_DSL_ARCH=sm_121a"
              -v "${CACHE_HOME}:/root/.cache"
              -v "${HF_HOME}:/root/.cache/huggingface"
              # Bound the first-boot JIT compiler so it can't OOM the unified pool.
              -e "MAX_JOBS=${COMPILE_JOBS:-2}"
              -e "FLASHINFER_NVCC_THREADS=1"
              # Persist flashinfer's fp4_gemm autotune tactics so the autotune (and
              # its uncounted GEMM workspace) runs once rather than every boot.
              -e "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/root/.cache/flashinfer_autotune"
              -e "VLLM_LOGGING_LEVEL=${VLLM_LOG_LEVEL:-INFO}")

# Hard memory backstop so a runaway is killed in the container cgroup, not as a
# host-wide global OOM that crashes the desktop. Set MEM_LIMIT= (empty) to disable.
MEM_LIMIT="${MEM_LIMIT:-112g}"
if [[ -n "$MEM_LIMIT" ]]; then
  docker_flags+=(--memory "$MEM_LIMIT" --memory-swap "$MEM_LIMIT")
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
  # not auto-restart into a crash loop after a reboot. Opt back in with
  # RESTART=unless-stopped once the JIT/autotune cache is warm.
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
