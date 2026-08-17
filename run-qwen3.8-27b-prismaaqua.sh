#!/usr/bin/env bash
# Serve rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm on the DGX Spark (GB10 /
# sm_121a) from the locally built from-source vLLM image (vllm-spark:latest).
#   https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm
#
# WHAT THIS CHECKPOINT IS (and why it needs almost no flags)
#   PrismaQuant AQUA-AURA is a *mixed-precision* quant of Qwen/Qwen3.8-27B: every
#   Linear was priced individually and given the format it earned — 247 NVFP4
#   (W4A4, group 16, mostly MLPs), 223 FP8-E4M3 (per-channel, mostly attention
#   out-projections), 26 BF16 (the tiny linear_attn gate projections). 21.98 GiB
#   on disk, 5.5 bpp over the quantizable body. lm_head, embed_tokens, the whole
#   visual tower, the MTP head, norms and routers stay BF16.
#
#   It ships as STOCK compressed-tensors: config.json declares
#   `format: "mixed-precision"` at the top and a per-`config_group` `format`
#   (`nvfp4-pack-quantized` / `float-quantized`) underneath. vLLM's
#   CompressedTensorsConfig reads the per-group format and falls back to the
#   global one only where a group omits it, so the whole mixed allocation is
#   auto-detected. Pass NO `--quantization` — naming one would flatten every
#   layer onto a single scheme and is exactly wrong here.
#
#   Despite the "3.8" name the arch is `qwen3_5` / `Qwen3_5ForConditionalGeneration`
#   — the same family run-qwen3.6-27b-unsloth.sh serves — so this image's
#   transformers 5.15.0 and vLLM v0.27.1 both know it natively (no
#   --trust-remote-code, no transformers-from-source build).
#
# FLAGS DELIBERATELY *NOT* PASSED (each one is a trap or a no-op here)
#   --quantization            : see above; auto-detection is the correct path.
#   --attention-backend       : this checkpoint is MULTIMODAL (vision tower,
#                               image+video tokens). Forcing flashinfer hits the
#                               same rejection gemma4 and unsloth-Qwen3.6 hit —
#                               "partial multimodal token full attention not
#                               supported". Auto-pick is correct.
#   --moe-backend             : the model is DENSE. No MoE layers to back.
#   --override-generation-config : the repo ships its own generation_config.json
#                               (temperature 1.0 / top_k 20 / top_p 0.95, the
#                               Qwen3.8 defaults). vLLM picks that up on its own;
#                               overriding it with another model's numbers is a
#                               downgrade, not a tune.
#   --chat-template           : chat_template.jinja is in the repo and already
#                               emits the XML tool-call form the parser expects.
#   --enable-chunked-prefill  : on by default in V1.
#   --max-num-batched-tokens  : the two Qwen3.6 presets pin this to 32768, and at
#                               boot vLLM even warns that spec decode clamped
#                               max_num_scheduled_tokens to 2048 and that you
#                               should "consider increasing max_num_batched_tokens".
#                               MEASURED ON THIS BOX, that advice is BACKWARDS:
#                               32768 is ~32% SLOWER at prefill (848 vs 1251 tok/s
#                               on a 105k-token prompt, warm) and costs 21% of the
#                               KV cache (417,888 vs 525,830 tokens, 1.59x vs 2.01x
#                               concurrency at full length). The Spark's unified
#                               memory does not reward big prefill chunks. Leave it
#                               at the default and ignore the warning.
#   --mamba-block-size        : 48 of the 64 layers are linear_attention, so the
#                               sibling qwen3_5 presets set this — but v0.27.1
#                               hard-rejects it here ("--mamba-block-size can only
#                               be set with --enable-prefix-caching"), because
#                               prefix caching is auto-disabled for this hybrid +
#                               MTP combination. It is a tuning knob, not a
#                               requirement; vLLM picks a valid block size itself.
#
# !! UNIFIED-MEMORY SAFETY (learned the hard way — see logs/crash-*.vllm.log) !!
# The Spark's 121 GB is shared by GPU *and* host. On a model's FIRST boot,
# flashinfer JIT-compiles its FP4 GEMM kernels, and by default spawns one `cicc`
# per core (20), ~5.8 GB each. Stacked on vLLM's memory reservation that can blow
# past 121 GB and trigger a *global* OOM that kills the desktop — the machine has
# to be power-cycled. Four defences, all on by default:
#   1. COMPILE_JOBS (MAX_JOBS) caps concurrent compilers to 2 (~12 GB spike).
#   2. CACHE_HOME persists /root/.cache so the compile happens ONCE, then is reused.
#   3. MEM_LIMIT hard-caps the container so any runaway is killed inside the cgroup
#      instead of taking down the host.
#   4. --no-enable-flashinfer-autotune: the fp4_gemm autotuner's GEMM workspaces
#      are NOT counted in --gpu-memory-utilization and stack on top of the KV
#      reservation (~38 GiB transient spike; it crashed this box twice).
#
# Run ./download-qwen3.8-27b-prismaaqua.sh once first to populate the HF cache.
#
# Usage:
#   ./run-qwen3.8-27b-prismaaqua.sh              # foreground, MTP spec decode (default)
#   ./run-qwen3.8-27b-prismaaqua.sh --no-spec    # plain autoregressive decode
#   DETACH=1 ./run-qwen3.8-27b-prismaaqua.sh     # background server
#   ./run-qwen3.8-27b-prismaaqua.sh --max-num-seqs 8   # append/override any vllm serve flag
#
# Env: IMAGE, PORT, HF_TOKEN, HF_HOME, GPU_MEM_UTIL, MAX_NUM_SEQS, MAX_MODEL_LEN,
#      SPEC_TOKENS, CHAT_TEMPLATE, CACHE_HOME, LOG_DIR, COMPILE_JOBS, MEM_LIMIT,
#      RESTART, VLLM_LOG_LEVEL, AUTOTUNE.
set -euo pipefail

IMAGE="${IMAGE:-vllm-spark:latest}"
MODEL="${MODEL_REPO:-rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm}"
PORT="${PORT:-8000}"
NAME="${NAME:-qwen38-27b-prismaaqua}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Persist flashinfer/torch.compile caches across containers so the expensive (and
# OOM-prone) first-boot JIT compile only happens once. HF cache nests inside it.
CACHE_HOME="${CACHE_HOME:-$HOME/.cache/vllm-spark}"
mkdir -p "$HF_HOME" "$CACHE_HOME"

# --- arg parsing -----------------------------------------------------------
# Spec-decode mode: mtp (DEFAULT) | none. The checkpoint ships the Qwen3.8 MTP
# head (mtp.* tensors, mtp_num_hidden_layers=1) at full BF16 precision — the
# allocator deliberately left it unquantized — so speculative decode costs no
# extra download and no accuracy. It measured +70% decode on this box (10.4 ->
# 17.7 tok/s) with tool calling intact, so it is ON by default; --no-spec turns
# it off. Everything we don't recognise passes straight through to `vllm serve`.
SPEC=mtp
passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mtp)     SPEC=mtp;  shift ;;
    --no-spec) SPEC=none; shift ;;
    *)         passthrough+=("$1"); shift ;;
  esac
done

# The MTP head adds a draft pass per step, so cap utilisation lower to leave
# headroom on the unified pool; plain decode can run a little higher.
case "$SPEC" in
  mtp)  MEM_UTIL="${GPU_MEM_UTIL:-0.52}" ;;
  none) MEM_UTIL="${GPU_MEM_UTIL:-0.72}" ;;
esac

# --- vllm serve options ----------------------------------------------------
vllm_args=(
  "$MODEL"
  --served-model-name qwen/qwen3.8-27b
  --port 8000
  # Full native 262144 window (config.json max_position_embeddings, rope_type
  # "default" — no scaling involved). VERIFIED end-to-end at 254,462 prompt
  # tokens (97% of the window) with correct mid-haystack needle retrieval; over
  # the limit the server returns a clean 400. Boot reserves 525,830 tokens of KV
  # cache = 2.01x concurrency at full length. Prefill is the cost, not memory:
  # ~1251 tok/s at 105k, 839 at 210k, 724 at 254k (attention going quadratic), so
  # a full-window prompt takes ~6 min to ingest. Decode is unaffected.
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-6}"
  --gpu-memory-utilization "$MEM_UTIL"
  --limit-mm-per-prompt '{"image":4,"video":2}'   # vision tower is BF16/unquantized
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml            # the repo's chat template emits
                                          # <tool_call><function=…><parameter=…>
  --reasoning-parser qwen3                # keeps <think>…</think> out of `content`
)

# See UNIFIED-MEMORY SAFETY above. Re-enable the flashinfer fp4_gemm autotuner
# only at low util (e.g. GPU_MEM_UTIL<=0.4), where its uncounted workspace fits.
if [[ "${AUTOTUNE:-0}" != 1 ]]; then
  vllm_args+=(--no-enable-flashinfer-autotune)
fi

# The in-checkpoint MTP head, served by vLLM's Qwen3_5MTP class.
#
# SPEC_TOKENS=3 is the MEASURED optimum on this box (image 6e448d0e, 300-token
# single stream, temp 0, 250-word prose prompt, steady state after warmup):
#   no-spec  10.4 tok/s
#   n=2      16.6 tok/s   (+60%)   accept 2.42 tok/forward, per-pos 81/61%
#   n=3      17.7 tok/s   (+70%)   accept 2.72 tok/forward, per-pos 80/53/39%
#   n=4      16.9 tok/s   (+63%)   accept 2.94 tok/forward, per-pos 80/53/37/24%
# n=4 still accepts *more* tokens per draft than n=3 — the extra draft pass just
# costs more than position 3's 24% hit rate returns. Sweep SPEC_TOKENS= for your
# workload; the turnover is shallow either side of 3.
case "$SPEC" in
  mtp)  vllm_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_TOKENS:-3}}") ;;
  none) : ;;
esac

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

# Optional community chat-template override: CHAT_TEMPLATE=/host/path/to/template.jinja
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
