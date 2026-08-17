# vLLM from source for the DGX Spark (GB10 / sm_121a)

A small, repeatable Docker build that **compiles vLLM from source** for this
machine's GPU — the NVIDIA **GB10 Grace Blackwell** (compute capability
**`sm_121a`**, `aarch64`, CUDA 13).

> 📊 **Benchmark:** [Qwen3.6-27B NVFP4 multi-turn tool calling](benchmarks/qwen3.6-27b-tool-calling.md)
> — unsloth (MTP) vs PrismaSCOUT (DFlash): throughput and tool-calling accuracy.

## Overview

vLLM ships no `sm_121` wheels, so this project builds its own image and gives you
scripts to serve models with it. There are two layers:

- **Build layer** (`Dockerfile`, `build.sh`, `build.lock`) — compiles vLLM's CUDA
  kernels for the GB10 on top of the official CUDA 13 nightly base, and records
  exactly what it built so the image is reproducible. This is the reusable core:
  it works for any vLLM version on any sm_121a Spark.
- **Serve layer** (`run.sh`, `run-qwen3.6-35b-a3b.sh`) — launches the built image as an
  OpenAI-compatible API server. `run.sh` is generic (model passed as an
  argument); `run-qwen3.6-35b-a3b.sh` is a tuned preset for one specific large model.

A typical first run is: `./build.sh` then `./run.sh Qwen/Qwen3-8B`, after which the
API is on `http://localhost:8000/v1`.

| File | What it is | Reusability |
|------|------------|-------------|
| `Dockerfile` | The build recipe (recompiles vLLM kernels for sm_121a) | Any vLLM version, any Spark |
| `build.sh` | Build driver — resolves pins, tags image, writes `build.lock` | Reusable |
| `build.lock` | Provenance pin for `--reproduce` | Reproducible |
| `run.sh` | Generic launcher — serve any HF model | Reusable across models |
| `run-qwen3.6-35b-a3b.sh` | Tuned preset for `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE) | Single-purpose template |
| `run-qwen3.6-27b-prismascout.sh` | Tuned preset for `Qwen3.6-27B` dense (PrismaSCOUT NVFP4) + DFlash | Single-purpose template |
| `download-qwen3.6-27b-prismascout.sh` | Pre-fetches the 27B model + DFlash drafter into the HF cache | Helper for the preset above |
| `run-qwen3.6-27b-unsloth.sh` | Tuned preset for `unsloth/Qwen3.6-27B-NVFP4` (b12x cute-DSL, built-in MTP) | Single-purpose template |
| `download-qwen3.6-27b-unsloth.sh` | Pre-fetches the unsloth 27B NVFP4 checkpoint into the HF cache | Helper for the preset above |
| `run-qwen3.8-27b-prismaaqua.sh` | Tuned preset for `rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm` (mixed-precision, built-in MTP) | Single-purpose template |
| `download-qwen3.8-27b-prismaaqua.sh` | Pre-fetches the Qwen3.8-27B PrismaAQUA checkpoint into the HF cache | Helper for the preset above |
| `observability/` | One-command Prometheus + Grafana stack for the server's `/metrics` | Reusable across models |

Everything targets `sm_121a` (GB10); building for a different GPU means changing
`ARCH_LIST` in `build.sh` / `TORCH_CUDA_ARCH_LIST` in the Dockerfile.

## How it works

vLLM has no official `sm_121` wheels, and the hard part of a Spark build is the
underlying GPU stack (PyTorch + FlashInfer + Triton + CUDA). Upstream's own
release image already gets that stack exactly right — it is built from the same
source we want, with the versions that release pins. The *only* thing it lacks is
kernels for this GPU: every official image is compiled for `8.0`–`12.0`, and GB10
is `12.1a`.

So we start **FROM the official vLLM image for the ref we are building** and
**recompile only vLLM's own C++/CUDA kernels** against it, targeting
`compute_121a` directly. Base and source stay in lockstep by construction —
`build.sh` derives one from the other.

```
FROM vllm/vllm-openai:v0.27.1-aarch64   # pinned by digest -> torch 2.13+cu130, flashinfer, nvcc 13
  + apt: git cmake ccache rust          # build tools the runtime image lacks
  +      cuda-libraries-dev-13-0        # CUDA dev headers/symlinks (cusparse.h, libnvrtc.so, ...)
  + pin torch (cu130)                   # so the source build can't swap it for a PyPI wheel
  + git clone vllm @ <commit>           # the same ref the base was built from
  + use_existing_torch.py               # keep the cu130 torch
  + TORCH_CUDA_ARCH_LIST=12.1a pip install .   # compile kernels for GB10
  + keep flashinfer cubins if they match # no first-boot JIT storm (see gotchas)
```

## Build

```bash
./build.sh                    # build the pinned vLLM release (v0.27.1)
./build.sh --vllm-ref v0.27.1 # build a specific tag / branch / commit SHA
./build.sh --vllm-ref main    # bleeding edge — uses the nightly base image
./build.sh --base-image REF   # override the auto-derived base image
./build.sh --refresh-base     # re-pull the base image and re-pin its digest
./build.sh --no-cache         # clean rebuild
./build.sh --reproduce        # rebuild exactly what build.lock records
```

Each build writes **`build.lock`** recording the base image + digest and the exact
vLLM commit, and tags the image `vllm-spark:<short-commit>`, `vllm-spark:latest`,
and — for a release ref — `vllm-spark:v0.27.1`.
The first compile takes a while (single-arch, but it's the full kernel set);
`ccache` makes later rebuilds much faster.

> The default is a **pinned release**. `--vllm-ref main` is still supported and
> is bleeding edge by design; it falls back to `vllm/vllm-openai:nightly-aarch64`
> as the base, which can lag `main` and reintroduces the version skew the release
> pairing avoids.

## Serve

```bash
./run.sh                                   # small default model (Qwen3-4B), interactive
./run.sh Qwen/Qwen3-8B                      # serve a specific model
HF_TOKEN=hf_xxx ./run.sh meta-llama/Llama-3.1-8B-Instruct
DETACH=1 ./run.sh Qwen/Qwen3-8B            # detached server, restarts on boot
./run.sh Qwen/Qwen3-8B --max-model-len 32768   # extra flags pass to `vllm serve`
```

Defaults are tuned for the Spark's unified memory (`--gpu-memory-utilization 0.85`).
The OpenAI-compatible API is then on `http://localhost:8000/v1`:

```bash
curl http://localhost:8000/v1/models
```

Tunables via env: `IMAGE`, `PORT`, `HF_HOME`, `GPU_MEM_UTIL`, `MAX_NUM_SEQS`.

### Tuned model presets

`run.sh` is generic. For a model that needs specific, repeatable tuning, a
dedicated launcher captures the whole flag set in one place — see
`run-qwen3.6-35b-a3b.sh`, which serves `nvidia/Qwen3.6-35B-A3B-NVFP4` with NVFP4 quant
auto-detect, FP8 KV cache, FlashInfer attention, Marlin MoE, 256K context, a
`0.65` memory-utilization default, and Qwen3's recommended sampling defaults:

```bash
./run-qwen3.6-35b-a3b.sh                  # foreground (Ctrl-C to stop)
./run-qwen3.6-35b-a3b.sh --mtp            # enable MTP speculative decoding (off by default)
DETACH=1 ./run-qwen3.6-35b-a3b.sh         # detached server, restarts on boot
./run-qwen3.6-35b-a3b.sh --max-num-seqs 8 # append/override any vllm serve flag
```

MTP speculative decoding is **opt-in** via `--mtp`; without it, no
`--speculative-config` is passed. The `--mtp` flag is consumed by the script —
any other flags pass straight through to `vllm serve`.

Sampling defaults (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`repetition_penalty`) are baked in via `--override-generation-config`. These are
server-side **defaults** — a request that sets a field still overrides them, so a
client should send these values explicitly (or omit them) to keep them in effect.

#### Qwen3.6-27B dense + DFlash (`run-qwen3.6-27b-prismascout.sh`)

A second preset serves the **dense** `Qwen3.6-27B` — specifically the
`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` mixed-precision NVFP4
checkpoint — with **DFlash** speculative decoding (the `z-lab/Qwen3.6-27B-DFlash`
block-diffusion drafter). Run the downloader once, then launch:

```bash
./download-qwen3.6-27b-prismascout.sh          # fetch model + drafter into the HF cache (~24 GB)
./run-qwen3.6-27b-prismascout.sh               # foreground, DFlash spec decode (default)
./run-qwen3.6-27b-prismascout.sh --mtp         # use the model's built-in MTP head instead
./run-qwen3.6-27b-prismascout.sh --no-spec     # plain decode, no drafter
DETACH=1 ./run-qwen3.6-27b-prismascout.sh      # background server, runs until stopped (RESTART=unless-stopped)
```

Defaults: `262144` context, `--gpu-memory-utilization 0.65`, `--mamba-block-size
256` (the model is GDN-hybrid), `SPEC_TOKENS=10`, and the qwen3 reasoning +
qwen3_coder tool-call parsers. **Attention backend (changed in v0.27.1):** the
script used to pass `--attention-backend flashinfer` unconditionally, because
with DFlash active vLLM overrode it to `FLASH_ATTN` — the only backend that
supports the drafter's non-causal attention
([vllm#41559](https://github.com/vllm-project/vllm/issues/41559)). **v0.27.1
picks the backend per model instead of overriding**, so flashinfer would be
honoured for the main decoder while only the drafter fell back to `FLASH_ATTN` —
and `FlashInferBackend` cannot do full CUDA graphs under spec-decode, forcing
`cudagraph_mode=PIECEWISE`. Measured on this box (image `6e448d0e`, 300-token
single stream, temp 0): **28.8 tok/s** with flashinfer vs **32.2 tok/s** letting
it auto-pick `FLASH_ATTN` (**+11.9%**, and 448,406 vs 444,200 KV tokens). The
script now sets the backend only for `--mtp`/`--no-spec`. The KV cache still
stays bf16 either way. Measured on
the GB10 (image `2396a611`, toolbench): DFlash at `SPEC_TOKENS=10` accepts **5.13
tokens per model forward** (~10 s/task), beating `n=5`'s 4.21 — see the
[benchmark](benchmarks/qwen3.6-27b-tool-calling.md#follow-up-dflash-spec_tokens-sweep-on-the-rebuilt-image-2396a611).
At 262K context the KV pool (~561K tokens) gives ~2.14× concurrency. The settings translate a community
`docker-compose` (kept as `qwen36-27b-notes.md` for reference) onto our
mainline-from-source image — aeon-vllm-ultimate-only env vars and multimodal
flags are dropped (see the script header for why). Because it's a thinking model,
give requests generous `max_tokens` (2048+) or the reply is all reasoning.

> **Unified-memory safety (important).** On a model's *first* boot, flashinfer
> JIT-compiles its FP4 kernels and autotunes `fp4_gemm`; both allocate memory the
> GPU and host share, **on top of** the `--gpu-memory-utilization` reservation.
> Unbounded, this overruns the 121 GB pool and triggers a *global* OOM that
> crashes the whole machine (it did, twice, during bring-up). The script defends
> against this by default: it caps the JIT compiler (`MAX_JOBS=2`), disables the
> `fp4_gemm` autotuner (`--no-enable-flashinfer-autotune`, the ~38 GiB offender),
> persists `/root/.cache` via `CACHE_HOME` so the compile is paid once, and adds a
> `--memory 112g` cgroup backstop. This preset defaults `RESTART=unless-stopped` so
> the detached server runs until explicitly stopped (surviving crashes/reboots);
> set `RESTART=no` for the very first cold-cache boot so a bad compile can't
> crash-loop. It also streams the vLLM log and a 2-second host-memory trace to
> `logs/` (gitignored) for post-mortem. Extra env knobs: `CACHE_HOME`, `LOG_DIR`,
> `COMPILE_JOBS`, `MEM_LIMIT`, `RESTART`, `AUTOTUNE` (re-enable only at low util).

#### Qwen3.6-27B unsloth NVFP4 (`run-qwen3.6-27b-unsloth.sh`)

A third preset serves **unsloth's** own NVFP4 quant, `unsloth/Qwen3.6-27B-NVFP4` —
the dense, **multimodal** (image + video) `qwen3_5` checkpoint — following
[unsloth's DGX Spark guide](https://unsloth.ai/docs/models/qwen3.6#dgx-spark-with-nvfp4-quants).
Unlike the DFlash preset above it uses the model's **built-in MTP head** — **on by
default** here (measured +79%, see below) — rather than an external drafter:

```bash
./download-qwen3.6-27b-unsloth.sh      # fetch the checkpoint into the HF cache (~16 GB)
./run-qwen3.6-27b-unsloth.sh           # foreground, MTP spec decode (default; num_speculative_tokens=2)
./run-qwen3.6-27b-unsloth.sh --no-spec # disable MTP -> plain autoregressive decode
./run-qwen3.6-27b-unsloth.sh --no-reasoning-parser  # return the raw <think> trace in `content`
DETACH=1 ./run-qwen3.6-27b-unsloth.sh  # background server (RESTART=no by default)
```

> **Thinking / reasoning traces.** This model thinks by default, and its
> `<think>` block alone can run 500+ tokens. Two gotchas, both handled: (1) the
> `qwen3` reasoning parser *strips and discards* the thinking (`reasoning_content`
> comes back `null`); and because it buffers until `</think>`, a **small client
> `max_tokens` truncates mid-thought and returns an empty response**. The preset
> sets `DEFAULT_MAX_TOKENS=4096` (a generation-config `max_new_tokens` default, env
> overridable) so no-`max_tokens` clients finish thinking, and offers
> **`--no-reasoning-parser`** to drop the parser so the raw `<think>…</think>answer`
> is returned verbatim in `content` (the client splits it — tool calling still
> works). If you don't want thinking at all, send
> `chat_template_kwargs: {"enable_thinking": false}`. Note `preserve_thinking` only
> affects how *past* turns are re-rendered, not the current response.

NVFP4 is auto-detected (`compressed-tensors`), so there is no `--quantization`;
the attention backend is **left to auto-pick** because forcing FlashInfer breaks
this model's multimodal attention (same as gemma4). It inherits the full
unified-memory safety machinery described above (`MAX_JOBS=2`, autotuner off,
`CACHE_HOME` persistence, `--memory 112g`, `RESTART=no`, `logs/` memtrace).

> **b12x, measured — read before trusting the guide's flags.** The guide tells
> you to pass `--moe-backend flashinfer_b12x` "or get 2× slower inference." On
> this **dense** checkpoint that flag is a **no-op** (there are no MoE layers) —
> it's kept only for guide parity / a future MoE variant. The dense NVFP4 GEMM
> auto-selects `FlashInferCutlassNvFp4LinearKernel` (cutlass), which is the best
> available path and **not** the marlin W4A16 worst case. There is nothing faster
> to force: `--linear-backend flashinfer_b12x` **hard-fails on boot** in this
> build (no b12x linear kernel for the layer type). `CUTE_DSL_ARCH=sm_121a` is set
> per the guide and is harmless. **Measured on the GB10:** single-stream decode is
> ~**11 tok/s** (the Spark's memory-bandwidth ceiling for a 27B), rising to
> ~**20 tok/s (+79%)** with the built-in MTP head — which keeps multi-turn tool
> calling fully correct (`get_weather` → answer → `add` → answer verified), draft
> acceptance ~65–80%. **MTP is therefore the default here**; pass `--no-spec` to
> disable it (drops to ~11 tok/s, `--gpu-memory-utilization` 0.72 instead of 0.52).

Verify the image supports the b12x (MoE) kernels (both should print `True`):

```bash
docker run --rm --gpus all --entrypoint python3 vllm-spark:latest -c \
  "import torch; from vllm.utils.flashinfer import has_flashinfer_b12x_gemm as g, \
   has_flashinfer_b12x_moe as m; print(torch.cuda.get_device_capability(), g(), m())"
```

#### Qwen3.8-27B PrismaAQUA 5.5 bpp (`run-qwen3.8-27b-prismaaqua.sh`)

A fourth preset serves [`rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm`](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm)
— a **mixed-precision** quant of `Qwen/Qwen3.8-27B` where every Linear was priced
individually and given the format it earned: 247 NVFP4 (W4A4, group 16, mostly
MLPs), 223 FP8-E4M3 (per-channel, mostly attention out-projections), 26 BF16 (the
tiny `linear_attn` gate projections). 21.98 GiB on disk, 5.5 bpp over the
quantizable body; `lm_head`, `embed_tokens`, the visual tower and the MTP head
stay BF16. Despite the "3.8" name the arch is `qwen3_5`, the same family the
unsloth preset above serves, so vLLM v0.27.1 and the base image's transformers
5.15.0 both know it natively.

```bash
./download-qwen3.8-27b-prismaaqua.sh      # fetch the checkpoint into the HF cache (~22 GiB)
./run-qwen3.8-27b-prismaaqua.sh           # foreground, MTP spec decode (default; num_speculative_tokens=3)
./run-qwen3.8-27b-prismaaqua.sh --no-spec # disable MTP -> plain autoregressive decode
DETACH=1 ./run-qwen3.8-27b-prismaaqua.sh  # background server (RESTART=no by default)
```

The whole mixed allocation is **auto-detected**: `config.json` declares
`format: "mixed-precision"` globally and a per-`config_group` `format`
(`nvfp4-pack-quantized` / `float-quantized`) underneath, and vLLM's
`CompressedTensorsConfig` reads the per-group format first. Passing
`--quantization` would flatten every layer onto one scheme and is exactly wrong
here. The attention backend is left to auto-pick (it lands on FlashAttention) —
the checkpoint is multimodal, so forcing FlashInfer hits the same rejection
gemma4 and the unsloth preset hit. `--moe-backend` is omitted because the model
is dense, and `--override-generation-config` is omitted because the repo ships
its own `generation_config.json` with the Qwen3.8 defaults.

> **`--mamba-block-size` is not passed here**, unlike the two Qwen3.6 presets.
> 48 of the 64 layers are `linear_attention`, so it looks applicable, but
> v0.27.1 **hard-rejects** it for this model: *"--mamba-block-size can only be
> set with --enable-prefix-caching"* — prefix caching is auto-disabled for this
> hybrid + MTP combination. It is a tuning knob, not a requirement; vLLM picks a
> valid block size itself.

> **Measured on the GB10** (image `6e448d0e`, 300-token single stream, temp 0,
> steady state after warmup): no-spec **10.4 tok/s**; MTP `n=2` **16.6**;
> `n=3` **17.7 (+70%)**; `n=4` **16.9**. Per-position draft acceptance at `n=3`
> is 80 / 53 / 39%. `n=4` accepts *more* tokens per draft (2.94 vs 2.72 per
> forward) — the extra draft pass simply costs more than position 3's 24% hit
> rate returns, which is why **`SPEC_TOKENS=3` is the default**. Boot gives
> 526,621 tokens of KV cache at `--gpu-memory-utilization 0.52`. Tool calling
> (`qwen3_xml`), the tool-result round trip, image input and streaming were all
> verified working with MTP on.

> **262144 context, verified.** The preset serves the full native window
> (`max_position_embeddings: 262144`, `rope_type: "default"` — no scaling), and it
> is exercised, not just configured: a **254,462-token** prompt (97% of the window)
> returned the correct mid-haystack needle, and over-limit requests get a clean
> `400`. Boot reserves **525,830 tokens** of KV cache = **2.01×** concurrency at
> full length, so two full-length requests fit. Memory is not the constraint —
> **prefill** is: ~1251 tok/s at 105k, 839 at 210k, 724 at 254k as attention goes
> quadratic, meaning a full-window prompt takes **~6 minutes to ingest**. Decode
> speed is unaffected by context length.

> **Do not add `--max-num-batched-tokens` here** — measured, and it contradicts
> both the Qwen3.6 presets (which pin 32768) and vLLM's own boot warning
> ("spec decode clamped `max_num_scheduled_tokens` to 2048… consider increasing
> `max_num_batched_tokens`"). On this box that advice is **backwards**: 32768 made
> prefill ~**32% slower** (848 vs 1251 tok/s on a warm 105k prompt) *and* cut the
> KV cache by 21% (417,888 vs 525,830 tokens, 1.59× vs 2.01× concurrency). The
> Spark's unified memory does not reward large prefill chunks. Ignore the warning.

> **Reasoning traces come back in `reasoning`, not `reasoning_content`.** vLLM
> v0.27.1 surfaces the parsed `<think>` block on the OpenAI-standard `reasoning`
> field (streaming deltas use `reasoning` too); `reasoning_content` is `null`.
> That is a field rename, not the discarded-thinking behaviour the Qwen3.6
> unsloth preset documents — the trace is fully present here, so this preset
> needs neither `--no-reasoning-parser` nor a `DEFAULT_MAX_TOKENS` workaround.

It inherits the full unified-memory safety machinery described above
(`MAX_JOBS=2`, autotuner off, `CACHE_HOME` persistence, `--memory 112g`,
`RESTART=no`, `logs/` memtrace).

## Notes

- **Base and source ref move together.** `build.sh`'s `base_for_ref()` maps a
  `v*` ref to that release's official image and anything else (`main`, a branch,
  a SHA) to `nightly-aarch64`. Base tags are **moving**, so `build.sh` pins the
  resolved **digest** into `build.lock`; use `--refresh-base` to intentionally
  move to a newer one, or `--base-image` to override the mapping.
- **`vllm/vllm-openai:cu130-nightly` is dead** — last built 2026-04-23 and still
  on torch 2.11, while v0.27.1 requires torch 2.13. It was the base for earlier
  builds here; don't go back to it.
- **Version drift is what the release pairing removes.** On a release build the
  base already carries the exact `torch` / `flashinfer` that ref pins, so
  flashinfer's prebuilt cubins stay valid and are **kept** — no first-boot JIT
  storm. flashinfer is still left **unpinned** in the build, which matters for
  `--vllm-ref main`: the nightly base can lag what `main` pins, and there
  flashinfer JIT-compiles at runtime via the in-image CUDA 13 toolchain (the
  first request hitting a new kernel shape pays a one-time compile). The
  Dockerfile drops the cubin/jit-cache packages only when their version actually
  diverges from `flashinfer-python`.
- `vllm.__version__` is a setuptools-scm string, not the release number — a
  release build reports `0.27.2.dev0+g<commit>.d<date>` (scm sees tag `v0.27.1`
  and bumps to the next dev). The `+g<commit>` and `/etc/vllm-source-commit` are
  the authoritative provenance.
- The compiled extension is **`vllm._C_stable_libtorch`**, not `vllm._C` —
  upstream moved to the stable libtorch ABI, so `import vllm._C` raising
  `ModuleNotFoundError` is normal, not a broken build.
- Kernels emit **`sm_120` SASS** even with `TORCH_CUDA_ARCH_LIST=12.1a`: sm_121
  is family-compatible with sm_120 cubins under CUDA 13, so they run on GB10.
  This matches the earlier benchmarked images byte for byte.
- To hack on vLLM, the source tree lives at `/opt/vllm` inside the image; the
  built commit is also at `/etc/vllm-source-commit`.
- Verify the GPU is visible from a container:
  `docker run --rm --gpus all vllm-spark:latest --help` (or `nvidia-smi` via an
  `--entrypoint bash` shell).

## Future plans

`run.sh` and `run-qwen3.6-35b-a3b.sh` currently duplicate the same Docker plumbing
(`docker run` flags, detach/restart handling, HF cache mount, port mapping). As
soon as a second tuned preset appears, that copy-paste becomes a maintenance
hazard.

The plan is to factor the shared boilerplate into a small helper — e.g. a
`_serve.sh` that takes a model id plus a `vllm_args` array and handles the Docker
invocation — leaving each per-model launcher as just its tuned `vllm_args`. That
keeps the generic path (`run.sh`) and the presets (`run-qwen3.6-35b-a3b.sh` and future
siblings) sharing one code path, so a fix to the run logic lands everywhere at
once.
