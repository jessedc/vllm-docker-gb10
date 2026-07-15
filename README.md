# vLLM from source for the DGX Spark (GB10 / sm_121a)

A small, repeatable Docker build that **compiles vLLM from source** for this
machine's GPU — the NVIDIA **GB10 Grace Blackwell** (compute capability
**`sm_121a`**, `aarch64`, CUDA 13).

## Overview

vLLM ships no `sm_121` wheels, so this project builds its own image and gives you
scripts to serve models with it. There are two layers:

- **Build layer** (`Dockerfile`, `build.sh`, `build.lock`) — compiles vLLM's CUDA
  kernels for the GB10 on top of the official CUDA 13 nightly base, and records
  exactly what it built so the image is reproducible. This is the reusable core:
  it works for any vLLM version on any sm_121a Spark.
- **Serve layer** (`run.sh`, `run-qwen3.6.sh`) — launches the built image as an
  OpenAI-compatible API server. `run.sh` is generic (model passed as an
  argument); `run-qwen3.6.sh` is a tuned preset for one specific large model.

A typical first run is: `./build.sh` then `./run.sh Qwen/Qwen3-8B`, after which the
API is on `http://localhost:8000/v1`.

| File | What it is | Reusability |
|------|------------|-------------|
| `Dockerfile` | The build recipe (recompiles vLLM kernels for sm_121a) | Any vLLM version, any Spark |
| `build.sh` | Build driver — resolves pins, tags image, writes `build.lock` | Reusable |
| `build.lock` | Provenance pin for `--reproduce` | Reproducible |
| `run.sh` | Generic launcher — serve any HF model | Reusable across models |
| `run-qwen3.6.sh` | Tuned preset for `nvidia/Qwen3.6-35B-A3B-NVFP4` (MoE) | Single-purpose template |
| `run-qwen3.6-27b.sh` | Tuned preset for `Qwen3.6-27B` dense (PrismaSCOUT NVFP4) + DFlash | Single-purpose template |
| `download-qwen3.6-27b.sh` | Pre-fetches the 27B model + DFlash drafter into the HF cache | Helper for the preset above |
| `run-qwen3.6-27b-nvfp4.sh` | Tuned preset for `unsloth/Qwen3.6-27B-NVFP4` (b12x cute-DSL, built-in MTP) | Single-purpose template |
| `download-qwen3.6-27b-nvfp4.sh` | Pre-fetches the unsloth 27B NVFP4 checkpoint into the HF cache | Helper for the preset above |
| `observability/` | One-command Prometheus + Grafana stack for the server's `/metrics` | Reusable across models |

Everything targets `sm_121a` (GB10); building for a different GPU means changing
`ARCH_LIST` in `build.sh` / `TORCH_CUDA_ARCH_LIST` in the Dockerfile.

## How it works

vLLM has no official `sm_121` wheels yet, and the hard part of a Spark build is
the underlying GPU stack (PyTorch + FlashInfer + Triton). The official nightly
image `vllm/vllm-openai:cu130-nightly` already gets that stack right for this
GPU, so we use it as the base and **recompile only vLLM's own C++/CUDA kernels**
from a pinned source revision, targeting `compute_121a` directly.

```
FROM vllm/vllm-openai:cu130-nightly   # pinned by digest -> torch 2.11+cu130, nvcc 13, ninja
  + apt: git cmake ccache rust        # build tools the runtime image lacks
  +      cuda-libraries-dev-13-0      # CUDA dev headers/symlinks (cusparse.h, libnvrtc.so, ...)
  + pin torch (cu130)                 # so the source build can't swap it for a PyPI wheel
  + git clone vllm @ <commit>         # tracks main HEAD by default
  + use_existing_torch.py             # keep the cu130 torch
  + TORCH_CUDA_ARCH_LIST=12.1a pip install .   # compile kernels for GB10
```

> **Verified:** this builds clean and serves — confirmed by compiling all
> kernels for `compute_121a/sm_121a`, importing the from-source `vllm._C`, and
> serving a completion on the GB10 (`torch 2.11.0+cu130`, `flashinfer 0.6.12`).

## Build

```bash
./build.sh                    # build from current vLLM main HEAD
./build.sh --vllm-ref v0.11.0 # build a specific tag / branch / commit SHA
./build.sh --refresh-base     # re-pull the nightly base and re-pin its digest
./build.sh --no-cache         # clean rebuild
./build.sh --reproduce        # rebuild exactly what build.lock records
```

Each build writes **`build.lock`** recording the base-image digest and the exact
vLLM commit, and tags the image `vllm-spark:<short-commit>` and `vllm-spark:latest`.
The first compile takes a while (single-arch, but it's the full kernel set);
`ccache` makes later rebuilds much faster.

> Tracking `main` is bleeding edge by design. To pin a known-good build,
> commit `build.lock` (or use `--vllm-ref <sha>`) and rebuild with `--reproduce`.

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
`run-qwen3.6.sh`, which serves `nvidia/Qwen3.6-35B-A3B-NVFP4` with NVFP4 quant
auto-detect, FP8 KV cache, FlashInfer attention, Marlin MoE, 256K context, a
`0.65` memory-utilization default, and Qwen3's recommended sampling defaults:

```bash
./run-qwen3.6.sh                  # foreground (Ctrl-C to stop)
./run-qwen3.6.sh --mtp            # enable MTP speculative decoding (off by default)
DETACH=1 ./run-qwen3.6.sh         # detached server, restarts on boot
./run-qwen3.6.sh --max-num-seqs 8 # append/override any vllm serve flag
```

MTP speculative decoding is **opt-in** via `--mtp`; without it, no
`--speculative-config` is passed. The `--mtp` flag is consumed by the script —
any other flags pass straight through to `vllm serve`.

Sampling defaults (`temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
`repetition_penalty`) are baked in via `--override-generation-config`. These are
server-side **defaults** — a request that sets a field still overrides them, so a
client should send these values explicitly (or omit them) to keep them in effect.

#### Qwen3.6-27B dense + DFlash (`run-qwen3.6-27b.sh`)

A second preset serves the **dense** `Qwen3.6-27B` — specifically the
`rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm` mixed-precision NVFP4
checkpoint — with **DFlash** speculative decoding (the `z-lab/Qwen3.6-27B-DFlash`
block-diffusion drafter). Run the downloader once, then launch:

```bash
./download-qwen3.6-27b.sh          # fetch model + drafter into the HF cache (~24 GB)
./run-qwen3.6-27b.sh               # foreground, DFlash spec decode (default)
./run-qwen3.6-27b.sh --mtp         # use the model's built-in MTP head instead
./run-qwen3.6-27b.sh --no-spec     # plain decode, no drafter
DETACH=1 ./run-qwen3.6-27b.sh      # background server (RESTART=no by default)
```

Defaults: `262144` context, `--gpu-memory-utilization 0.65`, FlashInfer
attention, `--mamba-block-size 256` (the model is GDN-hybrid), and the qwen3
reasoning + qwen3_coder tool-call parsers. Validated on the GB10: DFlash runs at
~51% draft acceptance (~5.1 accepted tokens/cycle), and at 262K context the KV
pool (~561K tokens) gives ~2.14× concurrency. The settings translate a community
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
> persists `/root/.cache` via `CACHE_HOME` so the compile is paid once, adds a
> `--memory 112g` cgroup backstop, and defaults `RESTART=no` so a bad boot can't
> crash-loop. It also streams the vLLM log and a 2-second host-memory trace to
> `logs/` (gitignored) for post-mortem. Extra env knobs: `CACHE_HOME`, `LOG_DIR`,
> `COMPILE_JOBS`, `MEM_LIMIT`, `RESTART`, `AUTOTUNE` (re-enable only at low util).

#### Qwen3.6-27B unsloth NVFP4 (`run-qwen3.6-27b-nvfp4.sh`)

A third preset serves **unsloth's** own NVFP4 quant, `unsloth/Qwen3.6-27B-NVFP4` —
the dense, **multimodal** (image + video) `qwen3_5` checkpoint — following
[unsloth's DGX Spark guide](https://unsloth.ai/docs/models/qwen3.6#dgx-spark-with-nvfp4-quants).
Unlike the DFlash preset above it uses the model's **built-in MTP head** (opt-in)
rather than an external drafter:

```bash
./download-qwen3.6-27b-nvfp4.sh    # fetch the checkpoint into the HF cache (~16 GB)
./run-qwen3.6-27b-nvfp4.sh         # foreground, plain autoregressive decode
./run-qwen3.6-27b-nvfp4.sh --mtp   # enable the built-in MTP head (num_speculative_tokens=2)
DETACH=1 ./run-qwen3.6-27b-nvfp4.sh  # background server (RESTART=no by default)
```

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
> ~**20 tok/s (+79%)** with `--mtp` — which keeps multi-turn tool calling fully
> correct (`get_weather` → answer → `add` → answer verified), draft acceptance
> ~65–80%. **Recommendation: run with `--mtp`.**

Verify the image supports the b12x (MoE) kernels (both should print `True`):

```bash
docker run --rm --gpus all --entrypoint python3 vllm-spark:latest -c \
  "import torch; from vllm.utils.flashinfer import has_flashinfer_b12x_gemm as g, \
   has_flashinfer_b12x_moe as m; print(torch.cuda.get_device_capability(), g(), m())"
```

## Notes

- `vllm/vllm-openai:cu130-nightly` is a **moving tag**; `build.sh` pins it by
  **digest** at build time so a given `build.lock` is reproducible. Use
  `--refresh-base` to intentionally move to a newer nightly.
- **Version drift:** the nightly base lags `main` by several minor versions, so
  `main` often pins a newer `flashinfer` than the base ships. We therefore leave
  flashinfer **unpinned** and let it install the version `main` wants;
  flashinfer JIT-compiles its kernels at runtime using the in-image CUDA 13
  toolchain (the first request that hits a new kernel shape pays a one-time
  compile). For a fully aligned, no-surprises build, pass `--vllm-ref <sha>`
  with a commit close to the base, or pin a stable release tag.
- `vllm.__version__` reports `0.1.dev1+g<commit>` because the build does a
  shallow clone (no tags for setuptools-scm); the `+g<commit>` and
  `/etc/vllm-source-commit` are the authoritative provenance.
- To hack on vLLM, the source tree lives at `/opt/vllm` inside the image; the
  built commit is also at `/etc/vllm-source-commit`.
- Verify the GPU is visible from a container:
  `docker run --rm --gpus all vllm-spark:latest --help` (or `nvidia-smi` via an
  `--entrypoint bash` shell).

## Future plans

`run.sh` and `run-qwen3.6.sh` currently duplicate the same Docker plumbing
(`docker run` flags, detach/restart handling, HF cache mount, port mapping). As
soon as a second tuned preset appears, that copy-paste becomes a maintenance
hazard.

The plan is to factor the shared boilerplate into a small helper — e.g. a
`_serve.sh` that takes a model id plus a `vllm_args` array and handles the Docker
invocation — leaving each per-model launcher as just its tuned `vllm_args`. That
keeps the generic path (`run.sh`) and the presets (`run-qwen3.6.sh` and future
siblings) sharing one code path, so a fix to the run logic lands everywhere at
once.
