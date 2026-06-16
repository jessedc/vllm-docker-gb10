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
| `run-qwen3.6.sh` | Tuned preset for `nvidia/Qwen3.6-35B-A3B-NVFP4` | Single-purpose template |

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
