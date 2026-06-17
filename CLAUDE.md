# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A small, repeatable Docker pipeline that **compiles vLLM from source** for this
machine — an NVIDIA **DGX Spark** (GB10 Grace Blackwell, `aarch64`, GPU compute
capability **`sm_121a`** = `TORCH_CUDA_ARCH_LIST=12.1a`, CUDA 13.0, 121 GB unified
memory). vLLM ships no `sm_121` wheels, so this builds its own image. The repo is
just five shell/Docker files plus docs — there is no Python application code here;
vLLM's source is cloned *inside* the image at build time.

## Commands

```bash
./build.sh                       # build from current vLLM main HEAD
./build.sh --vllm-ref v0.11.0    # build a specific tag / branch / commit SHA
./build.sh --refresh-base        # re-pull the cu130-nightly base, re-pin its digest
./build.sh --no-cache            # clean rebuild
./build.sh --reproduce           # rebuild exactly what build.lock records

./run.sh                                 # serve default small model (Qwen3-4B), interactive
./run.sh Qwen/Qwen3-8B --max-model-len 32768   # serve a model; extra flags pass to `vllm serve`
DETACH=1 ./run.sh Qwen/Qwen3-8B          # detached server, restarts on boot
./run-qwen3.6.sh [--mtp]                 # tuned preset for nvidia/Qwen3.6-35B-A3B-NVFP4
```

API is OpenAI-compatible on `http://localhost:8000/v1` (`curl http://localhost:8000/v1/models`).
There is no test suite or linter; "verification" means the image builds and serves a completion.
Verify GPU visibility: `docker run --rm --gpus all vllm-spark:latest --help`.

## Architecture

Two layers, deliberately separated:

- **Build layer** (`Dockerfile`, `build.sh`, `build.lock`) — the reusable core.
  Works for any vLLM version on any sm_121a Spark.
- **Serve layer** (`run.sh` generic, `run-qwen3.6.sh` tuned preset) — launches the
  built image as the API server. These currently duplicate the `docker run`
  plumbing (gpus/ipc/port/HF-cache mount, DETACH handling); the intended
  refactor is a shared `_serve.sh` helper taking a model id + `vllm_args` array.

**Core build strategy (the key idea):** do **not** rebuild the GPU stack. Start
`FROM vllm/vllm-openai:cu130-nightly`, which already has a known-good
torch 2.11+cu130 + FlashInfer + Triton + nvcc for this GPU, and recompile **only
vLLM's own C++/CUDA kernels** from a pinned source revision, targeting
`compute_121a`. `build.sh` resolves the base image to a **digest** and the vLLM ref
to a **commit**, passes both as build args, and records them in `build.lock` so
`--reproduce` rebuilds bit-for-bit.

**Reproducibility chain:** `build.sh` (resolve digest + commit) → `--build-arg`
into `Dockerfile` → image tagged `vllm-spark:<short-commit>` and `:latest` →
provenance written to `build.lock` (host) and `/etc/vllm-source-commit` (image).
Tracking `main` is the default and is bleeding-edge by design.

## Non-obvious gotchas (all already handled — preserve them when editing)

- The **runtime** base image lacks build tools the source compile needs: install
  `git cmake ccache build-essential` **and Rust** (vLLM main uses
  `setuptools-rust`) **and `cuda-libraries-dev-13-0`** (the base ships CUDA `.so`
  runtimes but not the dev headers/symlinks — without it you get missing
  `cusparse.h` / `libnvrtc.so NOTFOUND`).
- **Pin only the torch family**, leave **flashinfer unpinned**. The nightly base
  lags `main` by several minor versions, so `main` often pins a newer flashinfer
  than the base ships; flashinfer JIT-compiles its kernels at runtime via the
  in-image CUDA 13 toolchain. Pinning torch (`PIP_CONSTRAINT` +
  `use_existing_torch.py`) stops dependency resolution swapping in a generic PyPI
  wheel.
- vLLM's build requirements live at `requirements/build/cuda.txt` (a directory
  layout, not the old `build.txt`).
- `vllm.__version__` reports `0.1.dev1+g<commit>` because the build is a shallow
  clone with no tags (setuptools-scm) — the `+g<commit>` and
  `/etc/vllm-source-commit` are the authoritative provenance.
- Building for a different GPU = change `ARCH_LIST` in `build.sh` **and**
  `TORCH_CUDA_ARCH_LIST` in the `Dockerfile`.

## Serve-layer specifics

- `run.sh` env knobs: `IMAGE`, `PORT`, `HF_HOME`, `HF_TOKEN`, `GPU_MEM_UTIL`
  (default 0.85), `MAX_NUM_SEQS`, `DETACH`.
- `run-qwen3.6.sh` lets the checkpoint's `config.json` **auto-detect** NVFP4
  quantization — do **not** force `--quantization modelopt` (that selects the FP8
  path, wrong for this NVFP4 model; use `modelopt_fp4` if forcing). MTP
  speculative decoding is **opt-in** via `--mtp`; the script consumes `--mtp` and
  passes every other flag straight through to `vllm serve`. Sampling values set
  via `--override-generation-config` are server-side **defaults** only.
