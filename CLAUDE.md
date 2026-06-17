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
./run-gemma4.sh [--no-tools]             # tuned preset for RedHatAI/gemma-4-12B-it-NVFP4 (multimodal)
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
- **Very-new model archs need transformers from source, not just the pin.** vLLM
  `main` recognizes architectures (e.g. `gemma4_unified`) before they reach a
  tagged transformers release; vLLM only requires `transformers >= 5.5.3`, which
  is satisfied but insufficient — HF `AutoConfig` rejects the checkpoint before
  vLLM's own model class loads. The `Dockerfile` therefore installs
  `transformers @ git+…` (arg `TRANSFORMERS_REF`, default `main`) **after** the
  kernel compile so the cached build layer is reused; `PIP_CONSTRAINT` still
  guards the torch pin. Set `--build-arg TRANSFORMERS_REF=<release>` to opt out.
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
- `run-gemma4.sh` serves the unified multimodal `gemma4_unified` checkpoint
  (text+image+audio). Gotchas baked in: (1) do **not** force
  `--attention-backend flashinfer` — the model's bidirectional multimodal
  attention is rejected ("partial multimodal token full attention not
  supported"); let vLLM auto-pick (lands on `TRITON_ATTN`). (2) `GPU_MEM_UTIL`
  defaults to **0.65**: util is a fraction of *total* (121.7 GiB) but only
  ~97.5 GiB is free at startup, so 0.85 overshoots. NVFP4 is auto-detected
  (compressed-tensors `nvfp4-pack-quantized`) — no `--quantization`. Tool calling
  + reasoning use the `gemma4` parsers and the in-image chat template at
  `/opt/vllm/examples/tool_chat_template_gemma4.jinja`; `--no-tools` disables all
  three. Needs the transformers-from-source build above.
