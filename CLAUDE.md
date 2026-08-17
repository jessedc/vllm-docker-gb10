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
./build.sh                       # build the pinned vLLM release (v0.27.1)
./build.sh --vllm-ref v0.27.1    # build a specific tag / branch / commit SHA
./build.sh --vllm-ref main       # bleeding edge (falls back to the nightly base)
./build.sh --base-image REF      # override the auto-derived base image
./build.sh --refresh-base        # re-pull the base image, re-pin its digest
./build.sh --no-cache            # clean rebuild
./build.sh --reproduce           # rebuild exactly what build.lock records

./run.sh                                 # serve default small model (Qwen3-4B), interactive
./run.sh Qwen/Qwen3-8B --max-model-len 32768   # serve a model; extra flags pass to `vllm serve`
DETACH=1 ./run.sh Qwen/Qwen3-8B          # detached server, restarts on boot
./run-qwen3.6-35b-a3b.sh [--mtp]                 # tuned preset for nvidia/Qwen3.6-35B-A3B-NVFP4
./run-gemma4-12b.sh [--no-tools]             # tuned preset for RedHatAI/gemma-4-12B-it-NVFP4 (multimodal)

(cd observability && docker compose up -d) # Prometheus + Grafana on the server's /metrics (localhost:3000)
```

API is OpenAI-compatible on `http://localhost:8000/v1` (`curl http://localhost:8000/v1/models`).
There is no test suite or linter; "verification" means the image builds and serves a completion.
Verify GPU visibility: `docker run --rm --gpus all vllm-spark:latest --help`.

## Architecture

Two layers, deliberately separated:

- **Build layer** (`Dockerfile`, `build.sh`, `build.lock`) — the reusable core.
  Works for any vLLM version on any sm_121a Spark.
- **Serve layer** (`run.sh` generic, `run-qwen3.6-35b-a3b.sh` tuned preset) — launches the
  built image as the API server. These currently duplicate the `docker run`
  plumbing (gpus/ipc/port/HF-cache mount, DETACH handling); the intended
  refactor is a shared `_serve.sh` helper taking a model id + `vllm_args` array.

**Core build strategy (the key idea):** do **not** rebuild the GPU stack. Start
`FROM` the **official vLLM image built from the same source ref we compile** —
`vllm/vllm-openai:<ref>-aarch64` — which already ships exactly the
torch+cu130 / FlashInfer / Triton / nvcc versions that ref pins, and recompile
**only vLLM's own C++/CUDA kernels** targeting `compute_121a`. Every official
image is built for `8.0`–`12.0`; sm_121a is the sole gap we fill. `build.sh`
resolves the base image to a **digest** and the vLLM ref to a **commit**, passes
both as build args, and records them in `build.lock` so `--reproduce` rebuilds
bit-for-bit.

**Base/source lockstep (the invariant to preserve):** `base_for_ref()` in
`build.sh` derives the base from `VLLM_REF` — a `v*` ref → that release's image,
anything else (`main`, a branch, a SHA) → `nightly-aarch64`. Never hardcode a
base that doesn't correspond to the ref being built; the version skew that
pairing avoids is what most of the gotchas below used to work around.
`--base-image` overrides it when you truly need to.

**Reproducibility chain:** `build.sh` (resolve digest + commit) → `--build-arg`
into `Dockerfile` → image tagged `vllm-spark:<short-commit>`, `:latest`, and
`:<release>` for a `v*` ref → provenance written to `build.lock` (host) and
`/etc/vllm-source-commit` (image). The default is a **pinned release**;
`--vllm-ref main` is still supported and bleeding-edge by design.

## Non-obvious gotchas (all already handled — preserve them when editing)

- The **runtime** base image lacks build tools the source compile needs: install
  `git cmake ccache build-essential` **and Rust** (vLLM main uses
  `setuptools-rust`) **and `cuda-libraries-dev-13-0`** (the base ships CUDA `.so`
  runtimes but not the dev headers/symlinks — without it you get missing
  `cusparse.h` / `libnvrtc.so NOTFOUND`).
- **`vllm/vllm-openai:cu130-nightly` is a dead tag** — last built 2026-04-23,
  stuck on torch 2.11, while v0.27.1 requires torch 2.13. It was the base for
  every build here before v0.27.1; do not go back to it. Upstream's current nightlies are `nightly-aarch64` / `cu129-nightly`,
  and releases are `v<x.y.z>-aarch64` (all CUDA 13.0.2, arm64).
- **Pin only the torch family**, leave **flashinfer unpinned**. With the release
  base these already agree, so the pin is insurance; it still earns its keep for
  `--vllm-ref main`, where the nightly base can lag what `main` pins. Pinning
  torch (`PIP_CONSTRAINT` + `use_existing_torch.py`) stops dependency resolution
  swapping in a generic PyPI wheel.
- **Keep flashinfer's prebuilt cubins when they match.** `flashinfer-cubin` /
  `flashinfer-jit-cache` are valid only for their exact `flashinfer-python`
  version. Older builds deleted them unconditionally (the nightly base's copies
  were always stale), which forced a **first-boot JIT storm** — the compile that
  can drive the Spark's unified memory into a global OOM. The `Dockerfile` now
  compares versions *after* the vLLM install and drops them only on a real
  mismatch. Compare the **public** version: `flashinfer-jit-cache` carries a
  `+cu130` local suffix that `flashinfer-python` does not, and a naive string
  compare throws away good cubins.
- vLLM's build requirements live at `requirements/build/cuda.txt` (a directory
  layout, not the old `build.txt`).
- **The transformers-from-source hack is now opt-in and normally unnecessary.**
  It existed because vLLM `main` recognized archs (e.g. `gemma4_unified`) before
  they reached a tagged transformers release, and vLLM's `transformers >= 5.5.3`
  floor is satisfied but insufficient — HF `AutoConfig` rejects the checkpoint
  before vLLM's model class loads. The v0.27.1 base ships **transformers 5.15.0**,
  which already knows `gemma4_unified` and `qwen3_5`, so `TRANSFORMERS_REF` now
  defaults to **empty** (keep the base's) — overwriting it with a much newer
  `main` on a pinned release is the bigger risk. Set
  `--build-arg TRANSFORMERS_REF=main` to opt back in for an arch the release
  predates; the step still runs **after** the kernel compile so that cached layer
  is reused, and `PIP_CONSTRAINT` still guards the torch pin.
- `vllm.__version__` is a setuptools-scm string, not the release number: a
  release build reports `0.27.2.dev0+g<commit>.d<date>` (scm sees tag v0.27.1 and
  bumps to the next dev), and a `main` build reports `0.1.dev1+g<commit>` when the
  shallow clone carries no tags at all. Either way the `+g<commit>` and
  `/etc/vllm-source-commit` are the authoritative provenance.
- **The extension module is `vllm._C_stable_libtorch`, not `vllm._C`** (upstream
  moved to the stable libtorch ABI). `import vllm._C` now raises
  `ModuleNotFoundError` — that is not a broken build. The compiled artifacts are
  `vllm/_C_stable_libtorch.abi3.so`, `_moe_C_stable_libtorch.abi3.so`,
  `_qutlass_C.abi3.so`, `_flashkda_C.abi3.so`.
- **The kernels emit `sm_120` SASS, not `sm_121a`, and that is expected.** With
  `TORCH_CUDA_ARCH_LIST=12.1a`, `cuobjdump --list-elf` shows `sm_120` (plus a few
  `sm_80/89/90`) and no PTX. sm_121 is family-compatible with sm_120 cubins under
  CUDA 13, so they run on GB10. The previously benchmarked `2396a611` image has
  the byte-identical arch profile — do not "fix" this.
- Building for a different GPU = change `ARCH_LIST` in `build.sh` **and**
  `TORCH_CUDA_ARCH_LIST` in the `Dockerfile`.

## Serve-layer specifics

- `run.sh` env knobs: `IMAGE`, `PORT`, `HF_HOME`, `HF_TOKEN`, `GPU_MEM_UTIL`
  (default 0.85), `MAX_NUM_SEQS`, `DETACH`.
- `run-qwen3.6-35b-a3b.sh` lets the checkpoint's `config.json` **auto-detect** NVFP4
  quantization — do **not** force `--quantization modelopt` (that selects the FP8
  path, wrong for this NVFP4 model; use `modelopt_fp4` if forcing). MTP
  speculative decoding is **opt-in** via `--mtp`; the script consumes `--mtp` and
  passes every other flag straight through to `vllm serve`. Sampling values set
  via `--override-generation-config` are server-side **defaults** only.
- `run-gemma4-12b.sh` serves the unified multimodal `gemma4_unified` checkpoint
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
- `run-qwen3.6-27b-prismascout.sh` serves the dense
  `rdtand/…PrismaSCOUT…NVFP4` checkpoint with **DFlash** spec decode (external
  `z-lab/Qwen3.6-27B-DFlash` drafter) by default. **Attention-backend gotcha
  (CHANGED in v0.27.1 — measured, do not revert):** the script used to pass
  `--attention-backend flashinfer` unconditionally, on the belief that vLLM would
  override it to `FLASH_ATTN` under DFlash (the drafter needs *non-causal*
  attention, supported only by `FLASH_ATTN`/`FLEX_ATTENTION`). **v0.27.1 no longer
  overrides — it picks per model**, so flashinfer is honoured for the *main*
  decoder while only the *drafter* falls back to `FLASH_ATTN`. That split costs
  throughput, because `FlashInferBackend` cannot do full CUDA graphs with
  spec-decode (`cudagraph_mode` is forced down to `PIECEWISE`). Measured on this
  box (image `6e448d0e`, 300-token single stream, temp 0): flashinfer main =
  **28.8 tok/s**, PIECEWISE only; auto-pick `FLASH_ATTN` both = **32.2 tok/s**,
  8 FULL + 21 PIECEWISE graphs — **+11.9%**, plus slightly more KV cache
  (448,406 vs 444,200 tokens). So the script now sets the backend **only for
  `--mtp`/`--no-spec`** and lets DFlash auto-pick. The KV cache still stays
  **bf16** regardless of `--kv-cache-dtype` (do **not** try to force fp8 KV with
  DFlash — vllm#41559). **Spec width:**
  `SPEC_TOKENS=10` is the measured optimum on the GB10 (accepts 5.13 tok/forward
  vs `n=5`'s 4.21; ~22% lower per-task latency) — accept *length* drives speed, not
  the higher headline *rate* a smaller `n` shows. Needs
  `run-qwen3.6-27b-prismascout.sh`'s unified-memory OOM safety scaffolding (shared
  verbatim with the unsloth preset below).
- `run-qwen3.6-27b-unsloth.sh` serves `unsloth/Qwen3.6-27B-NVFP4` (dense,
  multimodal `qwen3_5`) per unsloth's DGX Spark guide. NVFP4 is auto-detected
  (`compressed-tensors`) so **no `--quantization`**, and `--attention-backend`
  is **left to auto-pick** (multimodal; forcing flashinfer hits the same
  rejection as gemma4). **b12x reality-check (measured, not the guide's
  wording):** this checkpoint is **dense**, so the guide's `--moe-backend
  flashinfer_b12x` is a **no-op here** (no MoE layers) — kept only for guide
  parity / a future MoE variant. The dense NVFP4 GEMM auto-selects
  `FlashInferCutlassNvFp4LinearKernel` (cutlass) — the best available path, **not**
  the marlin W4A16 worst case. Do **not** pass `--linear-backend flashinfer_b12x`:
  this build has no b12x *linear* kernel for the layer type and it **hard-fails on
  boot**. `CUTE_DSL_ARCH=sm_121a` is set per the guide (harmless). Spec decode is
  the model's **built-in MTP head** (`num_speculative_tokens=2`), not DFlash —
  **on by default** (`--no-spec` disables it) because it **measured +79% decode
  (11.3→20.2 tok/s) with tool calling intact**. MTP defaults `GPU_MEM_UTIL` to
  0.52 (vs 0.72 for `--no-spec`). **Reasoning caveat:** the `qwen3` reasoning
  parser *strips and discards* this checkpoint's `<think>…</think>` (leaves
  `reasoning_content` null), and because it buffers until `</think>`, a small
  client `max_tokens` truncates mid-thought → **empty response**. Mitigations
  baked in: `DEFAULT_MAX_TOKENS=4096` (injected as generation-config
  `max_new_tokens`) so no-max_tokens clients don't truncate, and
  `--no-reasoning-parser` to drop the parser entirely so the raw
  `<think>…</think>` is returned verbatim in `content` (client splits it itself;
  tool calling is unaffected). Single-stream ~11 tok/s without MTP is the Spark's
  memory-bandwidth ceiling for a 27B, not a misconfig. Image supports the b12x
  (MoE) kernels at **flashinfer 0.6.16.post3** (guide's ≥0.6.13 is conservative);
  verified on the v0.27.1 image: `has_flashinfer_b12x_gemm()` and
  `has_flashinfer_b12x_moe()` are both `True`, device capability `(12, 1)`.
  Distinct from `run-qwen3.6-27b-prismascout.sh` (PrismaSCOUT + DFlash); reuses that script's
  unified-memory OOM safety scaffolding verbatim.
