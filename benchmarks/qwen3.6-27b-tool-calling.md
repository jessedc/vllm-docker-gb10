# Benchmark: Qwen3.6-27B NVFP4 — multi-turn tool calling on the DGX Spark

A head-to-head of the two dense 27B NVFP4 presets in this repo, exercising **real
multi-round tool calling** (the model calls a tool → the harness executes it and
feeds the result back → repeat) and measuring both **throughput** and
**tool-calling accuracy**.

| Preset | Model | Spec decode |
|--------|-------|-------------|
| [`run-qwen3.6-27b-unsloth.sh`](../run-qwen3.6-27b-unsloth.sh) | `unsloth/Qwen3.6-27B-NVFP4` | built-in **MTP** head (default) |
| [`run-qwen3.6-27b-prismascout.sh`](../run-qwen3.6-27b-prismascout.sh) | `rdtand/…PrismaSCOUT…NVFP4` | external **DFlash** drafter (default) |

Both were served with **their script defaults** (sampling, spec-decode, memory
util — nothing overridden) on the GB10 / sm_121a, one at a time on `:8000`.

## How to reproduce

```bash
# 1. serve one preset (warm caches → ~150 s to ready, no OOM)
DETACH=1 ./run-qwen3.6-27b-unsloth.sh          # or ./run-qwen3.6-27b-prismascout.sh

# 2. run the harness (auto-detects the served model id)
python3 benchmarks/toolbench.py --trials 3 --out benchmarks/results-x.json

# 3. swap presets and repeat
```

[`toolbench.py`](./toolbench.py) needs only `requests`. It defines 8 tools
(`add`/`multiply`/`divide`/`power`/`sqrt`/`convert_currency`/…) with deterministic
implementations, then runs 5 agentic tasks × 3 trials. Each task requires **2–4
sequential or parallel tool calls** (arithmetic chains, a USD→EUR→JPY conversion
chain, a population→add→divide aggregation), so the harness grades the whole
tool-calling loop, not a single call.

## Results

5 tasks × 3 trials = **15 task-runs per model**.

| Metric | **unsloth NVFP4** (MTP) | **PrismaSCOUT** (DFlash) |
|---|---|---|
| Final-answer accuracy | **15/15 (100%)** | 14/15 (93.3%) |
| Tool-selection accuracy | **100%** | 93.3% |
| Malformed-JSON tool calls | 0 / 36 | 0 / 34 |
| Hallucinated tool names | 0 | 0 |
| **End-to-end throughput** (tok/s, wall-clock) | 21.6 | **41.1** |
| Per-task latency — all runs | 20.4 s | 17.9 s |
| Per-task latency — successful only | 20.4 s | **14.1 s** |
| Tokens generated (total) | 6,628 | 11,040 |

Raw per-run data: [`results-unsloth-nvfp4-mtp.json`](./results-unsloth-nvfp4-mtp.json),
[`results-prismascout-dflash.json`](./results-prismascout-dflash.json).

### Tokens/second

- **PrismaSCOUT (DFlash) generates ~1.9× faster** end-to-end (41 vs 22 tok/s
  wall-clock). DFlash's block-diffusion speculation beats the unsloth
  checkpoint's 2-token MTP head. unsloth's rate was *rock-steady* across every
  task (20.5–22.6 tok/s — the Spark's memory-bandwidth ceiling for a 27B);
  PrismaSCOUT varied more (29–40) because DFlash acceptance depends on how
  predictable the output is.
- Because PrismaSCOUT **thinks more** (11k vs 6.6k tokens for the same tasks), its
  ~2× speed edge only becomes a modest *per-task* latency win once you count
  everything (17.9 vs 20.4 s). On the runs where it didn't derail it's clearly
  faster (~14 vs ~20 s).
- **Measurement caveat — "decode tok/s" is not reported as a headline.** The
  qwen3 reasoning parser buffers the entire `<think>` block server-side and
  delivers it in a burst, so instantaneous decode rate reads as 100–650 tok/s and
  is meaningless. **End-to-end wall-clock throughput is the honest number** and is
  what the table shows.

### Tool-calling accuracy

- **When either model emits a tool call, the mechanics are flawless:** every one
  of the ~70 tool calls had valid JSON arguments, correct values, the right tool
  selected, and a correct final answer. Multi-step chains (multiply→divide,
  power→sqrt, USD→EUR→JPY, population→add→divide) and parallel calls all resolved.
- **unsloth NVFP4 was perfect (15/15).**
- **PrismaSCOUT failed 1/15 — a reproducible failure mode** (hit in two
  independent runs, always on the *simplest* task, `chain_mul_div`): the model
  falls into a runaway `<think>` and hits the 4096-token cap **before ever
  emitting a tool call** → empty/truncated result. This is exactly the
  reasoning-truncation risk the `run-qwen3.6-27b*.sh` header notes call out for
  these Qwen3.6 checkpoints; it over-thinks "128×4 then ÷8" and never stops.

## Follow-up: does `--no-reasoning-parser` fix PrismaSCOUT's stall?

**No — it makes tool calling substantially worse.** Re-ran the identical suite
against PrismaSCOUT served with `--no-reasoning-parser` (raw thinking returned in
`content` instead of being parsed out).
Raw data: [`results-prismascout-dflash-no-reasoning.json`](./results-prismascout-dflash-no-reasoning.json).

| Metric | Parser **ON** (default) | Parser **OFF** |
|---|---|---|
| Final-answer accuracy | 93.3% (14/15) | **80.0% (12/15)** |
| Tool-selection accuracy | 93.3% | **46.7% (7/15)** |
| Tool calls emitted | 34 | **16** |
| End-to-end throughput | 41.1 tok/s | 33.5 tok/s |
| First-round TTFT | 8.9 s | **0.39 s** |

Without the reasoning channel the model **rambles its plan in prose** ("*I need to
use the `convert_currency` tool. First call:…*") and frequently **never emits the
structured tool call** (`rounds=1`, no calls) — it answers inline (fine for easy
arithmetic, wrong for the currency/population tasks that need the tool's data). It
halves tool usage and drops selection accuracy from 93% → 47%. The only win is
TTFT (thinking streams immediately rather than buffering to `</think>`), which
does not offset the accuracy loss. **Keep the reasoning parser ON for agentic tool
calling.** (`--no-reasoning-parser` remains available on the script for a
non-tool-calling client that wants the raw thinking — e.g. opencode.)

## Bottom line

- Want **max throughput / lowest latency** and can tolerate an occasional
  over-thinking stall → **PrismaSCOUT + DFlash** (~2× tok/s), reasoning parser ON.
- Want **reliability** for agentic loops where a stalled turn breaks the chain →
  **unsloth NVFP4 + MTP** was 100% dependable at ~21.6 tok/s.
- The right lever for PrismaSCOUT's runaway-think stall is **raising the per-turn
  token budget** (`DEFAULT_MAX_TOKENS` / client `max_tokens`) so a long think can
  finish and still emit the tool call — **not** dropping the reasoning parser,
  which (measured above) degrades tool calling.

---

*Method notes: sampling left at each script's server-side defaults (so accuracy
reflects real default-temperature behavior); `n=15` per model is enough to
separate the throughput tiers cleanly but is a small sample for the accuracy
delta — the PrismaSCOUT failure reproduced across both runs on the same task,
which is the stronger signal. Both models booted from warm caches in ~150–165 s
with no unified-memory OOM.*

---

## Follow-up: DFlash `SPEC_TOKENS` sweep on the rebuilt image (`2396a611`)

Rebuilt the vLLM image from a newer `main` (commit `2396a611`, flashinfer
`0.6.14`) to pull in [#47914](https://github.com/vllm-project/vllm/pull/47914),
which reworks hybrid sliding-window/full-attention DFlash drafters on the V2
runner. Two things fell out of serving PrismaSCOUT on it:

- **DFlash now runs on `FLASH_ATTN`, not FlashInfer.** The drafter needs
  *non-causal* attention, which only `FLASH_ATTN`/`FLEX_ATTENTION` support
  ([vllm#41559](https://github.com/vllm-project/vllm/issues/41559)); the engine
  auto-selects it and **overrides** the script's `--attention-backend flashinfer`.
  The KV cache therefore stays **bf16** regardless of `--kv-cache-dtype`. (The
  flag is still honoured for `--mtp`/`--no-spec`.)
- **`SPEC_TOKENS=10` beats `5` on this box.** Re-ran the identical toolbench suite
  (5 tasks × 3 trials) at each setting. Spec-decode counters from `/metrics`:

| Metric | `SPEC_TOKENS=5` | **`SPEC_TOKENS=10`** |
|---|---|---|
| Acceptance **rate** | 64.3% | 41.3% |
| Accepted **per cycle** | 3.21 / 5 | 4.13 / 10 |
| **Mean accept length** (tok / model forward) | 4.21 | **5.13** |
| e2e throughput (tok/s wall-clock) | 31.6 | **34.4** |
| **Mean wall / task** | 13.1 s | **10.2 s** |
| Final-answer accuracy | 15/15 | 14/15 |
| Tool-selection accuracy | 100% | 100% |

Per-position acceptance (fraction of draft cycles accepting position *i*):

```
n=5   p0:90% p1:76% p2:63% p3:51% p4:41%
n=10  p0:90% p1:75% p2:60% p3:49% p4:37% p5:30% p6:25% p7:20% p8:14% p9:11%
```

**Takeaway — accept *length*, not *rate*, is what drives speed.** `n=5` posts a
higher headline acceptance *rate* only because it averages over fewer, easier
positions. DFlash's block-diffusion drafter fills the whole block cheaply, so at
`n=10` even positions 5–9 keep accepting (30→11%), buying ~0.9 more tokens per
expensive target forward (5.13 vs 4.21) and ~22% lower per-task latency. The
default is therefore `SPEC_TOKENS=10`. (Off-box guides tune lower — e.g. z-lab
n=5 on an RTX — but that didn't transfer to the GB10.) The 15/15-vs-14/15 accuracy
difference is stochastic (temp 0.7); spec decoding is output-distribution-neutral,
so it isn't a quality signal. Raw data:
[`results-prismascout-dflash-2396a611.json`](./results-prismascout-dflash-2396a611.json)
(n=5),
[`results-prismascout-dflash-2396a611-n10.json`](./results-prismascout-dflash-2396a611-n10.json)
(n=10).

*Caveat: not a controlled before/after vs the pre-rebuild image — the original
"~51% / 5.1 per-cycle" figure was a different workload with ~2× the tokens, so
these numbers stand on their own rather than proving the rebuild raised
acceptance.*
