#!/usr/bin/env python3
"""Multi-turn tool-calling benchmark for a vLLM OpenAI-compatible server.

Measures, over agentic tasks that each require several sequential/parallel tool
calls:
  * decode throughput (tokens/sec), via streaming + usage accounting
  * time-to-first-token (TTFT)
  * tool-calling accuracy: correct tool selection, valid-JSON args, final answer

Sampling is left to the server-side defaults set by the runner script
(temperature/top_p/etc.), honoring "use the defaults of the runner scripts".
"""
import json, sys, time, math, statistics, argparse
import requests

BASE = "http://localhost:8000/v1"

# ---------------------------------------------------------------- tool impls
def _add(a, b): return a + b
def _subtract(a, b): return a - b
def _multiply(a, b): return a * b
def _divide(a, b): return a / b
def _power(base, exp): return base ** exp
def _sqrt(x): return math.sqrt(x)

# fixed deterministic tables so the grader knows the ground truth
_RATES = {("USD", "EUR"): 0.90, ("EUR", "JPY"): 165.0, ("USD", "JPY"): 148.0}
def _convert_currency(amount, from_ccy, to_ccy):
    r = _RATES[(from_ccy.upper(), to_ccy.upper())]
    return round(amount * r, 4)

_POP = {"tokyo": 37000000, "paris": 11000000, "london": 9000000}
def _get_city_population(city):
    return _POP[city.strip().lower()]

IMPL = {
    "add": _add, "subtract": _subtract, "multiply": _multiply,
    "divide": _divide, "power": _power, "sqrt": _sqrt,
    "convert_currency": _convert_currency, "get_city_population": _get_city_population,
}

def _num(t, props):  # numeric param
    return {"type": "number", "description": t}
TOOLS = [
    {"type": "function", "function": {"name": "add", "description": "Add two numbers a+b",
        "parameters": {"type": "object", "properties": {"a": _num("first", 0), "b": _num("second", 0)}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "subtract", "description": "Subtract b from a (a-b)",
        "parameters": {"type": "object", "properties": {"a": _num("first", 0), "b": _num("second", 0)}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers a*b",
        "parameters": {"type": "object", "properties": {"a": _num("first", 0), "b": _num("second", 0)}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "divide", "description": "Divide a by b (a/b)",
        "parameters": {"type": "object", "properties": {"a": _num("numerator", 0), "b": _num("denominator", 0)}, "required": ["a", "b"]}}},
    {"type": "function", "function": {"name": "power", "description": "Raise base to the power exp",
        "parameters": {"type": "object", "properties": {"base": _num("base", 0), "exp": _num("exponent", 0)}, "required": ["base", "exp"]}}},
    {"type": "function", "function": {"name": "sqrt", "description": "Square root of x",
        "parameters": {"type": "object", "properties": {"x": _num("value", 0)}, "required": ["x"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount of money between currencies",
        "parameters": {"type": "object", "properties": {"amount": _num("amount", 0),
            "from_ccy": {"type": "string", "description": "source ISO currency e.g. USD"},
            "to_ccy": {"type": "string", "description": "target ISO currency e.g. EUR"}}, "required": ["amount", "from_ccy", "to_ccy"]}}},
    {"type": "function", "function": {"name": "get_city_population", "description": "Return the population of a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}}, "required": ["city"]}}},
]

# ---------------------------------------------------------------- tasks
# expected_tools: multiset of tool names a correct solution must call
# check: fn(final_text) -> bool  (is the final numeric answer correct)
def _has(text, value, tol=1e-6):
    """True if `value` appears in text (comma-insensitive, small tolerance)."""
    import re
    t = text.replace(",", "")
    for m in re.findall(r"-?\d+\.?\d*", t):
        try:
            if abs(float(m) - value) <= max(tol, abs(value) * 1e-4):
                return True
        except ValueError:
            pass
    return False

TASKS = [
    {"id": "chain_mul_div", "expected_tools": ["multiply", "divide"],
     "prompt": "Using the tools, multiply 128 by 4, then divide that result by 8. Give only the final number.",
     "check": lambda t: _has(t, 64)},
    {"id": "chain_pow_sqrt", "expected_tools": ["power", "sqrt"],
     "prompt": "Using the tools, compute 2 raised to the power 10, then take the square root of that result. Give only the final number.",
     "check": lambda t: _has(t, 32)},
    {"id": "parallel_add_mul", "expected_tools": ["add", "multiply"],
     "prompt": "Using the tools, compute 45 plus 55, and separately compute 9 times 9. Report both resulting numbers.",
     "check": lambda t: _has(t, 100) and _has(t, 81)},
    {"id": "currency_chain", "expected_tools": ["convert_currency", "convert_currency"],
     "prompt": "I have 300 USD. Using the convert_currency tool, first convert it to EUR, then convert that EUR amount to JPY. Report the final JPY amount.",
     "check": lambda t: _has(t, round(300 * 0.90 * 165.0, 4)) or _has(t, round(300 * 0.90 * 165.0))},
    {"id": "population_agg", "expected_tools": ["get_city_population", "get_city_population", "add", "divide"],
     "prompt": "Using the tools, add the population of Tokyo and the population of Paris, then divide the total by 2. Give only the final number.",
     "check": lambda t: _has(t, (37000000 + 11000000) / 2)},
]

# ---------------------------------------------------------------- streaming call
def stream_chat(model, messages, max_tokens=4096):
    """One streamed chat completion. Returns dict with assembled message,
    finish_reason, usage, ttft, decode_time."""
    payload = {"model": model, "messages": messages, "tools": TOOLS,
               "tool_choice": "auto", "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    ttft = None
    content = ""
    reasoning = ""
    tool_calls = {}  # index -> {id,name,args}
    finish = None
    usage = None
    with requests.post(BASE + "/chat/completions", json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {}) or {}
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                got = False
                if delta.get("reasoning_content"):
                    reasoning += delta["reasoning_content"]; got = True
                if delta.get("content"):
                    content += delta["content"]; got = True
                for tc in (delta.get("tool_calls") or []):
                    got = True
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(idx, {"id": None, "name": "", "args": ""})
                    if tc.get("id"): slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"): slot["name"] += fn["name"]
                    if fn.get("arguments"): slot["args"] += fn["arguments"]
                if got and ttft is None:
                    ttft = time.perf_counter() - t0
    end = time.perf_counter()
    calls = [tool_calls[i] for i in sorted(tool_calls)]
    return {"content": content, "reasoning": reasoning, "tool_calls": calls, "finish": finish,
            "usage": usage or {}, "ttft": ttft if ttft is not None else (end - t0),
            "decode_time": max(end - t0 - (ttft or 0), 1e-6), "wall": end - t0}

# ---------------------------------------------------------------- one task run
def run_task(model, task, max_rounds=8):
    messages = [
        {"role": "system", "content": "You are a precise assistant. Use the provided tools to do all arithmetic and lookups; do not compute in your head. When you have the answer, state it clearly."},
        {"role": "user", "content": task["prompt"]},
    ]
    called = []           # tool names actually invoked
    bad_json = 0          # tool calls with unparseable args
    unknown_tool = 0      # hallucinated tool names
    completion_tokens = 0
    decode_time = 0.0
    wall_time = 0.0
    ttfts = []
    rounds = 0
    final_text = ""
    err = None
    for _ in range(max_rounds):
        rounds += 1
        try:
            res = stream_chat(model, messages)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"; break
        completion_tokens += (res["usage"].get("completion_tokens") or 0)
        decode_time += res["decode_time"]
        wall_time += res["wall"]
        ttfts.append(res["ttft"])
        if res["tool_calls"]:
            # assistant turn that requested tools -> execute & feed back
            assistant_msg = {"role": "assistant", "content": res["content"] or None,
                             "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                                             "function": {"name": c["name"], "arguments": c["args"]}}
                                            for i, c in enumerate(res["tool_calls"])]}
            messages.append(assistant_msg)
            for c in res["tool_calls"]:
                name = c["name"]; called.append(name)
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    bad_json += 1; args = {}
                if name not in IMPL:
                    unknown_tool += 1
                    result = {"error": f"unknown tool {name}"}
                else:
                    try:
                        result = {"result": IMPL[name](**args)}
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                messages.append({"role": "tool", "tool_call_id": c["id"] or f"call_{len(called)}",
                                 "name": name, "content": json.dumps(result)})
            continue
        # no tool calls -> final answer
        final_text = res["content"]
        break

    correct = bool(task["check"](final_text)) if final_text and not err else False
    # tool-selection accuracy: expected multiset ⊆ called multiset
    from collections import Counter
    exp = Counter(task["expected_tools"]); got = Counter(called)
    tool_selection_ok = all(got[k] >= v for k, v in exp.items())
    return {
        "task": task["id"], "correct": correct, "tool_selection_ok": tool_selection_ok,
        "called": called, "n_tool_calls": len(called), "bad_json": bad_json,
        "unknown_tool": unknown_tool, "rounds": rounds,
        "completion_tokens": completion_tokens, "decode_time": decode_time,
        "wall_time": wall_time,
        "decode_tps": completion_tokens / decode_time if decode_time else 0.0,
        "e2e_tps": completion_tokens / wall_time if wall_time else 0.0,
        "sum_ttft": sum(t for t in ttfts if t),
        "ttft_first": ttfts[0] if ttfts else None, "final_text": final_text[:400], "err": err,
    }

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    model = requests.get(BASE + "/models", timeout=30).json()["data"][0]["id"]
    print(f"# model: {model}   trials/task: {args.trials}", flush=True)

    results = []
    for trial in range(args.trials):
        for task in TASKS:
            r = run_task(model, task)
            r["trial"] = trial
            results.append(r)
            flag = "OK " if r["correct"] else "XX "
            print(f"{flag}t{trial} {r['task']:<16} correct={r['correct']} sel_ok={r['tool_selection_ok']} "
                  f"e2e_tps={r['e2e_tps']:.1f} decode_tps={r['decode_tps']:.0f} "
                  f"wall={r['wall_time']:.1f}s ttft_sum={r['sum_ttft']:.1f}s "
                  f"ctoks={r['completion_tokens']} rounds={r['rounds']} "
                  f"badjson={r['bad_json']} unk={r['unknown_tool']}"
                  + (f" ERR={r['err']}" if r['err'] else ""), flush=True)

    # aggregates
    n = len(results)
    tot_ctok = sum(r["completion_tokens"] for r in results)
    tot_dt = sum(r["decode_time"] for r in results)
    tot_wall = sum(r["wall_time"] for r in results)
    agg = {
        "label": args.label, "model": model, "n_task_runs": n,
        "accuracy_final_answer": sum(r["correct"] for r in results) / n,
        "tool_selection_accuracy": sum(r["tool_selection_ok"] for r in results) / n,
        "total_tool_calls": sum(r["n_tool_calls"] for r in results),
        "bad_json_calls": sum(r["bad_json"] for r in results),
        "unknown_tool_calls": sum(r["unknown_tool"] for r in results),
        # end-to-end wall throughput = the honest number (counts prefill/TTFT of every round)
        "e2e_tps_wallclock": tot_ctok / tot_wall if tot_wall else 0.0,
        "mean_task_wall_s": statistics.mean(r["wall_time"] for r in results),
        # decode-only tps: smooth for MTP, inflated/bursty for DFlash block-diffusion
        "aggregate_decode_tps": tot_ctok / tot_dt if tot_dt else 0.0,
        "mean_ttft_s_first_round": statistics.mean(r["ttft_first"] for r in results if r["ttft_first"]),
        "total_completion_tokens": tot_ctok,
        "total_wall_s": tot_wall,
        "total_decode_time_s": tot_dt,
    }
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"{k:28}: {v}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"aggregate": agg, "runs": results}, f, indent=2)
        print(f"\nwrote {args.out}")

if __name__ == "__main__":
    main()
