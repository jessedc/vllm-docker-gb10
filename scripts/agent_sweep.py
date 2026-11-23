#!/usr/bin/env python3
"""Concurrency sweep simulating 1 main agent + up to 2 subagents.

c=1: main only; c=2: main + sub1; c=3: main + sub1 + sub2. All requests in a
level start together. Each prompt is real Python stdlib source as "code
context" with a unique nonce prefix (defeats prefix caching across runs).
Fixed output length via ignore_eos so levels are comparable.
"""
import json, os, sys, threading, time, urllib.request, uuid, glob, statistics

URL = os.environ.get("URL", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "thinkingmachines/Inkling-2-Mini")
OUT_TOKENS = int(os.environ.get("OUT_TOKENS", 400))
REPS = int(os.environ.get("REPS", 2))

srcs = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
srcs = [s for s in srcs if os.path.getsize(s) > 4000]

def corpus(start, chars):
    buf, i = [], start
    while sum(map(len, buf)) < chars:
        with open(srcs[i % len(srcs)], errors="ignore") as f:
            buf.append(f"\n# ===== {srcs[i % len(srcs)]} =====\n" + f.read())
        i += 1
    return "".join(buf)[:chars]

# ~3.6 chars/token for Python source
ROLES = {
    "main": (0,   58000, "You are the lead coding agent. Review the codebase below and write a detailed refactoring plan."),
    "sub1": (400, 22000, "You are a subagent. Find every function in the code below that handles errors and explain each."),
    "sub2": (800, 22000, "You are a subagent. Summarize the public API of the modules below in detail."),
}

def run(role, res):
    start, chars, task = ROLES[role]
    body = {
        "model": MODEL, "stream": True, "temperature": 0, "max_tokens": OUT_TOKENS,
        "ignore_eos": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user",
                      "content": f"[session {uuid.uuid4()}]\n{task}\n\n```python\n{corpus(start, chars)}\n```"}],
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter(); stamps = []; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                dl = ch.get("delta", {})
                if dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content"):
                    stamps.append(time.perf_counter())
    res[role] = dict(t0=t0, stamps=stamps, usage=usage)

def level(roles):
    res = {}; th = [threading.Thread(target=run, args=(r, res)) for r in roles]
    for t in th: t.start()
    for t in th: t.join()
    out = {}
    for r in roles:
        x = res[r]; s = x["stamps"]; n = x["usage"]["completion_tokens"]
        ttft = s[0] - x["t0"]; dec = (n - 1) / (s[-1] - s[0])
        out[r] = dict(prompt=x["usage"]["prompt_tokens"], ttft=ttft, decode=dec,
                      e2e=s[-1] - x["t0"], n=n)
    # aggregate decode while ALL streams decode simultaneously
    lo = max(res[r]["stamps"][0] for r in roles); hi = min(res[r]["stamps"][-1] for r in roles)
    cnt = sum(sum(1 for t in res[r]["stamps"] if lo <= t <= hi) for r in roles)
    out["_overlap"] = dict(agg=cnt / (hi - lo) if hi > lo else float("nan"), window=hi - lo,
                           wall=max(o["e2e"] for k, o in out.items() if not k.startswith("_")),
                           total_tokens=sum(out[r]["n"] for r in roles))
    return out

if __name__ == "__main__":
    print("warm-up..."); level(["sub1"])
    levels = [["main"], ["main", "sub1"], ["main", "sub1", "sub2"]]
    allres = {}
    for roles in levels:
        c = len(roles); allres[c] = []
        for rep in range(REPS):
            o = level(roles); allres[c].append(o)
            print(f"c={c} rep={rep}: " + "  ".join(
                f"{r}: pp={o[r]['prompt']} ttft={o[r]['ttft']:.1f}s dec={o[r]['decode']:.1f}t/s"
                for r in roles) + f"  | overlap agg={o['_overlap']['agg']:.1f}t/s over {o['_overlap']['window']:.1f}s"
                f"  wall={o['_overlap']['wall']:.1f}s", flush=True)
    json.dump(allres, open(sys.argv[1] if len(sys.argv) > 1 else "sweep.json", "w"), indent=1)
