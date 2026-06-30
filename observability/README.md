# Observability — Prometheus + Grafana for the vLLM Spark server

A one-command metrics stack for the locally-built vLLM image. vLLM exposes a
Prometheus-compatible `/metrics` endpoint on its OpenAI server; this dir scrapes
it with Prometheus and graphs it with Grafana using **vLLM's own dashboards**.

## Quick start

```bash
# 1. Serve a model (publishes the API + /metrics on host port 8000)
./run.sh Qwen/Qwen3-8B            # from the repo root

# 2. Bring up the metrics stack
cd observability
docker compose up -d

# 3. Open Grafana — dashboards are under the "vLLM" folder, already wired
open http://localhost:3000
```

Grafana runs with anonymous admin access (localhost-only convenience — no login).
Generate some traffic (`curl http://localhost:8000/v1/...`) and the panels fill in.
Pick your model in the **Deployment_ID** dropdown at the top of each dashboard.

## What's here

| File | Purpose |
|------|---------|
| `compose.yml` | Prometheus + Grafana services, persistent named volumes |
| `prometheus.yml` | Scrape config — targets the host's vLLM `/metrics` every 5s |
| `grafana/provisioning/` | Auto-wires the Prometheus datasource + dashboard loader |
| `grafana/dashboards/` | vLLM's official Grafana dashboards (vendored, see below) |

The two dashboards (`performance_statistics.json`, `query_statistics.json`) are
vLLM's own, copied from `examples/observability/dashboards/grafana/` in the vLLM
source tree at the **pinned build commit** so their panel queries match the
metrics this image actually emits.

**`build.sh` refreshes them automatically** to match `$VLLM_COMMIT` at the end of
every build (best-effort — a network hiccup warns but won't fail the build). After
a build that changed them, `docker compose restart grafana` to reload. To refresh
by hand without rebuilding:

```bash
C=$(grep VLLM_COMMIT ../build.lock | cut -d'"' -f2)
base="https://raw.githubusercontent.com/vllm-project/vllm/$C/examples/observability/dashboards/grafana"
curl -fsSL "$base/performance_statistics.json" -o grafana/dashboards/performance_statistics.json
curl -fsSL "$base/query_statistics.json"       -o grafana/dashboards/query_statistics.json
docker compose restart grafana
```

## Networking note

Prometheus runs in a container; the vLLM server is published on the **host**
(`run.sh` maps `-p 8000:8000`). The scrape target is therefore
`host.docker.internal:8000`, mapped to the host gateway via `extra_hosts` in
`compose.yml` (required on Linux). Serving on a different host port? Edit the
target in `prometheus.yml` and `docker compose restart prometheus`.

## Ports / teardown

```bash
GRAFANA_PORT=3001 PROM_PORT=9091 docker compose up -d   # avoid port clashes
docker compose down        # stop (keeps data)
docker compose down -v     # stop + wipe metrics history
```

Verify the scrape target is healthy at `http://localhost:9090/targets`.
