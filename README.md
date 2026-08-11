# OpenAI-Compatible Router Node for Ollama LLM

This repository runs an OpenAI-compatible router with Qdrant vector database for LAN access by LibreChat (hosted elsewhere). The router connects to **any Ollama instance** and applies per-model policy routing, context management, and think-control for all models served by that backend.

This node is single-tenant by design: do not run any other GPU workload or heavyweight CPU workload here. Extra containers, dev tools, or interactive sessions that touch the GPU break the memory budget.

## Works with Any Ollama-Served Model

This router is **hardware-agnostic** and works with any model available through your Ollama backend. The `OLLAMA_BASE_URL` environment variable (set in [`profiles/router.env`](profiles/router.env)) points to your Ollama instance, and the router applies policy rules to all models listed in the active profile's configuration.

You can:

- Use any model pulled into your Ollama instance
- Define custom model policies by editing the profile's `models.yaml`
- Run Ollama on the same machine or a remote server
- Mix and match models from the Qwen, Gemma, Devstral, Granite, and other families

## Profiles: Orin and Thor (Tested Examples)

The **Orin** and **Thor** profiles are pre-configured, tested examples for NVIDIA Jetson hardware. They demonstrate the router's capabilities but are not limitations — you can use these profiles as templates for your own custom configurations or create entirely new profiles.

| Profile | Target Board | RAM | CUDA Arch | Base Image | Status |
| --- | --- | --- | --- | --- | --- |
| `orin` (default) | Jetson AGX Orin 64 GB | 64 GB unified | SM 87 (Ampere) | `ubuntu:24.04` | Tested and working |
| `thor` | Jetson Thor | 64+ GB unified | SM 110 (Blackwell) | `ubuntu:24.04` | Tested and working |

Select the profile at runtime via the `HARDWARE_PROFILE` environment variable (default: `orin`). To use a custom configuration, set `HARDWARE_PROFILE` to a directory containing your `models.yaml` file, or edit the default profile directly.

### Profile File Layout

```
profiles/
├── router.env               # Router-specific runtime variables
├── qdrant.env               # Qdrant-specific runtime variables
├── jetson/
│   ├── orin.env             # Orin hardware-specific settings (MODEL_DEFAULT, etc.)
│   └── thor.env             # Thor hardware-specific settings (MODEL_DEFAULT, etc.)
├── orin/
│   └── models.yaml          # Orin model definitions and policy
└── thor/
    └── models.yaml          # Thor model definitions and policy
```

The active profile's `models.yaml` is mounted into the router container at `/app/model_policy.yml`.

### Environment File Structure

Environment variables are organized by concern and loaded via Docker Compose `env_file` directives:

| File | Purpose | Loaded By |
| --- | --- | --- |
| `.env.example` | Docker Compose variables (QDRANT_IMAGE, HARDWARE_PROFILE) | N/A (template only) |
| `profiles/router.env` | Router runtime variables (OLLAMA_BASE_URL, KEEP_ALIVE_DEFAULT, ROUTER_INTERNAL_PORT, etc.) | fastapi-router service |
| `profiles/qdrant.env` | Qdrant runtime variables (QDRANT_URL, QDRANT_TOP_K, etc.) | qdrant service |
| `profiles/jetson/<profile>.env` | Hardware-specific overrides (MODEL_DEFAULT, EMBEDDING_MODEL_DEFAULT) | Both services |

To deploy, copy `.env.example` to `.env` and adjust values for your environment. The active hardware profile is selected via `HARDWARE_PROFILE=orin` or `HARDWARE_PROFILE=thor`.

## Files

- [`docker-compose.yml`](docker-compose.yml): OpenAI-compatible router and optional Qdrant vector database.
- [`model_policy.yml`](model_policy.yml): Default policy file; the runtime profile overrides this via mount.
- [`.env.example`](.env.example): environment values for ports, profile selection, and model behavior.

## Network and Security Notes

### Port Configuration

The router uses two distinct port concepts for flexibility in deployment:

| Variable | Scope | Purpose | Default |
| --- | --- | --- | --- |
| `ROUTER_INTERNAL_PORT` | Container runtime (via [`profiles/router.env`](profiles/router.env)) | The port Uvicorn listens on inside the container | `4000` |
| `ROUTER_PORT` | Docker Compose host-side mapping (via `.env`) | The external port mapped to the internal container port | `4000` |
| `ROUTER_BIND_IP` | Docker Compose host-side binding (via `.env`) | The network interface on which the host port is exposed | `0.0.0.0` |

The Docker Compose port mapping follows this pattern: `${ROUTER_BIND_IP:-0.0.0.0}:${ROUTER_PORT:-4000}:${ROUTER_INTERNAL_PORT:-4000}` (host IP:host port:container port). This allows the internal container port to differ from the externally exposed host port when needed.

### Exposed Ports

- `0.0.0.0:<ROUTER_PORT>` (OpenAI-compatible router, default `4000`)
- `0.0.0.0:<QDRANT_PORT>` (Qdrant vector database, optional, default `6333`)
- Restrict access at host firewall/router ACLs to trusted LAN clients.

## Start Services

```bash
cp .env.example .env
```

Edit `.env` and set the desired profile:

```bash
HARDWARE_PROFILE=orin   # or HARDWARE_PROFILE=thor
```

Start the router stack:

```bash
docker compose up -d
```

Optional proxy build (profile-aware):

```bash
HARDWARE_PROFILE=orin docker compose -f router/docker-compose.yml --profile proxy up -d --build
# or
HARDWARE_PROFILE=thor docker compose -f router/docker-compose.yml --profile proxy up -d --build
```

Qdrant vector database deployment is controlled via the `QDRANT_IMAGE` variable in `.env`:

| QDRANT_IMAGE Value | GPU Support | Architecture | Host Requirements |
| --- | --- | --- | --- |
| `qdrant/qdrant:latest` (default) | No | AMD64/ARM64 | None |
| `qdrant/qdrant:gpu-nvidia-latest` | Yes (NVIDIA) | AMD64 | `nvidia-container-toolkit` |
| `qdrant/qdrant:gpu-amd-latest` | Yes (AMD/Vulkan) | AMD64 | `vulkan-tools`, `vulkan-radeon` |

To enable GPU acceleration, edit `.env` and set the appropriate `QDRANT_IMAGE` value, then run:

```bash
docker compose up -d
```

See [GPU Acceleration Support](#gpu-acceleration-support) for detailed configuration.

## Models

### Orin Profile (64 GB Unified Memory)

The Orin 64 GB node is sized to keep two MoE models warm by default while leaving headroom for KV cache, the router, and the OS.

| Model ID | Purpose | Default keep_alive | Default think | Supported `num_ctx` Ceiling | Notes |
| --- | --- | --- | --- | --- | --- |
| `qwen3-coder:30b` | Strict-JSON and structured-output workloads for boundary selection and cue-ID extraction. | `-1` | `false` | `65536` | Primary coding model; stays warm. |
| `qwen3.6:35b-a3b` | Default chat and narrative summarization workloads. | `10m` | `true` | `32768` | Hybrid attention keeps KV usage comparatively small. |
| `qwen3:4b` | Fast assistant / routing model. | `10m` | `true` | `65536` | Lightweight fallback for simple queries. |
| `qwen3-vl:4b` | Vision-language tasks (image understanding). | `10m` | `true` | `65536` | Multimodal input support. |
| `gemma4:12b` | Mid-tier general-purpose model. | `10m` | `true` | `65536` | Alternative for creative tasks. |
| `qwen3-embedding:4b` | Embedding service for Qdrant retrieval. | `5m` | `false` | `8192` | Used by retrieval pipeline. |
| `nemotron-cascade-2:30b-a3b-q4_K_M` | Optional reasoning verifier for ambiguous structured answers. | `10m` | `true` | `65536` | Only resident while actively in use; expect one warm-model eviction when it loads. |
| `gpt-oss:20b` | Secondary general-purpose model. | `10m` | `true` | `65536` | Fallback for diverse tasks. |
| `laguna-xs-2.1:q4_K_M` | Lightweight reasoning / utility model. | `10m` | `true` | `65536` | Small-footprint helper. |

#### Orin Budget Math

The two-warm plan is only valid at Q4_K_M with `q8_0` KV cache. The external LLM backend (Ollama) manages model residency via `keep_alive` settings.

**WARNING:** If a model is loaded without an explicit `num_ctx`, the external LLM backend will use its native context length (256K+ for these models), which inflates the resident footprint to about 33 GB per model and breaks the two-warm budget. Always set `num_ctx` per call.

| Component | Expected residency |
| --- | --- |
| `qwen3-coder:30b` weights | ~18-19 GB |
| `qwen3.6:35b-a3b` weights | ~22-24 GB |
| KV cache (`16K` detect, `32K` summary) | ~3-5 GB combined |
| CUDA + Ollama runtime (external) | ~2-3 GB |
| OS + Docker + router + misc | ~3-5 GB |
| Resident total | ~49-56 GB |
| Headroom on 64 GB | ~8-15 GB |

### Thor Profile 110 / Blackwell)

The Thor profile targets larger context windows and a broader model catalog, leveraging the newer Blackwell architecture.

| Model ID | Purpose | Default keep_alive | Default think | Supported `num_ctx` Ceiling | Notes |
| --- | --- | --- | --- | --- | --- |
| `qwen3-coder-next:q4_K_M` | Next-gen coding model for structured-output workloads. | `45m` | `false` | `262144` | Primary coding model; stays warm with large context. |
| `qwen3.6:35b-a3b-q8_0` | Default chat and narrative summarization workloads. | `-1` | `true` | `262144` | Stays warm; higher precision KV cache. |
| `gemma4:31b-it-q4_K_M` | Large general-purpose model. | `20m` | `true` | `131072` | Heavyweight but capable for complex tasks. |
| `devstral-small-2:24b-instruct-2512-q8_0` | French-language and multilingual instruction model. | `15m` | `true` | `131072` | Specialized language support. |
| `north-mini-code-1.0:q8_0` | Compact coding assistant. | `15m` | `true` | `131072` | Fast code generation. |
| `granite4.1-guardian:8b-q6_K` | Guardrail / safety classification model. | `20m` | `false` | `65536` | Non-thinking safety filter. |
| `qwen3:4b` | Fast assistant / routing model. | `30m` | `true` | `65536` | Lightweight fallback. |
| `qwen3-vl:4b` | Vision-language tasks (image understanding). | `30m` | `true` | `65536` | Multimodal input support. |
| `gemma4:12b` | Mid-tier general-purpose model. | `20m` | `true` | `65536` | Creative tasks alternative. |
| `reader-lm:1.5b` | Document reading / extraction specialist. | `20m` | `false` | `65536` | Non-thinking document processing. |
| `qwen3-embedding:4b` | Embedding service for Qdrant retrieval. | `30m` | `false` | `8192` | Used by retrieval pipeline. |
| `nemotron-cascade-2:30b-a3b-q4_K_M` | Optional reasoning verifier. | `15m` | `true` | `131072` | Reasoning verification. |
| `gpt-oss:20b` | Secondary general-purpose model. | `15m` | `true` | `131072` | Diverse task fallback. |
| `laguna-xs-2.1:q4_K_M` | Lightweight reasoning / utility model. | `-1` (stay warm) | `true` | `131072` | Small-footprint helper. |

## Operator Host Setup

Apply these host-level settings before you rely on the documented memory budget.

### Disable JetPack zram

Ubuntu for Jetson commonly enables `nvzramconfig.service`, which creates a zram swap device around half of system RAM. That is acceptable for bursty workloads, but it is hostile to resident LLM weights on unified memory because the kernel will start compressing and faulting model pages instead of leaving them GPU-accessible, which causes major latency spikes and intermittent upstream `500` errors under pressure.

Check the exact unit name on this JetPack build before changing it:

```bash
systemctl list-unit-files | grep -i zram
```

Disable it persistently on the host:

```bash
sudo systemctl disable --now nvzramconfig.service
sudo swapoff -a
# Optionally remove the unit file or mask it:
sudo systemctl mask nvzramconfig.service
```

Verify that swap is gone:

```bash
swapon --show
free -h
```

Do not attempt to manage zram from inside the container.

### Lock Jetson power and clocks

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

If the packaged `jetson_clocks` unit exists, enable it at boot:

```bash
sudo systemctl enable jetson_clocks
```

If that unit is not present on this JetPack version, create a simple oneshot service using the path reported by `command -v jetson_clocks`:

```ini
# /etc/systemd/system/jetson-clocks.service
[Unit]
Description=Lock Jetson clocks to maximum
After=multi-user.target

[Service]
Type=oneshot
ExecStart=<output of command -v jetson_clocks>
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Then enable it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now jetson-clocks.service
```

Verify the power model and clocks:

```bash
sudo nvpmodel -q
sudo jetson_clocks --show
```

### Kernel VM tunables for LLM residency

Create a sysctl drop-in on the host:

```conf
# /etc/sysctl.d/90-llm.conf
vm.swappiness = 10
vm.overcommit_memory = 1
```

- `vm.swappiness = 10`: strongly prefer keeping resident model pages in RAM instead of swapping under moderate pressure.
- `vm.overcommit_memory = 1`: allow the allocator to reserve memory for large model loads without conservative overcommit rejections.

Apply the settings:

```bash
sudo sysctl --system
```

## Operator Setup

Configure your external LLM backend (Ollama) by setting `OLLAMA_BASE_URL` in [`profiles/router.env`](profiles/router.env) to point to your Ollama instance. This can be a local or remote Ollama server — the router works with any Ollama endpoint.

```bash
OLLAMA_BASE_URL=http://YOUR_OLLAMA_HOST:11434
```

Pull models on your Ollama host. The lists below show the example models configured in each tested profile, but you can pull **any model** supported by Ollama and use it with this router.

### Automatic Model Pulling

The router can automatically pull missing models during startup warmup. When a model defined in `models.yaml` is not found on the Ollama host (HTTP 404), the router will:

1. Detect the missing model during warmup
2. Call Ollama's `/api/pull` endpoint to download the model
3. Retry the warmup request after the pull completes
4. Log success or failure appropriately

This behavior is controlled by the `AUTO_PULL_MISSING_MODELS` environment variable:

| Variable | Default | Description |
| --- | --- | --- |
| `AUTO_PULL_MISSING_MODELS` | `false` | Set to `true` to enable automatic model pulling during warmup and request preflight |
| `MODEL_PULL_TIMEOUT_SEC` | `7200` | Maximum time in seconds to wait for a model pull to complete |

**Note:** When `AUTO_PULL_MISSING_MODELS=false` (default), missing models are logged as warnings but not pulled automatically. You must manually pull models on the Ollama host before starting the router.

**Troubleshooting:** If warmup logs show `404 Not Found` for models defined in `models.yaml`, either:
- Pull the models manually on the Ollama host: `ollama pull <model-name>`
- Set `AUTO_PULL_MISSING_MODELS=true` in `profiles/router.env` to enable automatic pulling

### Example Models — Orin Profile

These are the models pre-configured in the Orin profile:

```bash
ollama pull qwen3-coder:30b
ollama pull qwen3.6:35b-a3b-q8_0
ollama pull qwen3:4b
ollama pull qwen3-vl:4b
ollama pull gemma4:12b
ollama pull qwen3-embedding:4b
# Optional models:
ollama pull nemotron-cascade-2:30b-a3b-q4_K_M
ollama pull gpt-oss:20b
ollama pull laguna-xs-2.1:q4_K_M
```

### Example Models — Thor Profile

These are the models pre-configured in the Thor profile:

```bash
ollama pull qwen3-coder-next:q4_K_M
ollama pull qwen3.6:35b-a3b-q8_0
ollama pull gemma4:31b-it-q4_K_M
ollama pull devstral-small-2:24b-instruct-2512-q8_0
ollama pull north-mini-code-1.0:q8_0
ollama pull granite4.1-guardian:8b-q6_K
ollama pull qwen3:4b
ollama pull qwen3-vl:4b
ollama pull gemma4:12b
ollama pull reader-lm:1.5b
ollama pull qwen3-embedding:4b
# Optional models:
ollama pull nemotron-cascade-2:30b-a3b-q4_K_M
ollama pull gpt-oss:20b
ollama pull laguna-xs-2.1:q4_K_M
```

## Test Models API

Using the router endpoint (`/v1`):

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/models | jq .
```

## Chat Completion Tests

## Think Control Policy (Router -> External LLM Backend)

The router injects the `think` flag in requests to the external LLM backend (Ollama) based on policy.

Policy defaults:

- Per-model defaults come from the active profile's `models.yaml` (mounted as `model_policy.yml`).
- `qwen3-coder:30b` (Orin) and `qwen3-coder-next:q4_K_M` (Thor) default to `think=false`.
- Most general-purpose models default to `think=true`.
- `think=false` for web/browse/search-style tool flows (for example `web_search`, `browser`, `http_get`, `fetch`, `scrape`).
- `think=true` for non-web tools such as `file_search` and `openweather` unless summarization/size heuristics trigger `think=false`.
- `think=false` when message content is very large (`DISABLE_THINK_CHAR_THRESHOLD`) and for summary-like last-user turns over recent tool-heavy or long context.

Manual override header:

- `X-Ollama-Think: true`
- `X-Ollama-Think: false`

If the override header is present, it takes precedence over policy.

The router forwards the following request fields unchanged to the external LLM backend:

- `options` such as `num_ctx`, `num_predict`, `cache_type_k`, `cache_type_v`, and `num_keep`
- `format` including JSON schema structured-output payloads
- `keep_alive`
- `X-Ollama-Think`

Example A: automatic `think=false` from web-search style tool call.

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6:35b-a3b",
    "messages": [
      {"role": "user", "content": "Use web search and summarize key takeaways."}
    ],
    "tools": [
      {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]
  }' | jq .
```

Example B: same request but force `think=true` with header override.

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Ollama-Think: true' \
  -d '{
    "model": "qwen3.6:35b-a3b",
    "messages": [
      {"role": "user", "content": "Use web search and summarize key takeaways."}
    ],
    "tools": [
      {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]
  }' | jq .
```

### Default/general model (stays warm)

`keep_alive: -1` is set by router policy for this model by default.

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6:35b-a3b",
    "messages": [{"role": "user", "content": "Give me a one-line summary of Jetson Orin."}]
  }' | jq .
```

### Structured-output model (stays warm, default think=false)

`keep_alive: -1` and `think: false` are set by router policy for this model by default.

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-coder:30b",
    "format": {
      "type": "json_schema",
      "json_schema": {
        "name": "cue_selection",
        "schema": {
          "type": "object",
          "properties": {
            "cue_ids": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["cue_ids"]
        }
      }
    },
    "options": {
      "num_ctx": 65536,
      "num_predict": 256,
      "cache_type_k": "q8_0",
      "cache_type_v": "q8_0",
      "num_keep": 128
    },
    "messages": [{"role": "user", "content": "Return cue IDs as JSON only."}]
  }' | jq .
```

### Optional verifier model

`keep_alive: 10m` (Orin) or `20m` (Thor) and `think: true` are set by router policy for this model by default.

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-cascade-2:30b-a3b-q4_K_M",
    "messages": [{"role": "user", "content": "Verify whether the answer is internally consistent."}]
  }' | jq .
```

## External LLM Backend Configuration

If LibreChat can call the external LLM backend (Ollama) directly, set base URL to `http://ORIN_IP:11434/v1` (Orin) or `http://THOR_IP:11434/v1` (Thor).

Per-request keep_alive can be set in payload if your client supports sending it.

## LibreChat Configuration

### Profile-Based Model Mapping

| Custom Mode | Orin Model ID | Thor Model ID |
| --- | --- | --- |
| `Coding` | `qwen3-coder:30b` | `qwen3-coder-next:q4_K_M` |
| `Chat` | `qwen3.6:35b-a3b` | `qwen3.6:35b-a3b-q8_0` |

- Base URL (recommended with policy routing): `http://DEVICE_IP:${ROUTER_PORT:-4000}/v1`
- Set LibreChat default model to `qwen3.6:35b-a3b` (Orin) or `qwen3.6:35b-a3b-q8_0` (Thor).

## Jetson Runtime Notes

- Jetson AI Lab recommends vLLM + AWQ/NVFP4 for top Qwen3.6 throughput on Orin/Thor.
- This stack connects to an external LLM backend (Ollama) for model residency control via `OLLAMA_BASE_URL`.
- `qwen2.5-coder` is supported by Ollama even if not currently listed in Jetson AI Lab model cards.
- If you switch to NVIDIA's vLLM command for Qwen3.6, use a separate endpoint (typically `:8000`) and keep this router stack for policy routing behavior.
- **CUDA Architecture**: Orin uses SM 87 (Ampere); Thor uses SM 110 (Blackwell). Ensure the correct `HARDWARE_PROFILE` is set to select the appropriate build artifacts.

## Memory Behavior Summary

The router connects to an external LLM backend (Ollama) for model residency control. Configure `OLLAMA_BASE_URL` in [`profiles/router.env`](profiles/router.env) to point to your Ollama instance.

Router policy defaults by profile:

### Orin Profile

| Model | keep_alive | think |
| --- | --- | --- |
| `qwen3-coder:30b` | `-1` (stay warm) | `false` |
| `qwen3.6:35b-a3b` | `10m` | `true` |
| `nemotron-cascade-2:30b-a3b-q4_K_M` | `10m` | `true` |
| `qwen3:4b` | `10m` | `true` |

### Thor Profile

| Model | keep_alive | think |
| --- | --- | --- |
| `qwen3-coder-next:q4_K_M` | `45m` | `false` |
| `qwen3.6:35b-a3b-q8_0` | `-1` (stay warm) | `true` |
| `gemma4:31b-it-q4_K_M` | `20m` | `true` |
| `granite4.1-guardian:8b-q6_K` | `20m` | `false` |

## Headroom-AI Context Management

The router uses the public Python library API from [headroom-ai](https://github.com/headroomlabs-ai/headroom) as an inline compression layer before forwarding a finalized request to Ollama. The image installs the latest `headroom-ai[code,relevance]` feature set: core content routing, AST-aware code compression, and semantic relevance scoring.

### How It Works

When enabled, the router injects optional retrieval context, constructs the complete Ollama payload, and calls `headroom.compress(messages, model=..., headroom_query=...)` once on the finalized message list. The router extracts the user's query from the last user message and passes it as `headroom_query` for semantic relevance scoring when enabled.

Headroom selects supported transforms internally based on content type (JSON, code, text, etc.). The router does not split history, drop turns, implement its own cache alignment, or maintain a separate compression store.

After compression, the router checks the compressed prompt against the configured model context window while reserving output and safety tokens. A compressed prompt that still cannot fit is rejected rather than silently discarding conversation history.

### API Contract

The router calls `headroom.compress()` with the following parameters:

- `messages`: The finalized message list (required)
- `model`: The model name for token counting (required)
- `headroom_query`: The user's query for relevance scoring (optional, extracted from last user message)

**Note:** The router does not pass invalid parameters such as `compress_user_messages`, `target_ratio`, `protect_recent`, or `relevance_threshold`. These are not part of the headroom.compress() API contract.

### Policy Configuration

Each model in the active profile's [`models.yaml`](profiles/orin/models.yaml) or [`models.yaml`](profiles/thor/models.yaml) defines headroom parameters:

| Parameter | Description | Example |
| --- | --- | --- |
| `reserved_output_tokens` | Tokens reserved for generation output | `4096` |
| `safety_headroom_tokens` | Extra buffer for unexpected tokens | `4096` |

The router does not expose Headroom MCP tools, proxy routes, custom CCR retrieval, or an internal retrieval tool. Those capabilities require a separately designed and approved integration.

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `HEADROOM_ENABLED` | `1` | Set to `0`, `false`, `off`, or `no` to disable headroom compression entirely |
| `HEADROOM_RELEVANCE_ENABLED` | `1` | Set to `0`, `false`, `off`, or `no` to disable semantic relevance scoring |

### Disabling Headroom

To bypass headroom-ai compression (e.g., for debugging or when using models with very large context):

```bash
HEADROOM_ENABLED=0 docker compose up -d
```

When disabled explicitly, requests proceed without Headroom compression. When enabled but the compression API is unavailable, the router returns an error instead of silently forwarding uncompressed input.

### Disabling Relevance Scoring

To disable semantic relevance scoring while keeping compression enabled:

```bash
HEADROOM_RELEVANCE_ENABLED=0 docker compose up -d
```

When relevance scoring is disabled, the router does not extract or pass `headroom_query` to the compression function. This may reduce compression quality for tool-heavy conversations but eliminates the overhead of semantic similarity calculations.

## GPU Acceleration Support

Qdrant supports GPU-accelerated vector indexing for NVIDIA and AMD GPUs. Select the appropriate deployment variant based on your hardware.

### Hardware Requirements

| GPU Type | Host Dependencies | Docker Image |
| --- | --- | --- |
| NVIDIA | `nvidia-container-toolkit` installed on host | `qdrant/qdrant:gpu-nvidia-latest` |
| AMD | Vulkan runtime (`vulkan-tools`, `vulkan-radeon`) installed on host | `qdrant/qdrant:gpu-amd-latest` |

### Environment Variables for GPU Support

| Variable | Default | Description |
| --- | --- | --- |
| `QDRANT__GPU__INDEXING` | `false` | Set to `true` to enable GPU-accelerated vector indexing |
| `QDRANT__OPTIMIZERS__MAX_MEMORY_BYTES` | `4294967296` (4GB) | Maximum memory for vector operations |
| `RUST_LOG` | `info` | Logging level for Qdrant container |

### GPU Verification

After starting a GPU-enabled deployment, verify GPU indexing is active:

```bash
# Check container logs for GPU initialization
docker logs qdrant | grep -i "gpu\|indexing"

# Verify health check endpoint
curl -sf http://127.0.0.1:6333/healthcheck
```

### Fallback Behavior

If GPU acceleration fails or is unavailable, Qdrant automatically falls back to CPU mode. Monitor logs for warnings about GPU initialization failures.

## Testing

The project includes a comprehensive pytest test suite with **207 tests** covering all core modules including headroom-ai compression, tokenization, think policy evaluation, Qdrant retrieval, embeddings, streaming, and usage tracking.

> **Important:** Tests must run inside the Docker container. Running tests on the host will fail because `headroom-ai` and its ML dependencies (PyTorch, transformers, etc.) are only available in the container environment. See [TESTING.md](TESTING.md) for details.

### Quick Start — Docker-Based Testing

Build the audit target (runs all tests during build):

```bash
docker build --target audit -t llmrouter-audit .
```

After building, run specific test markers or files:

```bash
# Run only headroom base tests
docker run --rm llmrouter-audit python3 -m pytest tests/ -m base -v

# Run specific test file
docker run --rm llmrouter-audit python3 -m pytest tests/test_tokenizer.py -v
```

### Test Markers

| Marker | Description | Count |
|--------|-------------|-------|
| `base` | Base compression | 31 |
| `code` | Code compression | 12 |
| `relevance` | Relevance scoring | 11 |
| `tokenizer` | Token counting | 19 |
| `policy` | Think policy | 35 |
| `retrieval` | Qdrant retrieval | 15 |
| `embeddings` | Embedding endpoints | 14 |
| `streaming` | SSE streaming | 23 |
| `usage` | Usage tracking | 17 |
| `integration` | Real headroom tests | 11 |

### Full Documentation

For complete testing documentation including marker reference, test structure, adding new tests, and CI configuration, see [TESTING.md](TESTING.md).

### Verify Services Running

Verify the router is running:

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/models | jq .
```

Verify Qdrant is running (if enabled):

```bash
curl -sf http://127.0.0.1:6333/healthcheck
```

## Validation Scripts

Run the included validation scripts to verify profile configuration:

```bash
# Validate shared stack configuration
scripts/validation/validate-shared-stack.sh orin
scripts/validation/validate-shared-stack.sh thor

# Validate model tags are available
PROFILE=orin scripts/validation/validate-model-tags.sh orin
PROFILE=thor scripts/validation/validate-model-tags.sh thor

# Validate runtime gates
PROFILE=orin scripts/validation/validate-runtime-gates.sh orin
PROFILE=thor scripts/validation/validate-runtime-gates.sh thor

# Validate Thor ASR configuration
./scripts/validation/validate-thor-asr.sh
```

## Additional Documentation

- [Testing Strategy](docs/testing-strategy.md) - Comprehensive guide to unit tests, validation scripts, and the recommended test workflow.
- [Unified Orin + Thor Profile Runbook](docs/UnifiedOrinThorProfiles.md) - Detailed profile management, start commands, and validation procedures.
