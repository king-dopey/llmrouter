# OpenAI-Compatible Router Node for Ollama LLM

This repository runs an OpenAI-compatible router with Qdrant vector database for LAN access by LibreChat (hosted elsewhere). The router connects to **any Ollama instance** and applies per-model policy routing, context management, and think-control for all models served by that backend.

This node is single-tenant by design: do not run any other GPU workload or heavyweight CPU workload here. Extra containers, dev tools, or interactive sessions that touch the GPU break the memory budget.

## Works with Any Ollama-Served Model

This router is **hardware-agnostic** and works with any model available through your Ollama backend. The `OLLAMA_HOST` environment variable points to your Ollama instance, and the router applies policy rules to all models listed in the active profile's configuration.

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
| `thor` | Jetson Thor | 64+ GB unified | SM 90 (Blackwell) | `ubuntu:24.04` | Tested and working |

Select the profile at runtime via the `PROFILE` environment variable (default: `orin`). To use a custom configuration, set `PROFILE` to a directory containing your `models.yaml` and `stack.env` files, or edit the default profile directly.

### Profile File Layout

```
profiles/
├── orin/
│   ├── models.yaml          # Orin model definitions and policy
│   └── stack.env            # Orin build/runtime defaults (CUDA arch, etc.)
└── thor/
    ├── models.yaml          # Thor model definitions and policy
    └── stack.env            # Thor build/runtime defaults (CUDA arch, etc.)
```

The active profile's `models.yaml` is mounted into the router container at `/app/model_policy.yml`.

## Files

- [`docker-compose.yml`](docker-compose.yml): OpenAI-compatible router and optional Qdrant vector database.
- [`model_policy.yml`](model_policy.yml): Default policy file; the runtime profile overrides this via mount.
- [`.env.example`](.env.example): environment values for ports, profile selection, and model behavior.

## Network and Security Notes

- Exposed ports intentionally bind on all interfaces for LAN use:
  - `0.0.0.0:4000` (OpenAI-compatible router)
  - `0.0.0.0:6333` (Qdrant vector database, optional)
- Restrict access at host firewall/router ACLs to trusted LAN clients.

## Start Services

```bash
cp .env.example .env
```

Edit `.env` and set the desired profile:

```bash
PROFILE=orin   # or PROFILE=thor
```

Start the router stack:

```bash
docker compose up -d
```

Optional proxy build (profile-aware):

```bash
PROFILE=orin docker compose -f router/docker-compose.yml --profile proxy up -d --build
# or
PROFILE=thor docker compose -f router/docker-compose.yml --profile proxy up -d --build
```

Optional Qdrant vector database:

```bash
docker compose --profile qdrant up -d
```

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

### Thor Profile (SM 90 / Blackwell)

The Thor profile targets larger context windows and a broader model catalog, leveraging the newer Blackwell architecture.

| Model ID | Purpose | Default keep_alive | Default think | Supported `num_ctx` Ceiling | Notes |
| --- | --- | --- | --- | --- | --- |
| `qwen3-coder-next:q4_K_M` | Next-gen coding model for structured-output workloads. | `-1` | `false` | `262144` | Primary coding model; stays warm with large context. |
| `qwen3.6:35b-a3b-q8_0` | Default chat and narrative summarization workloads. | `-1` | `true` | `262144` | Stays warm; higher precision KV cache. |
| `gemma4:31b-it-q4_K_M` | Large general-purpose model. | `20m` | `true` | `131072` | Heavyweight but capable for complex tasks. |
| `devstral-small-2:24b-instruct-2512-q8_0` | French-language and multilingual instruction model. | `20m` | `true` | `131072` | Specialized language support. |
| `north-mini-code-1.0:q8_0` | Compact coding assistant. | `20m` | `true` | `131072` | Fast code generation. |
| `granite4.1-guardian:8b-q6_K` | Guardrail / safety classification model. | `20m` | `false` | `65536` | Non-thinking safety filter. |
| `qwen3:4b` | Fast assistant / routing model. | `30m` | `true` | `65536` | Lightweight fallback. |
| `qwen3-vl:4b` | Vision-language tasks (image understanding). | `30m` | `true` | `65536` | Multimodal input support. |
| `gemma4:12b` | Mid-tier general-purpose model. | `30m` | `true` | `65536` | Creative tasks alternative. |
| `reader-lm:1.5b` | Document reading / extraction specialist. | `30m` | `false` | `65536` | Non-thinking document processing. |
| `qwen3-embedding:4b` | Embedding service for Qdrant retrieval. | `30m` | `false` | `8192` | Used by retrieval pipeline. |
| `nemotron-cascade-2:30b-a3b-q4_K_M` | Optional reasoning verifier. | `20m` | `true` | `131072` | Reasoning verification. |
| `gpt-oss:20b` | Secondary general-purpose model. | `20m` | `true` | `131072` | Diverse task fallback. |
| `laguna-xs-2.1:q4_K_M` | Lightweight reasoning / utility model. | `20m` | `true` | `131072` | Small-footprint helper. |

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

Configure your external LLM backend (Ollama) by setting `OLLAMA_HOST` in `.env` to point to your Ollama instance. This can be a local or remote Ollama server — the router works with any Ollama endpoint.

```bash
OLLAMA_HOST=http://YOUR_OLLAMA_HOST:11434
```

Pull models on your Ollama host. The lists below show the example models configured in each tested profile, but you can pull **any model** supported by Ollama and use it with this router.

### Example Models — Orin Profile

These are the models pre-configured in the Orin profile:

### Example Models — Orin Profile

These are the models pre-configured in the Orin profile:

```bash
ollama pull qwen3-coder:30b
ollama pull qwen3.6:35b-a3b
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
curl -sS http://127.0.0.1:4000/v1/models | jq .
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
curl -sS http://127.0.0.1:4000/v1/chat/completions \
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
curl -sS http://127.0.0.1:4000/v1/chat/completions \
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
curl -sS http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6:35b-a3b",
    "messages": [{"role": "user", "content": "Give me a one-line summary of Jetson Orin."}]
  }' | jq .
```

### Structured-output model (stays warm, default think=false)

`keep_alive: -1` and `think: false` are set by router policy for this model by default.

```bash
curl -sS http://127.0.0.1:4000/v1/chat/completions \
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
curl -sS http://127.0.0.1:4000/v1/chat/completions \
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

- Base URL (recommended with policy routing): `http://DEVICE_IP:4000/v1`
- Set LibreChat default model to `qwen3.6:35b-a3b` (Orin) or `qwen3.6:35b-a3b-q8_0` (Thor).

## Jetson Runtime Notes

- Jetson AI Lab recommends vLLM + AWQ/NVFP4 for top Qwen3.6 throughput on Orin/Thor.
- This stack connects to an external LLM backend (Ollama) for model residency control via `OLLAMA_HOST`.
- `qwen2.5-coder` is supported by Ollama even if not currently listed in Jetson AI Lab model cards.
- If you switch to NVIDIA's vLLM command for Qwen3.6, use a separate endpoint (typically `:8000`) and keep this router stack for policy routing behavior.
- **CUDA Architecture**: Orin uses SM 87 (Ampere); Thor uses SM 90 (Blackwell). Ensure the correct `PROFILE` is set to select the appropriate build artifacts.

## Memory Behavior Summary

The router connects to an external LLM backend (Ollama) for model residency control. Configure `OLLAMA_HOST` in `.env` to point to your Ollama instance.

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
| `qwen3-coder-next:q4_K_M` | `-1` (stay warm) | `false` |
| `qwen3.6:35b-a3b-q8_0` | `-1` (stay warm) | `true` |
| `gemma4:31b-it-q4_K_M` | `20m` | `true` |
| `granite4.1-guardian:8b-q6_K` | `20m` | `false` |

## Headroom-AI Context Management

The router uses [headroom-ai](https://github.com/lastmile-ai/headroom) to manage context window budgets and prevent requests from exceeding model context limits. This is critical for the Orin's constrained 64 GB unified memory environment where models are configured with specific `num_ctx` ceilings.

### How It Works

When a chat completion request arrives, the router performs a **preflight headroom check** before sending anything to Ollama:

1. [`check_and_trim()`](router_headroom.py:139) calculates the usable prompt budget:
   ```
   usable_prompt_budget = num_ctx - reserved_output_tokens - safety_headroom_tokens
   ```
2. The current message sequence is tokenized using [`tokenizer.count_prompt_tokens()`](tokenizer.py).
3. If the token count fits within the budget, the request proceeds unchanged.
4. If it exceeds the budget, headroom-ai's [`compress()`](router_headroom.py:109) function is invoked to intelligently compress/summarize older messages.
5. After compression, if the request still exceeds the budget, it is **rejected** with HTTP 413 and detailed token accounting.

### Policy Configuration

Each model in the active profile's [`models.yaml`](profiles/orin/models.yaml) or [`models.yaml`](profiles/thor/models.yaml) defines headroom parameters:

| Parameter | Description | Example |
| --- | --- | --- |
| `reserved_output_tokens` | Tokens reserved for generation output | `4096` |
| `safety_headroom_tokens` | Extra buffer for unexpected tokens | `4096` |
| `trim_strategy` | How to compress when over budget | `drop_oldest_then_summarize` |

Available trim strategies:

| Strategy | Behavior |
| --- | --- |
| `drop_oldest` | Remove oldest messages until within budget |
| `summarize_history` | Use LLM to summarize older conversation turns |
| `drop_oldest_then_summarize` | Drop very old messages first, then summarize remaining |

### Headroom Tool Integration

The router supports an internal `headroom_retrieve` tool that allows models to fetch previously compressed content from headroom-ai's compression store (CCR):

- When a model emits a `headroom_retrieve` tool call, the router calls [`retrieve_from_ccr()`](router_headroom.py:121) to look up the original uncompressed content.
- The [`_continue_after_headroom_retrieve()`](app.py:580) function transparently resolves these calls and continues generation.
- This enables multi-hop retrieval with a configurable limit (`HEADROOM_MAX_RETRIEVE_HOPS`, default 3).

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `HEADROOM_ENABLED` | `1` | Set to `0`, `false`, `off`, or `no` to disable headroom compression entirely |
| `HEADROOM_MAX_RETRIEVE_HOPS` | `3` | Maximum continuation hops for `headroom_retrieve` tool calls |

### Disabling Headroom

To bypass headroom-ai compression (e.g., for debugging or when using models with very large context):

```bash
HEADROOM_ENABLED=0 docker compose up -d
```

When disabled, requests exceeding the budget are rejected immediately without compression attempts.

## Testing

Verify the router is running:

```bash
curl -sS http://127.0.0.1:4000/v1/models | jq .
```

Verify Qdrant is running (if enabled):

```bash
curl -sS http://127.0.0.1:6333/readyz | jq .
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
