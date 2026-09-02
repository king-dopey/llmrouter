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

```text
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

## File Layout

All router Python modules live at the **project root** (not in a `router/` subdirectory).
The Dockerfile copies these into `/app/` inside the container.

| Module | Purpose |
| --- | --- |
| [`app.py`](app.py) | FastAPI router application, chat completions, embeddings |
| [`policy.py`](policy.py) | Think policy configuration and evaluation |
| [`tokenizer.py`](tokenizer.py) | Token counting (tiktoken + approximate fallback) |
| [`router_headroom.py`](router_headroom.py) | Headroom-ai compression integration and budget enforcement |
| [`retrieval.py`](retrieval.py) | Qdrant hybrid retrieval (dense + sparse + RRF) |
| [`ingest_repo.py`](ingest_repo.py) | Repository ingestion into Qdrant collection |

Other key files:

- [`docker-compose.yml`](docker-compose.yml): OpenAI-compatible router and optional Qdrant vector database.
- [`model_policy.yml`](model_policy.yml): Default policy file; the runtime profile overrides this via mount.
- [`.env.example`](.env.example): environment values for ports, profile selection, and model behavior.

> **Note on documentation paths**: Some plan documents reference `router/app.py`, `router/tokenizer.py`, etc.
> These are historical references from an earlier layout. The actual files are at the workspace root as shown above.

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

| Model ID | Purpose | Default keep_alive | Default think | tool_call_think | retry_on_failure | Supported `num_ctx` Ceiling | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6:35b-a3b` | Default chat and narrative summarization workloads. | `-1` (stay warm) | `true` | `false` | `true` | `32768` | Primary default model; hybrid attention keeps KV usage comparatively small. Safety invariant prevents think+tools loops. |
| `qwen3.6:35b-a3b-coding` | A/B candidate for Python/MCP implementation work. | `15m` | `true` | `false` | `true` | `65536` | Coding-specialized variant for comparison with qwen3.6:35b-a3b. MoE model ~18-24 GB. |
| `qwen3-embedding:4b` | Embedding service for Qdrant retrieval. | `-1` (stay warm) | `false` | - | - | `8192` | Always resident for continuous retrieval. |
| `qwen3-coder:30b` | Strict-JSON and structured-output workloads for boundary selection and cue-ID extraction. | `10m` | `false` | `false` | `true` | `65536` | Coding model; no longer pinned, stays warm during use. Safety invariant enforced. |
| `qwen3:4b` | Fast assistant / routing model. | `10m` | `true` | `false` | `true` | `65536` | Lightweight fallback for simple queries. Safety invariant enforced. |
| `qwen3-vl:4b` | Vision-language tasks (image understanding). | `10m` | `true` | `false` | `true` | `65536` | Multimodal input support. Safety invariant enforced. |
| `qwen3.8:27b-q8_0` | Marginal model for short-context tasks. | `10m` | `true` | `false` | `true` | `32768` | WARNING: ~27 GB at Q8, leaves minimal headroom. Short-context only (<16K tokens). Prefer qwen3.6:35b-a3b for production. |
| `gemma4:12b` | Mid-tier general-purpose model. | `10m` | `true` | - | - | `65536` | Alternative for creative tasks. No safety invariant (non-Qwen). |
| `nemotron-cascade-2:30b-a3b-q4_K_M` | Optional reasoning verifier for ambiguous structured answers. | `10m` | `true` | - | - | `65536` | Only resident while actively in use; expect one warm-model eviction when it loads. |
| `gpt-oss:20b` | Secondary general-purpose model. | `10m` | `true` | - | - | `65536` | Fallback for diverse tasks. |
| `laguna-xs-2.1:q4_K_M` | Lightweight reasoning / utility model. | `10m` | `true` | - | - | `65536` | Small-footprint helper. |

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

| Model ID | Purpose | Default keep_alive | Default think | tool_call_think | retry_on_failure | Supported `num_ctx` Ceiling | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen3.6:35b-a3b-q8_0` | Default chat and primary coding/planning model. | `-1` (stay warm) | `true` | `false` | `true` | `262144` | Primary default; stays warm for all standard workloads. Safety invariant prevents think+tools loops. |
| `qwen3.6:35b-a3b-coding` | A/B candidate for Python/MCP implementation work. | `15m` | `true` | `false` | `true` | `262144` | Coding-specialized variant for comparison with qwen3.6:35b-a3b-q8_0. |
| `qwen3-embedding:8b` | Code search / indexing embedding model. | `-1` (stay warm) | `false` | - | - | `40960` | Always hot; native 4096-dim embeddings for code indexing. |
| `qwen3.8:27b-q8_0` | Escalation model for complex coding/planning/reasoning. | `45m` | `true` | `false` | `true` | `262144` | Hard task upgrade over qwen3.6; router-side medium reasoning recommended. Safety invariant enforced. |
| `qwen3.8:27b-bf16` | Explicit quality-first model for important complex tasks. | `15m` | `true` | `false` | `true` | `262144` | Manual escalation only; ~8-15x task-time vs qwen3.6 baseline. Safety invariant enforced. |
| `qwen3-coder-next:q4_K_M` | Direct-edit / no-think coding fallback for straightforward tasks. | `30m` | `false` | `false` | `true` | `131072` | Parser-stable output; deprecated as primary coder. Safety invariant enforced. |
| `laguna-xs-2.1:q4_K_M` | Experimental coding alternative (high loop risk). | `15m` | `true` | - | - | `131072` | A/B testing only; demoted due to loop risk in this environment. |
| `gemma4:31b-it-q4_K_M` | Large non-Qwen general-purpose model. | `20m` | `true` | - | - | `131072` | Heavyweight but capable for complex tasks. No safety invariant (non-Qwen). |
| `devstral-small-2:24b-instruct-2512-q8_0` | French-language and multilingual instruction model. | `15m` | `true` | - | - | `131072` | Specialized language support. |
| `north-mini-code-1.0:q8_0` | Compact coding assistant. | `15m` | `true` | - | - | `131072` | Fast code generation. |
| `granite4.1-guardian:8b-q6_K` | Security review / policy gate model. | `20m` | `false` | - | - | `65536` | Primary model for security-review stage. |
| `qwen3:4b` | Fast assistant / routing model. | `30m` | `true` | `false` | `true` | `65536` | Lightweight fallback. Safety invariant enforced. |
| `qwen3.5:4b` | Small replacement path for qwen3:4b and qwen3-vl:4b. | `30m` | `true` | `false` | `true` | `131072` | Successor to qwen3:4b. Safety invariant enforced. |
| `qwen3-vl:4b` | Vision-language tasks (image understanding). | `30m` | `true` | `false` | `true` | `131072` | Multimodal input support; kept until callers migrate. Safety invariant enforced. |
| `gemma4:12b` | Medium-size general alternative. | `20m` | `true` | - | - | `131072` | Creative tasks alternative. No safety invariant (non-Qwen). |
| `reader-lm:1.5b` | Document reading / extraction specialist. | `20m` | `false` | - | - | `65536` | Non-thinking document processing. |
| `qwen3-embedding:4b` | Transition embedding model (deprecated in favor of 8b). | `20m` | `false` | - | - | `8192` | Keep until all clients move to 8b. |
| `nemotron-cascade-2:30b-a3b-q4_K_M` | Optional reasoning verifier. | `15m` | `true` | - | - | `131072` | Reasoning verification. |
| `gpt-oss:20b` | Secondary general-purpose model. | `15m` | `true` | - | - | `131072` | Diverse task fallback. |

### Hardware Compatibility Notes

**Orin 64GB:**

- ✅ qwen3.6:35b-a3b (MoE, ~18-24 GB) - Primary default
- ✅ qwen3.6:35b-a3b-coding (MoE, ~18-24 GB) - A/B candidate
- ✅ qwen3-coder:30b (~18-19 GB) - Coding worker
- ⚠️ qwen3.8:27b-q8_0 (~27 GB) - Marginal, short-context only
- ❌ qwen3-coder-next:q4_K_M (~52 GB) - Excluded (too large)
- ❌ qwen3.8:27b-bf16 (~54 GB) - Excluded (too large)

**Thor 128GB:**

- ✅ All Orin models
- ✅ qwen3.8:27b-q8_0 (~27 GB) - Comfortable fit
- ✅ qwen3.8:27b-bf16 (~54 GB) - High-quality reasoning
- ✅ qwen3-coder-next:q4_K_M (~52 GB) - Direct-edit fallback
- ✅ qwen3.6:35b-a3b-coding - A/B candidate for Python/MCP work

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
ollama pull qwen3.6:35b-a3b
ollama pull qwen3-embedding:4b
ollama pull qwen3-coder:30b
ollama pull qwen3:4b
ollama pull qwen3-vl:4b
ollama pull gemma4:12b
# Optional models:
ollama pull nemotron-cascade-2:30b-a3b-q4_K_M
ollama pull gpt-oss:20b
ollama pull laguna-xs-2.1:q4_K_M
```

### Example Models — Thor Profile

These are the models pre-configured in the Thor profile:

```bash
ollama pull qwen3.6:35b-a3b-q8_0
ollama pull qwen3-embedding:8b
ollama pull qwen3.8:27b-q8_0
ollama pull qwen3.8:27b-bf16
ollama pull qwen3-coder-next:q4_K_M
ollama pull laguna-xs-2.1:q4_K_M
ollama pull gemma4:31b-it-q4_K_M
ollama pull devstral-small-2:24b-instruct-2512-q8_0
ollama pull north-mini-code-1.0:q8_0
ollama pull granite4.1-guardian:8b-q6_K
ollama pull qwen3:4b
ollama pull qwen3.5:4b
ollama pull qwen3-vl:4b
ollama pull gemma4:12b
ollama pull reader-lm:1.5b
ollama pull qwen3-embedding:4b
# Optional models:
ollama pull nemotron-cascade-2:30b-a3b-q4_K_M
ollama pull gpt-oss:20b
```

## Test Models API

Using the router endpoint (`/v1`):

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/models | jq .
```

## Chat Completion Tests

## Think Control Policy (Router -> External LLM Backend)

The router implements a ReAct-aligned think control policy that determines when models should use extended thinking based on request context, model capabilities, and safety requirements. The policy uses a three-tier rule hierarchy evaluated top-to-bottom (first match wins).

### Three-Tier Rule Hierarchy

**Rule 1: Safety Invariant (Absolute Priority)**
- Condition: Tools present AND `tool_call_think=False`
- Action: Return `think=False` (unconditionally)
- Rationale: Prevents Qwen3 models from entering deterministic looping behavior

**Rule 2: Tool Request Handling (Fail-Safe)**
- Condition: Tools present AND `tool_call_think` is not `False`
- Action: Evaluate override and policy
  - Valid override (`tool_call_true`) → `think=True`
  - Invalid override → `think=False` (deny by default)
  - Explicit `tool_call_think` policy → use it
  - No explicit policy → `think=False` (fail-safe)

**Rule 3: Non-Tool Request Handling (ReAct-Aligned)**
- Condition: No tools present
- Action: Evaluate override and character threshold
  - Override present → use it
  - Combined message characters ≥ `char_threshold` → `think=False`
  - Otherwise → return `default_think` (from model config)

### Recent Changes (ReAct-Aligned Policy Redesign)

The policy has been redesigned to align with ReAct (Reasoning and Acting) principles:

**Removed Legacy Heuristics:**
- `DISABLE_THINK_TOOL_PATTERNS` - Previously disabled thinking based on tool name patterns
- `DISABLE_THINK_SUMMARY_PATTERNS` - Previously disabled thinking based on summary keywords
- Pattern matching functions (`_collect_tool_names()`, `_last_user_text()`, etc.)

**Rationale:** The legacy heuristics incorrectly disabled thinking for summary and analysis tasks when conversation history contained tool references. The new policy evaluates each turn based on its own task requirements, using `char_threshold` as the sole determinant for non-tool requests.

**Simplified Configuration:**
`ThinkPolicyConfig` now contains only `char_threshold` (default: 12000). The legacy fields have been removed.

### Safety Invariant for Tool-Call Thinking

**Critical:** The router enforces a configuration-driven safety invariant to prevent Qwen3 models from entering infinite loops when tools are present.

Any model configured with `tool_call_think: false` in `models.yaml` **cannot** be overridden to `think: true` when tools are present in the request. This safety invariant executes before any override logic, ensuring it cannot be bypassed.

**Example Configuration:**

```yaml
# profiles/orin/models.yaml or profiles/thor/models.yaml
- model: qwen3.6:35b-a3b
  think: true                    # Default behavior when no tools present
  tool_call_think: false         # Safety invariant: cannot be overridden when tools present
  keep_alive: 15m
```

**Behavior:**

- Request with tools + `tool_call_think: false` → `think: false` (safety invariant enforced)
- Request with tools + `tool_call_think: false` + `X-Ollama-Think: tool_call_true` → `think: false` (safety invariant cannot be bypassed)
- Request without tools + `tool_call_think: false` → `think: true` (safety invariant only applies when tools present)

For complete technical documentation, see [docs/think-policy.md](docs/think-policy.md).

### Tool-Call Retry Guard

The router includes a one-shot retry guard to prevent infinite agent loops when models mention tools in text but fail to produce structured tool_calls.

**How It Works:**
1. When a model returns assistant text mentioning a tool but no structured `tool_calls`
2. The router detects this failure condition
3. Automatically retries once with `think: false` and low temperature (0.1)
4. Returns the retry result or a structured error if retry fails

**Configuration:**

```yaml
# profiles/orin/models.yaml or profiles/thor/models.yaml
- model: qwen3.6:35b-a3b
  retry_on_tool_call_failure: true   # Enable retry guard (default: false)
  retry_temperature: 0.1              # Temperature for retry (default: 0.1)
```

**Behavior:**

- Retry is **opt-in per model** via `retry_on_tool_call_failure` (defaults to `false` when not set)
- Forces `think: false` to prevent reasoning loops
- Uses low temperature (configurable) for deterministic output
- Removes invalid assistant response before retry
- Returns HTTP 502 with `retry_attempted: true` if retry fails
- Works for both streaming and non-streaming requests

**Example Error Response:**

```json
{
  "error": "tool_call_retry_failed",
  "message": "Model failed to produce structured tool_calls after retry",
  "retry_attempted": true,
  "retry_error": "Retry did not produce tool_calls"
}
```

All Qwen models in Orin and Thor profiles have retry guard enabled by default.

### Policy Defaults

The router implements a ReAct-aligned think policy that evaluates each turn based on its own task requirements, not conversation history:

- Per-model defaults come from the active profile's `models.yaml` (mounted as `model_policy.yml`).
- `qwen3-coder:30b` (Orin) and `qwen3-coder-next:q4_K_M` (Thor) default to `think=false`.
- Most general-purpose models default to `think=true`.
- **Rule 1**: Models with `tool_call_think: false` and tools present → `think=false` (safety invariant, no overrides)
- **Rule 2**: Models with tools present → explicit `tool_call_think` policy or fail-safe to `false`
- **Rule 3**: No tools present → apply character threshold heuristic, then default
  - `think=false` when message content exceeds `DISABLE_THINK_CHAR_THRESHOLD` (12000 chars)
  - Otherwise, use model's default `think` value from `models.yaml`

**Note**: The legacy `DISABLE_THINK_TOOL_PATTERNS` and `DISABLE_THINK_SUMMARY_PATTERNS` environment variables have been removed. The policy no longer disables thinking based on tool name patterns or summary keywords in conversation history. Each turn is evaluated independently per ReAct principles.

### Override Headers

The router accepts three override values via the `X-Ollama-Think` header:

| Header Value | Behavior | Use Case |
| --- | --- | --- |
| `true` | Force `think: true` | Enable thinking for non-tool requests |
| `false` | Force `think: false` | Disable thinking for any request |
| `tool_call_true` | Force `think: true` for tool requests | Enable thinking for non-safety-constrained models with tools |

**Override Validation:**

- Invalid override values are rejected with HTTP 400
- When tools are present, only `tool_call_true` is accepted as a valid override
- Any other override value with tools present defaults to `think: false` (fail-safe)
- Safety invariant takes precedence over all override attempts for models with `tool_call_think: false`

For complete override handling documentation, see [docs/think-policy.md](docs/think-policy.md).

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

Example B: same request but force `think=true` with header override (for non-safety-constrained model).

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Ollama-Think: tool_call_true' \
  -d '{
    "model": "gemma4:12b",
    "messages": [
      {"role": "user", "content": "Use web search and summarize key takeaways."}
    ],
    "tools": [
      {"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
    ]
  }' | jq .
```

**Note:** The `tool_call_true` override is only valid for models without `tool_call_think: false` configured. For Qwen models with the safety invariant enabled, this override will be ignored and `think: false` will be enforced.

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
| `qwen3.6:35b-a3b` | `-1` (stay warm) | `true` |
| `qwen3-embedding:4b` | `-1` (stay warm) | `false` |
| `qwen3-coder:30b` | `10m` | `false` |
| `nemotron-cascade-2:30b-a3b-q4_K_M` | `10m` | `true` |
| `qwen3:4b` | `10m` | `true` |

### Thor Profile

| Model | keep_alive | think |
| --- | --- | --- |
| `qwen3.6:35b-a3b-q8_0` | `-1` (stay warm) | `true` |
| `qwen3-embedding:8b` | `-1` (stay warm) | `false` |
| `qwen3.8:27b-q8_0` | `45m` | `true` |
| `qwen3.8:27b-bf16` | `15m` | `true` |
| `qwen3-coder-next:q4_K_M` | `30m` | `false` |
| `granite4.1-guardian:8b-q6_K` | `20m` | `false` |

## Headroom-AI Context Management

The router uses the public Python library API from [headroom-ai](https://github.com/headroomlabs-ai/headroom) as an inline compression layer before forwarding a finalized request to Ollama. The image installs `headroom-ai[code]` for core content routing and AST-aware code compression.

### How It Works

Compaction (headroom.compress) runs on **every request unconditionally** — this is HEADROOM's core purpose. When enabled, the router injects optional retrieval context, constructs the complete Ollama payload, and calls `headroom.compress(messages, model=..., config=_HEADROOM_CONFIG)` once on the finalized message list. The `_HEADROOM_CONFIG` singleton is configured with `compress_system_messages=False` to preserve Zoo Code's tool definitions, usage instructions, and protocol from being compressed.

Headroom selects supported transforms internally based on content type (JSON, code, text, etc.). The router does not split history, drop turns, implement its own cache alignment, or maintain a separate compression store.

After compression, the router validates that tool_calls structure survived intact via `_validate_tool_integrity()`. If validation fails, the router falls back to original uncompressed messages with a warning log.

### Proactive Summarization and Recovery Pipeline

The router has two distinct phases:

1. Proactive summarization/compression runs before the upstream call when a model's configured threshold is reached.
2. Reactive overflow recovery runs only after Ollama rejects the request due to context overflow.

A request is considered proactive when its prompt size reaches the model's configured threshold before generation. That threshold is evaluated using the same budget math as the router's input budget:

`effective_input_budget = model_context_window - reserved_output_tokens - safety_headroom_tokens`

The proactive path is non-dropping: it may apply stronger Headroom-native compression or summarization, but it must not strip messages or tool-call segments.

When upstream Ollama returns an HTTP 400 error indicating context overflow (for example "context length exceeded", "prompt too long", "input length exceeds"), the router uses a bounded recovery ladder:

1. Detect context overflow with `is_context_overflow_error(status_code, response_text)`.
2. Run the v2 ladder via `apply_recovery_ladder(...)` when `HEADROOM_RECOVERY_LADDER_V2=1`.
3. Retry upstream only when the payload changed materially (`changed=True`).
4. Stop after `HEADROOM_RECOVERY_MAX_RETRIES` attempts and propagate the upstream error if still failing.

The v2 ladder is stage-based and fixed: it always tries stronger Headroom compression first, then optional large-text compression, and only then performs oldest-first segment stripping as a last resort. The `recovery_strategy` name is compatibility metadata only; it does not select a different recovery algorithm when v2 is enabled.

### Zoo Context Condensing Interception

Zoo Code context-condensing operations are intercepted by the router so Headroom remains the single context-management authority.

Flow:

1. Zoo sends a minimal condense routing signal (recommended canonical prompt):
  `ZOO_CONTEXT_CONDENSE_V1: delegate context condensing to router headroom pipeline.`
2. The router detects this explicit condense signal only.
3. The router removes trigger-only messages and runs the same Headroom compaction/proactive/recovery pipeline on the remaining conversation context.
4. The router returns a synthetic OpenAI-compatible condensed-context response back to Zoo.

This allows Zoo to run at the model's true configured context window and avoids context waste from an oversized client-side condense template.

#### Segment and Tool Integrity Guarantees

1. No role-only stripping is used in ladder v2.
2. Final segment stripping removes complete chronological segments, not isolated message roles.
3. Segment removal is blocked when it would orphan `tool_call_id` linkage across segment boundaries.
4. The router preserves at least `HEADROOM_RECOVERY_MIN_SEGMENTS_TO_KEEP` newest segments while trimming older history.
5. If no safe material reduction is possible, the router returns the upstream error instead of mutating into inconsistent history.

### API Contract

The router calls `headroom.compress()` with the following parameters:

- `messages`: The finalized message list (required)
- `model`: The model name for token counting (required)
- `config`: Explicit `CompressConfig` singleton with `compress_system_messages=False`, `compress_user_messages=False`, `protect_recent=4`

### Policy Configuration

Each model in the active profile's [`models.yaml`](profiles/orin/models.yaml) or [`models.yaml`](profiles/thor/models.yaml) defines headroom parameters:

| Parameter | Description | Example |
| --- | --- | --- |
| `reserved_output_tokens` | Space reserved for the model's generated response. Higher values leave less room for input and trigger proactive/compression/recovery sooner; lower values increase usable input context but raise output-overflow risk. | `4096` |
| `safety_headroom_tokens` | Extra buffer for tokenizer variance, wrappers, tool/schema overhead, and unexpected token growth. Higher values make the request more conservative and reduce overflow surprises; lower values optimize utilization but raise overflow risk. | `4096` |
| `proactive_summarization_enabled` | Enables the pre-flight summary/compression pass when prompt token count reaches the configured threshold. | `true` |
| `proactive_summarization_threshold_tokens` | Input-context threshold before generation. The router compares the current prompt token count against this value using the same budget math as the active model context window. | `12000` |
| `recovery_strategy` | Legacy compatibility field only. When `HEADROOM_RECOVERY_LADDER_V2=1`, the router ignores the name and always uses the fixed v2 stage order: stronger compression, then text compression, then last-resort stripping. | `drop_oldest` |

The effective input budget is calculated as:

`effective_input_budget = model_context_window - reserved_output_tokens - safety_headroom_tokens`

Example policy entry:

```yaml
- model: qwen3-coder:30b
  keep_alive: 10m
  think: false
  reserved_output_tokens: 2048
  safety_headroom_tokens: 2048
  proactive_summarization_enabled: true
  proactive_summarization_threshold_tokens: 12000
  recovery_strategy: drop_oldest
```

The router does not expose Headroom MCP tools, proxy routes, custom CCR retrieval, or an internal retrieval tool. Those capabilities require a separately designed and approved integration.

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `HEADROOM_ENABLED` | `1` | Set to `0`, `false`, `off`, or `no` to disable headroom compression entirely |
| `HEADROOM_RECOVERY_LADDER_V2` | `1` in profiles/router.env | Enables the modern overflow-recovery ladder instead of the legacy trim-only path |
| `HEADROOM_RECOVERY_ENABLE_STRONGER_COMPRESSION` | `1` | Enables stronger Headroom compression profiles before message removal |
| `HEADROOM_RECOVERY_ENABLE_TEXT_COMPRESSION` | `1` | Enables optional large-text compression via Headroom transforms when available |
| `HEADROOM_RECOVERY_ENABLE_SEGMENT_STRIPPING` | `1` | Enables last-resort linear segment stripping when earlier recovery steps are still over budget |
| `HEADROOM_RECOVERY_MAX_RETRIES` | `1` | Maximum overflow retries after a successful material reduction |
| `HEADROOM_RECOVERY_MIN_TOKEN_DELTA` | `32` | Minimum token reduction required before a recovery step counts as material |
| `HEADROOM_RECOVERY_MIN_STRIP_TOKEN_DELTA` | `1` | Minimum token reduction required from segment stripping before retrying |
| `HEADROOM_RECOVERY_MIN_SEGMENTS_TO_KEEP` | `1` | Minimum newest segments retained while truncating older history |
| `ZOO_CONTEXT_CONDENSE_TRIGGER` | `ZOO_CONTEXT_CONDENSE_V1: delegate context condensing to router headroom pipeline.` | Canonical Zoo condense signal string matched by router interception |

### Disabling Headroom

To bypass headroom-ai compression (e.g., for debugging or when using models with very large context):

```bash
HEADROOM_ENABLED=0 docker compose up -d
```

When disabled explicitly, requests proceed without Headroom compression. When enabled but the compression API is unavailable, the router returns an error instead of silently forwarding uncompressed input.

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

## Embedding Endpoint Reliability

The embedding endpoint (`/v1/embeddings`) implements automatic retry logic with exponential backoff to handle Ollama's model eviction/reload cycle gracefully. When an embedding model is being reloaded into VRAM/RAM, Ollama returns HTTP 503 (Service Unavailable). The router now retries these requests up to 3 times with increasing delays (1s, 2s, 4s) before returning an error to the client.

### Retry Behavior

| Response | Action | Max Retries |
|----------|--------|-------------|
| HTTP 503 (model reloading) | Retry with exponential backoff | 3 attempts |
| `httpx.ConnectError` | Retry with exponential backoff | 3 attempts |
| `httpx.ConnectTimeout` | Retry with exponential backoff | 3 attempts |
| HTTP 4xx (client errors) | No retry, pass through immediately | N/A |
| HTTP 5xx (other server errors) | No retry, pass through immediately | N/A |

This behavior eliminates transient failures during model warm-up and improves reliability when embedding models are evicted from memory due to GPU pressure.

## Testing

The project includes a comprehensive pytest test suite with **336 tests** covering all core modules including headroom-ai compression (with config validation), tokenization, think policy evaluation, tool-call retry guard, Qdrant retrieval, embeddings, streaming, and usage tracking.

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
| --- | --- | --- |
| `base` | Base compression + config/validation | 45 |
| `code` | Code compression | 12 |
| `tokenizer` | Token counting | 19 |
| `policy` | Think policy | 35 |
| `retry` | Tool-call retry guard | 17 |
| `retrieval` | Qdrant retrieval | 15 |
| `embeddings` | Embedding endpoints | 14 |
| `streaming` | SSE streaming | 23 |
| `usage` | Usage tracking | 17 |
| `integration` | Real headroom tests | 10 |

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
