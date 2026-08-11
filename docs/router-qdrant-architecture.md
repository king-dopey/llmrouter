# Router + Qdrant Architecture

**Generated:** 2026-08-06  
**Scope:** OpenAI-compatible API gateway with Qdrant vector database for retrieval-augmented generation (RAG)  
**Workspace:** This repository (`llmrouter`) — router and qdrant services only  
**Parent Document:** [Master Architectural Map](master-architectural-map.md)

---

## Executive Summary

This document describes the **Router + Qdrant** project: an OpenAI-compatible API gateway that routes requests to Ollama LLM backends, applies per-model policy routing, context management via headroom-ai compression, and think-control for all models. It optionally integrates with Qdrant vector database for RAG retrieval.

This workspace depends on two external services:
- **[Jetson Ollama Architecture](jetson-ollama-architecture.md)** — External LLM backend (Ollama) serving models via `OLLAMA_BASE_URL`
- **[Jetson ASR Architecture](jetson-asr-architecture.md)** — Optional speech recognition service for transcription

**Critical:** Router and ASR have **zero cross-dependencies**. No Router file imports from `asr/` and vice versa. They can be cleanly separated.

---

## Purpose

The router provides:
1. **OpenAI-compatible API endpoints** (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`) for client compatibility with LibreChat, Zoo Code, and other tools
2. **Per-model policy routing** — Selects appropriate Ollama backend based on model name from `models.yaml`
3. **Think-control policy** — Injects or suppresses the `think` flag based on tool patterns, content size, and summary heuristics
4. **Headroom-ai context management** — Inline compression layer before forwarding to Ollama
5. **Token counting** — Approximate and tiktoken-based token estimation
6. **Qdrant retrieval** (optional) — Vector database integration for RAG with document ingestion

---

## Docker Services

### Router Service (`fastapi-router`)

| Property | Value |
|----------|-------|
| Image | `local/ollama-openai-router:0.1.0` |
| Container Name | `ollama-openai-router` |
| Restart Policy | `unless-stopped` |
| Ports | `${ROUTER_BIND_IP:-0.0.0.0}:${ROUTER_PORT:-4000}:${ROUTER_INTERNAL_PORT:-4000}` (host IP:host port:container port) |

**Dependencies:** None (connects to Ollama via HTTP)

### Qdrant Service (`qdrant`)

| Property | Value |
|----------|-------|
| Image | `${QDRANT_IMAGE:-qdrant/qdrant:latest}` |
| Container Name | `qdrant` |
| Restart Policy | `unless-stopped` |
| Ports | `${ROUTER_BIND_IP:-0.0.0.0}:${QDRANT_PORT:-6333}:6333` |
| Memory Limit | `${QDRANT_MEMORY_LIMIT:-4G}` |

**Dependencies:** None (standalone vector database)

---

## Core Application Files

### Python Modules (Root Level)

| File | Purpose | Dependencies |
|------|---------|--------------|
| [`app.py`](../app.py) | Main FastAPI entry point with OpenAI-compatible endpoints | `policy`, `retrieval`, `router_headroom`, `tokenizer` |
| [`policy.py`](../policy.py) | Think-control policy engine (tool patterns, summary heuristics, character threshold) | `os`, `re`, `dataclasses`, `typing` |
| [`retrieval.py`](../retrieval.py) | Vector retrieval module using Qdrant for RAG | `qdrant_client`, `httpx`, `hashlib` |
| [`tokenizer.py`](../tokenizer.py) | Token counting utility (approximate + tiktoken modes) | `tiktoken`, `yaml`, `json` |
| [`router_headroom.py`](../router_headroom.py) | Headroom-ai compression adapter for context management | `headroom`, `tokenizer` |
| [`ingest_repo.py`](../ingest_repo.py) | Document ingestion script for Qdrant vector store | `qdrant_client`, `argparse`, `ast` |
| [`entrypoint.sh`](../entrypoint.sh) | Container entrypoint for starting uvicorn | N/A (shell) |

### Docker & Configuration Files

| File | Purpose | Belongs To |
|------|---------|------------|
| [`docker-compose.yml`](../docker-compose.yml) | Root compose defining router + qdrant services | **Router** |
| [`Dockerfile`](../Dockerfile) | Router container image definition (multi-stage with `audit` target) | **Router** |
| [`requirements.txt`](../requirements.txt) | Python dependencies for router | **Router** |
| [`.env.example`](../.env.example) | Environment configuration template | **Router** |

### Test Files

| File | Purpose | Marker | Count |
|------|---------|--------|-------|
| [`tests/test_app.py`](../tests/test_app.py) | Core application logic and request handling | `integration` | 9 |
| [`tests/test_chat_completions_integration.py`](../tests/test_chat_completions_integration.py) | Chat completion flow end-to-end (mocked) | `integration` | 5 |
| [`tests/test_embeddings_endpoint.py`](../tests/test_embeddings_endpoint.py) | Embeddings API endpoint | `embeddings` | 4 |
| [`tests/test_headroom_base.py`](../tests/test_headroom_base.py) | Base compression tests | `base`, `compression`, `error_handling` | 31 |
| [`tests/test_headroom_code.py`](../tests/test_headroom_code.py) | Code compression with tree-sitter | `code`, `compression` | 12 |
| [`tests/test_headroom_integration.py`](../tests/test_headroom_integration.py) | Integration tests with real headroom | `integration` | 11 |
| [`tests/test_headroom_relevance.py`](../tests/test_headroom_relevance.py) | Relevance scoring with fastembed | `relevance`, `compression` | 11 |
| [`tests/test_nonstream_usage.py`](../tests/test_nonstream_usage.py) | Non-streaming response usage tracking | `usage` | 2 |
| [`tests/test_orin_policy.py`](../tests/test_orin_policy.py) | Orin profile policy tests | `policy` | — |
| [`tests/test_policy.py`](../tests/test_policy.py) | Think policy engine (unittest) | `policy` | 7 |
| [`tests/test_profile_alignment.py`](../tests/test_profile_alignment.py) | Cross-profile consistency checks | `integration` | — |
| [`tests/test_retrieval.py`](../tests/test_retrieval.py) | Qdrant retrieval module | `retrieval` | 15 |
| [`tests/test_stream_usage.py`](../tests/test_stream_usage.py) | Stream usage chunk output | `streaming` | 2 |
| [`tests/test_streaming.py`](../tests/test_streaming.py) | SSE streaming and tool call translation | `streaming` | 23 |
| [`tests/test_think_policy.py`](../tests/test_think_policy.py) | Think flag injection policy | `policy` | 35 |
| [`tests/test_thor_policy.py`](../tests/test_thor_policy.py) | Thor profile policy tests | `policy` | — |
| [`tests/test_tokenizer.py`](../tests/test_tokenizer.py) | Token counting (approximate + tiktoken) | `tokenizer` | 19 |
| [`tests/test_usage_tracking.py`](../tests/test_usage_tracking.py) | Usage token tracking in responses | `usage` | 17 |

### Shared Configuration Files

These files are used by **multiple projects** and should be referenced from external workspaces:

| File | Router Use | Ollama Use | ASR Use | Notes |
|------|------------|------------|---------|-------|
| [`profiles/orin/stack.env`](../profiles/orin/stack.env) | ✅ | ✅ | ✅ | Environment variables for Orin profile |
| [`profiles/thor/stack.env`](../profiles/thor/stack.env) | ✅ | ✅ | ✅ | Environment variables for Thor profile |
| [`profiles/orin/models.yaml`](../profiles/orin/models.yaml) | ✅ | ✅ | — | Model/residency policy (Router + Ollama) |
| [`profiles/thor/models.yaml`](../profiles/thor/models.yaml) | ✅ | ✅ | — | Model/residency policy (Router + Ollama) |

---

## Environment Variables

### Docker Compose Variables (`.env.example`)

These variables control Docker Compose behavior and are set in the root `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROUTER_BIND_IP` | `192.168.1.20` | Router bind interface address |
| `ROUTER_PORT` | `4000` | Router HTTP port |
| `QDRANT_PORT` | `6333` | Qdrant HTTP port |
| `HARDWARE_PROFILE` | `orin` | Hardware profile selection (`orin` or `thor`) |
| `QDRANT_IMAGE` | `qdrant/qdrant:latest` | Qdrant container image (CPU, NVIDIA GPU, or AMD GPU variants) |
| `QDRANT_MEMORY_LIMIT` | `4G` | Maximum container memory for Qdrant |
| `HF_CACHE_HOST_PATH` | `./data/hf_cache` | Hugging Face model cache directory |
| `HEADROOM_CCR_HOST_PATH` | `./data/headroom_ccr` | Headroom CCR store path |
| `QDRANT_STORAGE_HOST_PATH` | `./data/qdrant_storage` | Qdrant storage directory |

### Router Runtime Variables (`profiles/router.env`)

These variables are loaded via Docker Compose `env_file` directive for the router container:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama backend URL (points to [Jetson Ollama](jetson-ollama-architecture.md)) |
| `ROUTER_INTERNAL_PORT` | `4000` | Port Uvicorn listens on inside the container |
| `ROUTER_BIND_IP` | `0.0.0.0` | Router bind interface (Docker Compose host-side binding) |
| `ROUTER_PORT` | `4000` | Host-side port mapped to internal container port |
| `KEEP_ALIVE_DEFAULT` | `-1` | Default model keep_alive duration |
| `DISABLE_THINK_TOOL_PATTERNS` | `web_search,search,browser,browse,serp,http_get,http,fetch,scrape` | Tool patterns that disable think |
| `DISABLE_THINK_SUMMARY_PATTERNS` | `summarize,tl;dr,tldr,bullet summary,key takeaways` | Summary patterns that disable think |
| `DISABLE_THINK_CHAR_THRESHOLD` | `12000` | Character threshold for disabling think on large content |
| `AUTO_PULL_MISSING_MODELS` | `true` | Automatically pull missing models during warmup |
| `MODEL_PULL_TIMEOUT_SEC` | `7200` | Maximum time to wait for model pull (seconds) |
| `MODEL_PULL_MAX_RETRIES` | `2` | Retry count for pull operations |
| `MODEL_PULL_BACKOFF_SEC` | `5` | Backoff delay between retries (seconds) |
| `HEADROOM_ENABLED` | `1` | Enable headroom-ai compression (`0` to disable) |
| `HEADROOM_RELEVANCE_ENABLED` | `1` | Enable relevance scoring (`0` to disable) |
| `HF_HOME` | `/data/hf_cache` | Hugging Face home directory for kompress-v2-base ONNX model |
| `HUGGINGFACE_HUB_CACHE` | `/data/hf_cache/hub` | Hugging Face hub cache |
| `TRANSFORMERS_CACHE` | `/data/hf_cache/models` | Transformers model cache |
| `HEADROOM_CCR_STORE_PATH` | `/data/headroom_ccr` | Headroom CCR store path |

### Qdrant Runtime Variables (`profiles/qdrant.env`)

These variables are loaded via Docker Compose `env_file` directive for the qdrant container:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_QDRANT_RETRIEVAL` | `true` | Enable/disable RAG retrieval |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL (router connects here) |
| `QDRANT_COLLECTION` | `repo_chunks` | Vector collection name for document storage |
| `QDRANT_EMBEDDING_MODEL` | `qwen3-embedding:4b` | Embedding model for vector generation |
| `QDRANT_TOP_K` | `20` | Maximum candidate documents to retrieve |
| `QDRANT_FINAL_K` | `8` | Final documents passed to LLM after filtering |
| `QDRANT__GPU__INDEXING` | `false` | Enable GPU-accelerated vector indexing |
| `QDRANT__TELEMETRY_DISABLED` | `true` | Disable Qdrant telemetry |
| `QDRANT_OPTIMIZERS_MAX_MEMORY_BYTES` | `4294967296` | Maximum memory for optimizer operations (4 GB) |
| `QDRANT_CLUSTER_ENABLE` | `false` | Enable clustering for distributed deployments |
| `RUST_LOG` | `info` | Rust logging level for Qdrant container |

### Hardware-Specific Variables (`profiles/jetson/<profile>.env`)

These variables are loaded based on `HARDWARE_PROFILE`:

| Variable | Orin Default | Thor Default | Description |
|----------|--------------|--------------|-------------|
| `MODEL_DEFAULT` | `qwen3.6:35b-a3b-q8_0` | (see thor.env) | Default chat model ID |
| `EMBEDDING_MODEL_DEFAULT` | `qwen3-embedding:4b` | `qwen3-embedding:4b` | Default embedding model ID |

---

## Hardware Profiles

### Orin Profile (Jetson AGX Orin 64 GB)

| Property | Value |
|----------|-------|
| CPU | ARM64 (aarch64) 8-core Cortex-A78AE |
| GPU | NVIDIA Ampere SM 87 (384 CUDA cores) |
| RAM | 64 GB unified LPDDR5 |
| Architecture | System-on-Module (SoM) with shared CPU/GPU memory |

**Models defined in:** [`profiles/orin/models.yaml`](../profiles/orin/models.yaml)

### Thor Profile (Jetson Thor)

| Property | Value |
|----------|-------|
| CPU | ARM64 (aarch64) |
| GPU | NVIDIA Blackwell SM 110 |
| RAM | 64+ GB unified LPDDR5 |
| Architecture | Next-generation Jetson platform |

**Models defined in:** [`profiles/thor/models.yaml`](../profiles/thor/models.yaml)

---

## Think Control Policy

The router injects the `think` flag in requests to Ollama based on policy rules:

### Policy Defaults

- Per-model defaults come from the active profile's `models.yaml` (mounted as `/app/model_policy.yml`)
- `qwen3-coder:30b` (Orin) and `qwen3-coder-next:q4_K_M` (Thor) default to `think=false`
- Most general-purpose models default to `think=true`
- `think=false` for web/browse/search-style tool flows (`web_search`, `browser`, `http_get`, `fetch`, `scrape`)
- `think=true` for non-web tools (`file_search`, `openweather`) unless summarization/size heuristics trigger `think=false`
- `think=false` when message content exceeds `DISABLE_THINK_CHAR_THRESHOLD` (12000 chars) or for summary-like last-user turns

### Manual Override Header

Clients can override policy with:
- `X-Ollama-Think: true` — Force think enabled
- `X-Ollama-Think: false` — Force think disabled

If the override header is present, it takes precedence over policy.

---

## Headroom-AI Context Management

The router uses [headroom-ai](https://github.com/headroomlabs-ai/headroom) as an inline compression layer before forwarding requests to Ollama.

### How It Works

1. Router injects optional retrieval context from Qdrant
2. Constructs complete Ollama payload
3. Calls `headroom.compress(messages, model=..., headroom_query=...)` once on finalized message list
4. Extracts user's query from last user message as `headroom_query` for relevance scoring (when enabled)
5. Checks compressed prompt against configured context window with reserved output and safety tokens
6. Rejects if compressed prompt still cannot fit (no silent history discard)

### API Contract

The router calls `headroom.compress()` with:
- `messages`: Finalized message list (required)
- `model`: Model name for token counting (required)
- `headroom_query`: User's query for relevance scoring (optional, extracted from last user message)

**Note:** The router does not pass invalid parameters such as `compress_user_messages`, `target_ratio`, `protect_recent`, or `relevance_threshold`. These are not part of the headroom.compress() API contract.

### Policy Configuration

Each model in `models.yaml` defines:
- `reserved_output_tokens`: Tokens reserved for generation output (e.g., `4096`)
- `safety_headroom_tokens`: Extra buffer for unexpected tokens (e.g., `4096`)

---

## Qdrant Retrieval Integration

### Configuration

Enable retrieval by setting `ENABLE_QDRANT_RETRIEVAL=true` in [`profiles/qdrant.env`](../profiles/qdrant.env).

### Document Ingestion

Use [`ingest_repo.py`](../ingest_repo.py) to ingest repository documents into Qdrant:

```bash
python3 ingest_repo.py --path /path/to/repo --collection repo_chunks
```

### Retrieval Flow

1. Client sends chat completion request
2. Router queries Qdrant for relevant chunks based on user message embedding
3. Retrieved context injected into system prompt before forwarding to Ollama
4. Response returned with compressed history via headroom-ai

---

## Testing

The project includes a comprehensive pytest test suite with **207 tests** covering all core modules.

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
| `base` | Base compression (headroom-ai without extras) | 31 |
| `code` | Code compression with tree-sitter | 12 |
| `relevance` | Relevance scoring with fastembed | 11 |
| `tokenizer` | Token counting and estimation | 19 |
| `policy` | Think policy configuration | 35 |
| `retrieval` | Qdrant context retrieval | 15 |
| `embeddings` | Embedding generation endpoints | 14 |
| `streaming` | SSE streaming and tool calls | 23 |
| `usage` | Usage tracking in responses | 17 |
| `integration` | End-to-end pipeline tests | 11 |

### Validation Scripts

Run validation scripts to verify profile configuration:

```bash
# Validate shared stack configuration
scripts/validation/validate-shared-stack.sh orin
scripts/validation/validate-shared-stack.sh thor

# Validate model tags are available (requires Ollama running)
HARDWARE_PROFILE=orin scripts/validation/validate-model-tags.sh orin
HARDWARE_PROFILE=thor scripts/validation/validate-model-tags.sh thor

# Validate runtime gates (requires Ollama running)
HARDWARE_PROFILE=orin scripts/validation/validate-runtime-gates.sh orin
HARDWARE_PROFILE=thor scripts/validation/validate-runtime-gates.sh thor
```

---

## GPU Acceleration Support

Qdrant supports GPU-accelerated vector indexing for NVIDIA and AMD GPUs.

### Hardware Requirements

| GPU Type | Host Dependencies | Docker Image |
|----------|-------------------|--------------|
| NVIDIA | `nvidia-container-toolkit` installed on host | `qdrant/qdrant:gpu-nvidia-latest` |
| AMD | Vulkan runtime (`vulkan-tools`, `vulkan-radeon`) installed on host | `qdrant/qdrant:gpu-amd-latest` |

### Environment Variables for GPU Support

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT__GPU__INDEXING` | `false` | Set to `true` to enable GPU-accelerated vector indexing |
| `QDRANT__OPTIMIZERS__MAX_MEMORY_BYTES` | `4294967296` (4 GB) | Maximum memory for vector operations |
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

---

## Cross-Project Dependencies

### Router → Ollama (External Service)

| Dependency Type | Details |
|-----------------|---------|
| **Runtime** | HTTP connection to `OLLAMA_BASE_URL` (see [Jetson Ollama Architecture](jetson-ollama-architecture.md)) |
| **Code Imports** | NONE — Router connects via HTTP, no source code dependency |
| **Configuration** | Reads model definitions from `models.yaml` (shared with Ollama) |

### Router → ASR (External Service)

| Dependency Type | Details |
|-----------------|---------|
| **Runtime** | Optional HTTP connection to ASR service (`ASR_BASE_URL`, `ASR_PORT`) |
| **Code Imports** | NONE — Zero cross-dependencies |
| **Configuration** | ASR variables in `profiles/router.env` (optional) |

---

## Startup and Deployment

### Start Services

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

### Verify Services Running

Verify the router is running:

```bash
curl -sS http://127.0.0.1:${ROUTER_PORT:-4000}/v1/models | jq .
```

Verify Qdrant is running (if enabled):

```bash
curl -sf http://127.0.0.1:6333/healthcheck
```

---

## Network and Security Notes

### Port Configuration

The router uses two distinct port concepts for flexibility in deployment:

| Variable | Scope | Purpose | Default |
| --- | --- | --- | --- |
| `ROUTER_INTERNAL_PORT` | Container runtime (via [`profiles/router.env`](../profiles/router.env)) | The port Uvicorn listens on inside the container | `4000` |
| `ROUTER_PORT` | Docker Compose host-side mapping (via `.env`) | The external port mapped to the internal container port | `4000` |
| `ROUTER_BIND_IP` | Docker Compose host-side binding (via `.env`) | The network interface on which the host port is exposed | `0.0.0.0` |

The Docker Compose port mapping follows this pattern: `${ROUTER_BIND_IP:-0.0.0.0}:${ROUTER_PORT:-4000}:${ROUTER_INTERNAL_PORT:-4000}` (host IP:host port:container port). This allows the internal container port to differ from the externally exposed host port when needed.

### Exposed Ports

- `0.0.0.0:<ROUTER_PORT>` (OpenAI-compatible router, default `4000`)
- `0.0.0.0:<QDRANT_PORT>` (Qdrant vector database, optional, default `6333`)
- Restrict access at host firewall/router ACLs to trusted LAN clients.

---

## Related Documentation

| Document | Description | Location |
|----------|-------------|----------|
| [Master Architectural Map](master-architectural-map.md) | Complete repository analysis for three-project separation | `docs/` |
| [Jetson Ollama Architecture](jetson-ollama-architecture.md) | External LLM backend (Ollama) serving models | `../jetson-ollama/docs/` |
| [Jetson ASR Architecture](jetson-asr-architecture.md) | Speech recognition service with dynamic providers | `../jetson-asr/docs/` |
| [Unified Orin + Thor Profile Runbook](UnifiedOrinThorProfiles.md) | Profile management, start commands, validation gates | `docs/` |
| [Testing Strategy](testing-strategy.md) | Comprehensive test suite documentation | `docs/` |
| [TESTING.md](../TESTING.md) | Quick testing guide with Docker-based workflow | Root directory |

---

*This document covers only the router and qdrant services. For Ollama backend architecture, see [Jetson Ollama Architecture](jetson-ollama-architecture.md). For ASR service architecture, see [Jetson ASR Architecture](jetson-asr-architecture.md).*
