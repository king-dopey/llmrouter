# Testing Strategy

This document describes the comprehensive testing strategy for the OpenAI-Compatible Router for Ollama LLM. Tests are organized into two categories:

1. **Unit/Integration Tests** — Python `unittest` suite in [`tests/`](../tests/) that validates core logic without requiring a running container.
2. **Validation Scripts** — Bash scripts in [`scripts/validation/`](../scripts/validation/) that verify profile configuration, model availability, and runtime state against the active profile.

## Quick Reference

| Category | Location | Runner | Requires Running Stack? |
|---|---|---|---|
| Python unit tests | [`tests/`](../tests/) | `python3 -m pytest tests/` or `python3 -m unittest discover tests/` | No |
| Model tag validation | [`scripts/validation/validate-model-tags.sh`](../scripts/validation/validate-model-tags.sh) | `bash scripts/validation/validate-model-tags.sh [profile]` | Yes (Ollama container) |
| Runtime gate validation | [`scripts/validation/validate-runtime-gates.sh`](../scripts/validation/validate-runtime-gates.sh) | `bash scripts/validation/validate-runtime-gates.sh [profile]` | Yes (Ollama container) |
| Shared stack validation | [`scripts/validation/validate-shared-stack.sh`](../scripts/validation/validate-shared-stack.sh) | `bash scripts/validation/validate-shared-stack.sh [profile]` | Partial (docker compose) |
| Model warmup | [`scripts/warmup.sh`](../scripts/warmup.sh) | `bash scripts/warmup.sh` | Yes (Ollama container) |

---

## Python Unit Tests

All Python tests use the standard library [`unittest`](https://docs.python.org/3/library/unittest.html) framework. They stub out external dependencies (`httpx`, `fastapi`) so they run in isolation without a live router or Ollama backend.

### Test Files

| File | Purpose | Key Functions Tested |
|---|---|---|
| [`test_app.py`](../tests/test_app.py) | Core application logic and request handling | Router endpoint behavior, model policy enforcement, error responses |
| [`test_chat_completions_integration.py`](../tests/test_chat_completions_integration.py) | Chat completion flow end-to-end (mocked) | Message routing, tool call handling, streaming vs non-streaming paths |
| [`test_embeddings_endpoint.py`](../tests/test_embeddings_endpoint.py) | Embeddings API endpoint | Embedding request forwarding to Ollama, response parsing |
| [`test_headroom.py`](../tests/test_headroom.py) | Headroom adapter | Public compression call, telemetry normalization, post-compression context check, explicit missing dependency, disabled passthrough |
| [`test_nonstream_usage.py`](../tests/test_nonstream_usage.py) | Non-streaming response usage tracking | `prompt_tokens`, `completion_tokens`, `total_tokens` accounting |
| [`test_policy.py`](../tests/test_policy.py) | Think-control policy engine | Web-search disable, file-search enable, summarization heuristics, character threshold |
| [`test_profile_alignment.py`](../tests/test_profile_alignment.py) | Cross-profile consistency | Ensures shared models have matching `num_ctx` across Orin, Thor, and router policy |
| [`test_retrieval.py`](../tests/test_retrieval.py) | Vector retrieval (Qdrant integration) | Embedding-based document retrieval, similarity search mocking |
| [`test_stream_usage.py`](../tests/test_stream_usage.py) | Streaming response usage chunks | `_build_stream_usage_chunk()` output, inclusion/omission based on `include_usage` flag |
| [`test_tokenizer.py`](../tests/test_tokenizer.py) | Token counting (approximate mode) | `count_prompt_tokens()`, `count_completion_tokens()` with approximate tokenizer map |

### Running the Tests

```bash
# Run all tests
cd /home/heaps/repos/llmrouter
python3 -m pytest tests/ -v

# Run a single test file
python3 -m pytest tests/test_policy.py -v

# Run with unittest discover
python3 -m unittest discover tests/ -v
```

### Test Architecture

Each test file follows a common pattern:

1. **Dependency stubbing** — The `_install_dependency_stubs()` function creates minimal mock modules for `httpx` and `fastapi`, allowing the router's production code to be imported without installing all dependencies.
2. **Policy file setup** — `MODEL_POLICY_FILE` environment variable points to the default [`model_policy.yml`](../model_policy.yml).
3. **Isolated assertions** — Tests use `unittest.TestCase` methods (`assertEqual`, `assertTrue`, `assertFalse`, `assertIsNotNone`) for clear failure messages.

Some tests (e.g., [`test_headroom.py`](../tests/test_headroom.py), [`test_tokenizer.py`](../tests/test_tokenizer.py)) use `importlib.reload()` to dynamically change environment variables like `TOKENIZER_MAP` between test cases without process restart.

---

## Validation Scripts

Bash validation scripts verify the operational readiness of the router stack. They are designed to be run before deployment or as part of CI/CD pipelines.

### 1. Model Tag Validation — [`validate-model-tags.sh`](../scripts/validation/validate-model-tags.sh)

**Purpose:** Ensures all models declared in a profile's `models.yaml` are available (pulled) in the Ollama instance.

**How it works:**
1. Reads model names from the profile's `models.yaml` using `awk`.
2. For each model, queries `OLLAMA_HOST/api/tags`.
3. If a model is missing, attempts to pull it with retry logic.
4. Exits with error if any model cannot be pulled.

**Usage:**
```bash
# Validate Orin profile models
PROFILE=orin bash scripts/validation/validate-model-tags.sh orin

# Validate Thor profile models
PROFILE=thor bash scripts/validation/validate-model-tags.sh thor
```

**Exit codes:** `0` = all models available; `1` = model pull failed or missing profile file.

### 2. Runtime Gate Validation — [`validate-runtime-gates.sh`](../scripts/validation/validate-runtime-gates.sh)

**Purpose:** Verifies that critical models are resident (warm) in the Ollama instance at runtime, confirming the warmup process succeeded.

**How it works:**
1. Queries `OLLAMA_HOST/api/tags` to confirm Ollama is reachable.
2. Queries `OLLAMA_HOST/api/ps` to list currently loaded models.
3. For **Orin**: checks that `qwen3-coder:30b` is resident.
4. For **Thor**: checks that both `qwen3-coder-next:q4_K_M` and `qwen3.6:35b-a3b-q8_0` are resident.

**Usage:**
```bash
# Validate Orin runtime gates
PROFILE=orin bash scripts/validation/validate-runtime-gates.sh orin

# Validate Thor runtime gates
PROFILE=thor bash scripts/validation/validate-runtime-gates.sh thor
```

**Environment variables:**
| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API endpoint |

### 3. Shared Stack Validation — [`validate-shared-stack.sh`](../scripts/validation/validate-shared-stack.sh)

**Purpose:** Validates that Docker Compose configuration correctly mounts profile files and uses consistent image pins across the main stack and router stack.

**How it works:**
1. Checks that `models.yaml` and `librechat-modelspecs.yaml` exist for the profile.
2. Runs `docker compose config` (both main and router compose files) to parse and expand the configuration.
3. Asserts specific content in the expanded config:
   - Shared Ollama image pin (`ollama/ollama:0.30.10`)
   - Correct volume mount source and target for profile files
   - ASR Dockerfile environment variable presence
4. Verifies `librachat.json` is pinned to LibreChat 0.8.6.

**Usage:**
```bash
bash scripts/validation/validate-shared-stack.sh orin
bash scripts/validation/validate-shared-stack.sh thor
```

### 4. Thor ASR Validation — [`validate-thor-asr.sh`](../scripts/validation/validate-thor-asr.sh)

**Purpose:** Verifies that the Thor Automatic Speech Recognition (ASR) configuration files exist and Docker Compose can parse the Thor profile.

**How it works:**
1. Checks for `asr/Dockerfile.thor`.
2. Checks for `asr/requirements-thor.txt`.
3. Runs `docker compose config` to verify YAML validity.

**Usage:**
```bash
bash scripts/validation/validate-thor-asr.sh
```

---

## Model Warmup Script — [`warmup.sh`](../scripts/warmup.sh)

**Purpose:** Ensures critical models are pulled and resident in Ollama with the correct context length before the router begins serving traffic.

### How It Works

1. **Wait for Ollama** — Polls `/api/tags` for up to 120 seconds.
2. **Parse model entries** — Each entry in `WARMUP_MODELS` is formatted as `model_name@num_ctx` (e.g., `qwen3-coder:30b@16384`).
3. **Check residency** — Queries `/api/ps` to see if the model is already resident with the correct context length.
4. **Pull if needed** — Streams pull progress with exponential backoff retry (up to `PULL_MAX_RETRIES`, default 3).
5. **Warm the model** — Sends a minimal `/api/generate` request with `keep_alive=-1` and the target `num_ctx`.
6. **Confirm residency** — Verifies the model appears in `/api/ps` with the correct `context_length`.

### Configuration Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://ollama:11434` | Ollama API endpoint |
| `WARMUP_MODELS` | `qwen3-coder:30b@16384 qwen3.6:35b-a3b@32768` | Space-separated list of models with context lengths |
| `WARMUP_DEFAULT_NUM_CTX` | `16384` | Default context length if not specified in entry |
| `KV_CACHE_TYPE` | `q8_0` | KV cache type for warm requests |
| `PULL_MAX_RETRIES` | `3` | Maximum pull attempts per model |
| `PULL_BACKOFF_SEC` | `10` | Base backoff delay for retries |

### Status Codes

| Status | Meaning |
|---|---|
| `already-warm` | Model was already resident with correct context |
| `pulled-warmed` | Model was pulled and warmed in this run |
| `already-pulled-warmed` | Model was already pulled and rewarmed |
| `pull-failed` | Pull request returned non-zero exit code |
| `post-pull-missing` | Pull succeeded but model not in `/api/tags` |
| `warm-failed` | Warm request returned HTTP error |
| `not-resident` | Model loaded but not confirmed resident after 5 attempts |
| `wrong-ctx` | Model is resident but with incorrect context length |

---

## Test Coverage Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Testing Strategy Overview                     │
├──────────────────────┬──────────────────────────────────────────┤
│ Layer                │ Tests                                    │
├──────────────────────┼──────────────────────────────────────────┤
│ Policy Engine        │ test_policy.py                           │
│ Think Control        │ test_policy.py                           │
│ Headroom/Context     │ test_headroom.py                         │
│ Tokenizer            │ test_tokenizer.py                        │
│ Chat Completions     │ test_chat_completions_integration.py     │
│ Stream Usage         │ test_stream_usage.py                     │
│ Non-Stream Usage     │ test_nonstream_usage.py                  │
│ Embeddings           │ test_embeddings_endpoint.py              │
│ Retrieval            │ test_retrieval.py                        │
│ App Core             │ test_app.py                              │
│ Profile Alignment    │ test_profile_alignment.py                │
├──────────────────────┼──────────────────────────────────────────┤
│ Pre-Deploy Validation│ validate-model-tags.sh                   │
│ Runtime Verification │ validate-runtime-gates.sh                │
│ Stack Config         │ validate-shared-stack.sh                 │
│ Model Warmup         │ warmup.sh                                │
└──────────────────────┴──────────────────────────────────────────┘
```

---

## Recommended Test Workflow

1. **Before deployment:**
   ```bash
   # Run all unit tests (no running stack required)
   python3 -m pytest tests/ -v
   
   # Validate profile configuration
   bash scripts/validation/validate-shared-stack.sh orin
   
   # Start the stack
   docker compose up -d
   
   # Wait for Ollama to be ready, then warm models
   bash scripts/warmup.sh
   
   # Validate models are available
   PROFILE=orin bash scripts/validation/validate-model-tags.sh orin
   
   # Validate runtime residency
   PROFILE=orin bash scripts/validation/validate-runtime-gates.sh orin
   ```

2. **CI/CD pipeline:**
   ```bash
   # Step 1: Unit tests (fast, no dependencies)
   python3 -m pytest tests/ -v --tb=short
   
   # Step 2: Stack config validation
   bash scripts/validation/validate-shared-stack.sh "$PROFILE"
   
   # Step 3: Start stack, warm models, runtime validation
   docker compose up -d
   bash scripts/warmup.sh
   PROFILE="$PROFILE" bash scripts/validation/validate-runtime-gates.sh "$PROFILE"
   ```
