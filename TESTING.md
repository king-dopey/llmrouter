# Testing Guide

Comprehensive pytest suite for the OpenAI-Compatible Router Node for Ollama LLM.

## Table of Contents

- [Overview](#overview)
- [Running Tests](#running-tests)
- [Test Structure](#test-structure)
- [Test Categories](#test-categories)
- [Adding New Tests](#adding-new-tests)

## Overview

The project uses **pytest** (v7.4+) for unit testing with a comprehensive suite of **207 tests** covering all core modules.

**Important:** Tests must be run inside the Docker container. Running tests on the host system will fail because `headroom-ai` and its dependencies (PyTorch, transformers, etc.) require a specific environment that is only available in the container.

## Running Tests

### Docker-Based Testing (Required)

All tests must be run inside the Docker container. The `audit` target in the [`Dockerfile`](Dockerfile) runs the full test suite automatically during build.

**Important:** The audit target is a multi-stage build target in the Dockerfile, not a docker-compose service. You must use `docker build` directly:

```bash
# Build the audit target (runs all tests during build)
docker build --target audit -t llmrouter-audit .
```

The audit target:
1. Installs all dependencies from [`requirements.txt`](requirements.txt) including `headroom-ai[code,relevance]`
2. Copies all test files
3. Runs `python3 -m pytest tests/ -v`

**Note:** `docker compose build audit` will NOT work because the audit target is not defined as a service in docker-compose.yml.

### Why Host Testing Fails

The integration tests ([`tests/test_headroom_integration.py`](tests/test_headroom_integration.py)) require the actual `headroom-ai` library with ML dependencies. These tests will fail at collection time if headroom is not installed:

```
RuntimeError: headroom library is not installed. Integration tests require headroom-ai to be installed (see requirements.txt).
```

The host system typically lacks:
- PyTorch with correct CUDA/CPU configuration
- transformers library
- sentence-transformers
- headroom-ai with [code,relevance] extras

### Running Specific Tests in Docker

After building the audit image, you can run specific test markers or files:

```bash
# First, build the audit image (required before running tests)
docker build --target audit -t llmrouter-audit .

# Run only headroom base tests
docker run --rm llmrouter-audit python3 -m pytest tests/ -m base -v

# Run specific test file
docker run --rm llmrouter-audit python3 -m pytest tests/test_tokenizer.py -v

# Run with coverage
docker run --rm llmrouter-audit python3 -m pytest tests/ --cov=app --cov=router_headroom --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures for headroom compression
├── test_headroom_base.py          # Base compression (31 tests)
├── test_headroom_code.py          # Code compression (12 tests)
├── test_headroom_relevance.py     # Relevance scoring (11 tests)
├── test_headroom_integration.py   # Integration tests with real headroom (11 tests)
├── test_tokenizer.py              # Token counting (19 tests)
├── test_think_policy.py           # Think policy evaluation (35 tests)
├── test_retrieval.py              # Qdrant retrieval (15 tests)
├── test_embeddings.py             # Embedding generation (14 tests)
├── test_streaming.py              # SSE streaming (23 tests)
├── test_usage_tracking.py         # Usage token tracking (17 tests)
├── test_app.py                    # App endpoint tests (unittest, 9 tests)
├── test_chat_completions_integration.py  # Integration tests (unittest, 5 tests)
├── test_embeddings_endpoint.py    # Embeddings endpoint (unittest, 4 tests)
├── test_nonstream_usage.py        # Non-stream usage (unittest, 2 tests)
├── test_policy.py                 # Think policy (unittest, 7 tests)
├── test_stream_usage.py           # Stream usage chunk (unittest, 2 tests)
```

### Shared Fixtures (`conftest.py`)

The [`tests/conftest.py`](tests/conftest.py) module provides centralized fixtures for headroom compression testing:

| Fixture | Purpose |
|---------|---------|
| `mock_compress_result()` | Factory to create CompressResult-like SimpleNamespace |
| `mock_compress_result_dict()` | Factory to create dict-based CompressResult |
| `mock_compress_list()` | Factory to create plain list results |
| `patch_headroom_compress()` | Patch headroom.compress with CompressResult return |
| `patch_headroom_compress_list()` | Patch headroom.compress with plain list return |
| `patch_headroom_compress_dict()` | Patch headroom.compress with dict return |
| `patch_token_count()` | Patch tokenizer count with configurable side effects |
| `patch_headroom_disabled()` | Disable HEADROOM_ENABLED via environment |
| `patch_headroom_enabled()` | Enable HEADROOM_ENABLED via environment |
| `patch_relevance_disabled()` | Disable HEADROOM_RELEVANCE_ENABLED via environment |
| `patch_relevance_enabled()` | Enable HEADROOM_RELEVANCE_ENABLED via environment |

## Test Categories

### Headroom Compression Tests (54 tests)

Tests for the headroom-ai inline compression layer.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `base` | Base compression without extras | `test_headroom_base.py` | 31 |
| `code` | Code compression with tree-sitter | `test_headroom_code.py` | 12 |
| `relevance` | Relevance scoring with fastembed | `test_headroom_relevance.py` | 11 |

**Coverage:**
- `check_and_trim()` function with all output types (CompressResult, dict, plain list)
- `_compress()` with policy options (compress_user_messages, target_ratio, relevance_threshold)
- Budget enforcement (reserved_output_tokens, safety_headroom_tokens)
- Configuration flags (HEADROOM_ENABLED, HEADROOM_RELEVANCE_ENABLED)
- Error handling (missing dependency, TypeError fallback, general exceptions)
- Telemetry extraction (tokens_before, tokens_after, transforms_applied)

### Integration Tests (11 tests)

Tests with the real headroom library (not mocked) to verify the API contract.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `integration` | End-to-end pipeline tests | `test_headroom_integration.py` | 11 |

**Note:** These tests require headroom-ai to be installed and will fail at collection time if not available.

### Tokenizer Tests (19 tests)

Tests for token counting with tiktoken and approximate methods.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `tokenizer` | Token counting and estimation | `test_tokenizer.py` | 19 |

### Think Policy Tests (35 tests)

Tests for the think flag injection policy based on tool names, content size, and summary requests.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `policy` | Think policy configuration | `test_think_policy.py` | 35 |

### Retrieval Tests (15 tests)

Tests for Qdrant-based context retrieval and injection.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `retrieval` | Qdrant context retrieval | `test_retrieval.py` | 15 |

### Embeddings Tests (14 tests)

Tests for the `/v1/embeddings` endpoint.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `embeddings` | Embedding generation endpoints | `test_embeddings.py` | 14 |

### Streaming Tests (23 tests)

Tests for SSE streaming and tool call translation.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `streaming` | SSE streaming and tool calls | `test_streaming.py` | 23 |

### Usage Tracking Tests (17 tests)

Tests for non-streaming response usage token tracking.

| Marker | Description | File | Count |
|--------|-------------|------|-------|
| `usage` | Usage tracking in responses | `test_usage_tracking.py` | 17 |

## Adding New Tests

### 1. Create Test File

Create a new test file following the naming convention `test_<module>.py`:

```python
"""Tests for <module> module with pytest markers."""

import pytest

# Import functions to test
from app import _build_ollama_payload


@pytest.mark.payload
class TestPayloadBuilding:
    """Tests for Ollama payload building."""

    def test_policy_defaults_applied(self):
        """Test that policy defaults are applied to payload."""
        body = {
            "model": "qwen3-coder-next:q4_K_M",
            "messages": [{"role": "user", "content": "hi"}],
        }
        # Test implementation...
```

### 2. Register Marker

Add the marker to [`pytest.ini`](pytest.ini):

```ini
markers =
    ...
    payload: Tests for Ollama payload building
```

### 3. Add Fixtures (if needed)

Add shared fixtures to [`tests/conftest.py`](tests/conftest.py):

```python
@pytest.fixture
def mock_ollama_response():
    """Factory fixture to create mock Ollama API responses."""
    def _create(payload=None, status_code=200):
        return {"status_code": status_code, "json": lambda: payload or {}}
    return _create
```

### 4. Run and Verify

```bash
# Build the audit image (runs all tests during build)
docker build --target audit -t llmrouter-audit .

# Run specific tests in container
docker run --rm llmrouter-audit python3 -m pytest tests/test_<module>.py -v
```

## Best Practices

### Test Organization

1. **One module per test file**: Each test file should cover a single source module
2. **Class-based organization**: Group related tests in classes with descriptive names
3. **Marker consistency**: Use markers consistently to enable filtering by feature area
4. **No duplication**: Avoid testing the same functionality in multiple files

### Test Naming

1. **Descriptive names**: Use `test_<functionality>_<condition>` format
2. **Include expected outcome**: Names should indicate what is being tested and the expected result

### Mocking

1. **Use fixtures**: Leverage shared fixtures from `conftest.py` where possible
2. **Mock external dependencies**: Always mock HTTP clients, file I/O, and environment-dependent code
3. **Verify call arguments**: Use `assert call_args.kwargs.get(...)` to verify parameters passed to mocked functions

### Environment Variables

1. **Clean up after tests**: Use `try/finally` or fixtures to restore environment variables
2. **Test both enabled/disabled**: Test functionality with and without feature flags
3. **Document required env vars**: Add docstrings explaining which environment variables affect the test

## Troubleshooting

### Tests fail with "headroom library is not installed"

This is expected when running outside Docker. The integration tests require headroom-ai to be installed. Run tests inside the Docker container:

```bash
docker build --target audit -t llmrouter-audit .
```

### Unknown Marker Warning

If you see `PytestUnknownMarkWarning: Unknown pytest.mark.<name>`, ensure the marker is registered in [`pytest.ini`](pytest.ini).

### Import Errors

If tests fail with `ImportError: cannot import name ...`, verify that:
1. The module path is correct relative to the test file
2. Dependencies are installed (`pip install -r requirements.txt`)
3. The `sys.path` includes the project root (handled by `ROUTER_DIR` in test files)

### Fixture Not Found

If a fixture is not found, verify:
1. The fixture is defined in `conftest.py` or the test file itself
2. The fixture scope matches the test requirements (function, class, module, session)
3. No circular imports are preventing fixture loading

## See Also

- [Testing Strategy](docs/testing-strategy.md) - Comprehensive guide to unit tests and validation scripts
- [pytest Documentation](https://docs.pytest.org/) - Official pytest documentation
- [pytest Markers](https://docs.pytest.org/en/stable/how-to/mark.html) - Marker configuration reference
