# Run Tests

description: Use these rules when you need to run the pytest test suite for the LLM Router project.

## Critical Constraint

**Tests must run inside the Docker container.** Running tests on the host will fail because `headroom-ai` and ML dependencies are only available in the container environment.

**Important:** The audit target is a multi-stage build target in the Dockerfile, not a docker-compose service. Use `docker build --target audit`, not `docker compose build audit`.

```bash
# Correct: Build audit target (runs all tests)
docker build --target audit -t llmrouter-audit .

# Wrong: docker compose (will fail - no audit service)
docker compose build audit  # FAILS: audit is not a compose service

# Wrong: Running on host (will fail)
python3 -m pytest tests/ -v  # FAILS: headroom not installed
```

## Quick Commands

```bash
# Build audit image (runs all tests during build)
docker build --target audit -t llmrouter-audit .

# Run specific marker in container (after building)
docker run --rm llmrouter-audit python3 -m pytest tests/ -m base -v

# Run specific file in container (after building)
docker run --rm llmrouter-audit python3 -m pytest tests/test_tokenizer.py -v
```

## Test Markers

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

## Files

| File | Purpose |
|------|---------|
| [`TESTING.md`](../../TESTING.md) | Full testing documentation |
| [`tests/conftest.py`](../../tests/conftest.py) | Shared fixtures |
| [`pytest.ini`](../../pytest.ini) | Pytest configuration |

## Troubleshooting

- **"headroom library is not installed"**: Expected on host. Use Docker.
- **Unknown marker warning**: Check [`pytest.ini`](../../pytest.ini) for registered markers.
- **Import errors**: Verify dependencies in [`requirements.txt`](../../requirements.txt).
