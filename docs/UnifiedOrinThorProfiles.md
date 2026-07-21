# Unified Orin + Thor Profile Runbook

## Shared baseline

- One shared compose/runtime path
- Ollama image: `ollama/ollama:0.30.10`
- LibreChat baseline: `0.8.6` (tracked in `librachat.json`)
- Profile selection: `PROFILE=orin|thor` (default `orin`)

## Profile files

- `profiles/orin/models.yaml`
- `profiles/thor/models.yaml`
- `profiles/orin/librechat-modelspecs.yaml`
- `profiles/thor/librechat-modelspecs.yaml`
- `profiles/orin/stack.env`
- `profiles/thor/stack.env`

Stable user-facing LibreChat preset names are:

- `Coding`
- `Chat`

Mapping by profile:

- Orin: `Coding -> qwen3-coder:30b`, `Chat -> qwen3.6:35b-a3b`
- Thor: `Coding -> qwen3-coder-next:q4_K_M`, `Chat -> qwen3.6:35b-a3b-q8_0`

## Start commands

### Orin

```bash
PROFILE=orin docker compose up -d
PROFILE=orin docker compose -f router/docker-compose.yml --profile proxy up -d --build
```

### Thor

```bash
PROFILE=thor docker compose up -d
PROFILE=thor docker compose -f router/docker-compose.yml --profile proxy up -d --build
```

## Validation commands

### Shared-stack checks

```bash
scripts/validation/validate-shared-stack.sh orin
scripts/validation/validate-shared-stack.sh thor
```

### Explicit model-tag validation (fails on unavailable tags)

```bash
PROFILE=orin scripts/validation/validate-model-tags.sh orin
PROFILE=thor scripts/validation/validate-model-tags.sh thor
```

### Runtime gate checks

```bash
PROFILE=orin scripts/validation/validate-runtime-gates.sh orin
PROFILE=thor scripts/validation/validate-runtime-gates.sh thor
```

### Manual health checks

```bash
docker compose ps
curl -fsS http://127.0.0.1:11434/api/tags | jq .
curl -fsS http://127.0.0.1:11434/api/ps | jq .
```

### Swap checks (host)

```bash
swapon --show
free -h
```

## Notes

- Orin default is preserved when `PROFILE` is unset.
- Runtime profile model policy is mounted into router at `/app/model_policy.yml` from `profiles/<profile>/models.yaml`.
- The repository does not include a first-party LibreChat compose file; it provides profile-resolved LibreChat modelspec files for external LibreChat deployment.

## ASR Service Configuration

The ASR service uses different Python versions per profile to match the Jetson wheel ABI:

| Profile | Python Base | Torch Index | Dockerfile |
|---------|-------------|-------------|------------|
| `orin` | Python 3.10 (cp310 wheels) | `jp6/cu126` | `asr/Dockerfile` |
| `thor` | Python 3.12 (cp312 wheels) | `jp7/cu126` | `asr/Dockerfile.thor` |

### Starting ASR by Profile

```bash
# Orin ASR (default)
PROFILE=orin docker compose --profile asr up -d

# Thor ASR
PROFILE=thor docker compose --profile asr up -d
```

### Build-time Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_DOCKERFILE` | `Dockerfile` | ASR Dockerfile to use (`Dockerfile` or `Dockerfile.thor`) |
| `ASR_TORCH_INDEX_URL` | `jp6/cu126` | Jetson wheel index (Orin: `jp6`, Thor: `jp7`) |

### Validation

```bash
# Validate Thor ASR configuration
./scripts/validation/validate-thor-asr.sh
```
