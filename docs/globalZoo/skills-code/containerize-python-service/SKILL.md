---
name: containerize-python-service
description: Package a Python service with pyproject-based packaging, an installable entry point, tests, and a multi-stage Docker image.
---

# Goal

Create a Python service that installs cleanly and runs cleanly inside Docker.

# Instructions

1. Standardize the Python project first:
   - `pyproject.toml`
   - `requires-python`
   - `[project.scripts]`
   - `src/` layout for new projects

2. Ensure the service has:
   - a clear entrypoint
   - configuration through environment or CLI arguments
   - logging to standard streams
   - tests for startup and core behavior

3. Build the Docker image with:
   - a build stage
   - a runtime stage
   - `.dockerignore`
   - absolute `WORKDIR`
   - non-root `USER` when possible
   - exec-form `CMD`

4. Keep runtime images free of unnecessary build tooling.
5. Add a `HEALTHCHECK` when the service exposes a meaningful readiness endpoint or command.

# Deliverables

- `pyproject.toml`
- Python entrypoint
- Dockerfile
- `.dockerignore`
- optional `compose.yaml` for local development

# Acceptance checklist

- service can run as an installed Python command
- image builds reproducibly
- container starts with the intended entrypoint
- logs are visible on stdout/stderr
- tests, linting, and type checking remain runnable
