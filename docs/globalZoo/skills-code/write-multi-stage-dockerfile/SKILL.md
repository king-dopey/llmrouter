---
name: write-multi-stage-dockerfile
description: Create or refactor a Dockerfile into a cache-friendly multi-stage build with a smaller runtime image.
---

# Goal

Produce a Dockerfile that is efficient to build and leaner to run.

# Instructions

1. Identify:
   - build dependencies
   - runtime dependencies
   - final process command
   - required ports and health probe path if any

2. Create separate stages for:
   - base/shared setup if useful
   - build
   - runtime

3. Order layers to maximize cache reuse:
   - base image
   - dependency metadata
   - dependency install
   - source copy
   - build step

4. Use:
   - `.dockerignore`
   - absolute `WORKDIR`
   - exec-form `CMD` or `ENTRYPOINT`
   - `USER` for non-root execution when possible

5. Add `HEALTHCHECK` when service health matters.

# Deliverables

- Dockerfile
- `.dockerignore`
- short build/run examples
- notes on cache behavior and runtime assumptions

# Acceptance checklist

- final image excludes unnecessary build tooling
- Dockerfile is readable and stage names are clear
- command form is correct for signal handling
- context is minimized
