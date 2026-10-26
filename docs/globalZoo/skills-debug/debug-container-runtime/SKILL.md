---
name: debug-container-runtime
description: Debug a failing or misbehaving container by separating build, startup, health, permission, and entrypoint issues.
---

# Goal

Find the real source of a container problem quickly and fix it without breaking image intent.

# Instructions

1. Determine whether the problem is:
   - build-time
   - startup-time
   - steady-state runtime
   - healthcheck-related
   - Compose wiring

2. Reproduce with the exact image and command.
3. Check:
   - entrypoint and command form
   - working directory
   - copied files
   - user and permissions
   - environment variables
   - exposed and mapped ports
   - healthcheck command
4. If the issue is build-related, inspect:
   - `.dockerignore`
   - context paths
   - cache invalidation
   - stage-to-stage copies
5. Validate the fix by rebuilding and re-running the container.

# Output expectations

- failure category
- root cause
- minimal Dockerfile or Compose fix
- validation steps
