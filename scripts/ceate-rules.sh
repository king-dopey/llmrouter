#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: generate-zoo-usecase-rules.sh [--root DIR] [--include LIST] [-h|--help]

Create .roo rules and skills for selected use cases.

Options:
  --root DIR       Project root (default: current directory)
  --include LIST   Comma-separated use cases: bash,python,docker
                   Default: bash,python,docker
  -h, --help       Show this help and exit
EOF
}

err() {
  printf '%s\n' "$*" >&2
}

ROOT="."
INCLUDE_LIST="bash,python,docker"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { err "Missing value for --root"; exit 2; }
      ROOT="$2"
      shift 2
      ;;
    --include)
      [[ $# -ge 2 ]] || { err "Missing value for --include"; exit 2; }
      INCLUDE_LIST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "Unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

declare -A SELECTED=()
parse_include_list() {
  local raw="$1"
  local item
  IFS=',' read -r -a items <<< "$raw"
  for item in "${items[@]}"; do
    item="${item,,}"
    item="${item//[[:space:]]/}"
    [[ -n "$item" ]] || continue
    case "$item" in
      bash|python|docker)
        SELECTED["$item"]=1
        ;;
      *)
        err "Unknown use case: $item"
        exit 2
        ;;
    esac
  done
}

parse_include_list "$INCLUDE_LIST"

has_usecase() {
  [[ -n "${SELECTED[$1]:-}" ]]
}

mkdir -p "$ROOT/.roo"

declare -a WRITTEN_FILES=()

write_file() {
  local path="$1"
  mkdir -p "$(dirname "$path")"
  cat > "$path"
  WRITTEN_FILES+=("$path")
  printf 'Wrote %s\n' "${path#$ROOT/}"
}

write_common_files() {
  write_file "$ROOT/.roo/rules/10-rules-vs-skills-policy.md" <<'EOF'
# Rules vs Skills Policy

## Rules

- Keep rules short, stable, and always-on.
- Put only constraints here that should apply on nearly every task in the matching mode.
- Prefer one topic per file.
- Prefer prescriptive bullets over long explanation.
- Do not put large templates, walkthroughs, or scaffolds in rules files.

## Skills

- Use skills for long, task-specific workflows.
- Use skills for scaffolds, migrations, debugging playbooks, and reusable checklists.
- Keep skill names specific and reusable.
- Put the detailed "how" in skills and the always-on "must" in rules.

## Slash commands

- Reserve slash commands for explicitly-invoked operator workflows.
- Do not use slash commands to hold baseline coding standards.

## File organization

- Use generic `.roo/rules/` only for cross-cutting policy.
- Put mode-specific constraints in `.roo/rules-architect/`, `.roo/rules-code/`, `.roo/rules-tester/`, and `.roo/rules-debug/`.
- Use `.roo/skills/` or `.roo/skills-<mode>/` for detailed workflows.
EOF

  write_file "$ROOT/.roo/rules/20-general-quality-gates.md" <<'EOF'
# General Quality Gates

## Rules

- Prefer small, reviewable edits.
- Update or add tests when behavior changes.
- Keep linting, tests, and build checks in CI for the selected stacks.
- Do not introduce a new tool unless it simplifies the stack or closes a real quality gap.
- Prefer standard library features before adding dependencies for small problems.
- Keep generated configuration committed when the team is expected to share it.
- Keep examples aligned with the actual project structure and commands.
EOF
}

write_bash_files() {
  write_file "$ROOT/.roo/rules-architect/10-bash-architecture.md" <<'EOF'
# Bash Architecture Rules

## Rules

- Use Bash for small utilities, wrappers, automation glue, and orchestration.
- Prefer Python over Bash for complex parsing, rich data structures, or large business logic.
- Require every Bash script to have one clear purpose.
- Require a documented CLI contract before implementation.
- Require support for `-h` and `--help`.
- Recommend support for `--version`.
- Require separation of parsing, validation, execution, output, and cleanup for non-trivial scripts.
- Require normal output on stdout and errors on stderr.
EOF

  write_file "$ROOT/.roo/rules-code/10-bash-implementation.md" <<'EOF'
# Bash Implementation Rules

## Rules

- Use an explicit Bash shebang for Bash scripts.
- Use `set -euo pipefail` for non-trivial scripts unless there is a documented reason not to.
- Quote variable expansions unless unquoted behavior is intentional and safe.
- Use `"$@"` instead of `$*`.
- Use `$(...)` instead of backticks.
- Prefer `[[ ... ]]` over `[ ... ]` in Bash scripts.
- Use functions for non-trivial scripts and end with `main "$@"`.
- Use `local` for function variables and `readonly` for constants.
- Support `-h` and `--help`.
- Reject unknown options and missing option values.
- Use `getopts` for short-option portable scripts or a `while` plus `case` parser when long options are needed.
- Use `mktemp` for temporary resources and clean them up with `trap`.
- Keep scripts `shellcheck`-clean and `shfmt`-formatted.
- Avoid `eval` unless there is no safe alternative and the reason is documented.
EOF

  write_file "$ROOT/.roo/rules-tester/10-bash-testing.md" <<'EOF'
# Bash Testing Rules

## Rules

- Verify `-h` and `--help` exit `0` and write help to stdout.
- Verify invalid usage exits non-zero and writes errors to stderr.
- Verify unknown flags are rejected.
- Verify missing option values are rejected.
- Verify required positional arguments are enforced.
- Verify `--` ends option parsing when supported.
- Test arguments containing spaces, quotes, glob characters, and leading dashes.
- Verify temp-file creation and cleanup behavior.
- Run `shellcheck` and fail validation on unresolved issues.
- Run `shfmt --diff` or equivalent formatting validation.
EOF

  write_file "$ROOT/.roo/rules-debug/10-bash-debugging.md" <<'EOF'
# Bash Debugging Rules

## Rules

- Reproduce the issue with the exact failing command first.
- Capture stdout, stderr, and exit code separately.
- Check quoting, word splitting, globbing, and unset variables first.
- Verify parser branches shift the correct number of arguments.
- Verify `set -e`, `set -u`, and `pipefail` interactions before changing logic.
- Add temporary debug output only to stderr.
- Use `set -x` only when needed and remove it before completion.
- Verify traps and cleanup do not hide the original failure.
- Add or update a regression test after fixing a bug.
EOF

  write_file "$ROOT/.roo/skills-code/create-bash-cli-scaffold/SKILL.md" <<'EOF'
---
name: create-bash-cli-scaffold
description: Create or update a Bash CLI with standard help output, safe argument parsing, cleanup handling, and shellcheck-friendly structure.
---

# Goal

Create a small Bash command-line tool with a predictable Unix-style interface.

# Use this skill when

- creating a new Bash utility
- adding CLI parsing to an existing shell script
- standardizing help, usage, and exit-code behavior
- refactoring an ad hoc script into a maintainable Bash tool

# Instructions

1. First decide whether Bash is still the right language.
   - If the task is mostly orchestration and command execution, proceed with Bash.
   - If the task needs rich data structures or complex parsing, recommend Python instead.

2. Build the script around these sections:
   - constants
   - stderr helper
   - usage/help functions
   - cleanup function
   - parse_args
   - validation
   - main
   - `main "$@"`

3. Implement the CLI contract:
   - support `-h` and `--help`
   - support `--version` when appropriate
   - reject unknown options
   - reject missing option values
   - print help to stdout and usage errors to stderr
   - use non-zero exit status for invalid usage

4. Apply Bash safety defaults:
   - explicit Bash shebang
   - `set -euo pipefail`
   - quoted expansions
   - `"$@"`
   - `$(...)`
   - `[[ ... ]]`
   - `local` variables
   - `readonly` constants

5. If temporary resources are created:
   - use `mktemp`
   - register cleanup with `trap`

6. Keep output disciplined:
   - stdout for intended results
   - stderr for diagnostics and errors

7. Produce code that is ready for:
   - `shellcheck`
   - `shfmt`

# Deliverables

- a complete Bash script
- help and usage text
- argument parser
- cleanup handling if needed
- one or two example invocations

# Acceptance checklist

- script has standard help behavior
- parser fails clearly on bad input
- quoting is safe
- no unnecessary globals
- structure is easy to lint and test
EOF

  write_file "$ROOT/.roo/skills-debug/debug-bash-script/SKILL.md" <<'EOF'
---
name: debug-bash-script
description: Debug a Bash script systematically by reproducing the failure, isolating parser and quoting issues, and adding a regression test.
---

# Goal

Diagnose and fix a Bash failure without masking the original problem.

# Instructions

1. Reproduce the issue exactly.
2. Capture:
   - command
   - inputs
   - stdout
   - stderr
   - exit code
3. Check the parser first:
   - option matching
   - shift counts
   - `--` handling
   - required argument validation
4. Check shell semantics next:
   - quoting
   - word splitting
   - globbing
   - unset variables
   - pipeline failure behavior
5. If needed, add narrow tracing or stderr debug output.
6. Review trap and cleanup behavior for side effects.
7. Fix the smallest real cause.
8. Add a regression test and re-run linting.

# Output expectations

- root cause
- minimal fix
- regression coverage
- confirmation that help, stderr/stdout, and exit codes still behave correctly
EOF
}

write_python_files() {
  write_file "$ROOT/.roo/rules-architect/20-python-architecture.md" <<'EOF'
# Python Architecture Rules

## Rules

- Use `pyproject.toml` as the primary project configuration file.
- For new projects, prefer a `src/` layout for import isolation.
- Separate library logic from CLI entrypoints.
- Declare `requires-python`.
- Expose installable CLIs through `[project.scripts]`.
- Prefer standard-library solutions before adding dependencies for small problems.
- Keep modules focused and avoid mixing CLI, domain logic, and I/O in one file.
EOF

  write_file "$ROOT/.roo/rules-code/20-python-implementation.md" <<'EOF'
# Python Implementation Rules

## Rules

- Follow project-consistent PEP 8 style.
- Add type hints for new and changed public functions and important internal boundaries.
- Prefer `pathlib.Path` for filesystem paths in new code.
- Use `argparse` for standard-library CLIs.
- Use `logging.getLogger(__name__)` for logging.
- Configure logging in the application entrypoint, not in import-time library code.
- Use `print()` only for intentional CLI user-facing output.
- Group imports as standard library, third-party, then local imports.
- Catch specific exceptions; do not use broad `except:` unless re-raising after cleanup or logging.
- Keep `pyproject.toml` as the source of truth for build metadata and tool configuration.
- Keep Ruff and a type checker configured and passing.
EOF

  write_file "$ROOT/.roo/rules-tester/20-python-testing.md" <<'EOF'
# Python Testing Rules

## Rules

- Use `pytest` for tests unless the project already standardizes on another framework.
- For new packages, prefer `src/` layout with tests outside the package.
- Configure `testpaths` when the project structure benefits from explicit discovery.
- Prefer `--import-mode=importlib` for new pytest-based projects.
- Test public behavior before internal implementation details.
- Add tests for CLI help, invalid arguments, and exit codes when a CLI exists.
- Use fixtures for setup reuse, but keep fixtures small and readable.
- Reproduce every confirmed bug with a regression test.
- Run linting, tests, and type checking together in CI.
EOF

  write_file "$ROOT/.roo/rules-debug/20-python-debugging.md" <<'EOF'
# Python Debugging Rules

## Rules

- Reproduce the failure with the exact command, environment, and inputs first.
- Preserve the traceback and do not replace exceptions prematurely.
- Check import-path and packaging issues before changing application logic.
- Confirm whether the failure is in domain logic, CLI parsing, filesystem handling, or environment configuration.
- Narrow broad exception handling during debugging.
- Verify logging and user-visible error behavior after the fix.
- Re-run tests, linting, and type checking after every fix.
- Add a regression test before closing the bug.
EOF

  write_file "$ROOT/.roo/skills-code/create-python-cli/SKILL.md" <<'EOF'
---
name: create-python-cli
description: Create or update a Python CLI using pyproject entry points, argparse, pathlib, logging, typing, pytest, Ruff, and type checking.
---

# Goal

Build a package-installable Python CLI that is easy to run, test, lint, and type-check.

# Instructions

1. Use `pyproject.toml` with:
   - `[build-system]`
   - `[project]`
   - `requires-python`
   - `[project.scripts]`

2. For a new project, prefer:
   - `src/<package_name>/`
   - `tests/`

3. Structure the CLI so that:
   - parser creation is separate from execution
   - `main()` returns an exit code
   - application logic is not buried inside argument parsing

4. Use:
   - `argparse` for CLI parsing and help
   - `pathlib.Path` for filesystem work
   - `logging.getLogger(__name__)` for logging
   - type hints on public and boundary functions

5. Add tests for:
   - normal behavior
   - help output
   - invalid arguments
   - key edge cases

6. Add project tooling configuration for:
   - Ruff
   - pytest
   - a type checker

# Deliverables

- `pyproject.toml`
- `src/` package layout
- CLI entrypoint
- tests
- minimal README usage section

# Acceptance checklist

- CLI can be invoked as an installed command
- help output is useful
- parser and business logic are separate
- paths use `pathlib`
- tests, linting, and type checks are ready to run
EOF

  write_file "$ROOT/.roo/skills-architect/package-python-project/SKILL.md" <<'EOF'
---
name: package-python-project
description: Design or reorganize a Python project around pyproject.toml, src layout, entry points, tests, linting, and type checking.
---

# Goal

Produce a maintainable Python project structure suitable for team development and CI.

# Instructions

1. Standardize project layout around:
   - `pyproject.toml`
   - `src/`
   - `tests/`

2. Put build metadata and tool configuration in `pyproject.toml` where supported.
3. Declare `requires-python`.
4. Use `[project.scripts]` for installed commands.
5. Keep package code separate from tests and local tooling files.
6. Recommend:
   - Ruff for linting/formatting
   - pytest for tests
   - a type checker for static validation
7. If migrating an existing project:
   - preserve import stability where practical
   - update tests and CI along with the layout
   - verify editable install and test discovery

# Output expectations

- target folder structure
- `pyproject.toml` plan
- migration notes if restructuring existing code
- lint/test/type-check plan
EOF
}

write_docker_files() {
  write_file "$ROOT/.roo/rules-architect/30-docker-architecture.md" <<'EOF'
# Docker Architecture Rules

## Rules

- Use Docker to package runnable artifacts and consistent environments.
- Use Compose for local multi-service development and orchestration.
- Prefer one focused service definition per workload.
- Separate build-time and runtime concerns.
- Keep runtime images smaller and simpler than build images.
- Prefer modern Compose-spec files and avoid the obsolete top-level `version` key.
EOF

  write_file "$ROOT/.roo/rules-code/30-docker-implementation.md" <<'EOF'
# Docker Implementation Rules

## Rules

- Use multi-stage builds for production images when build tooling is not needed at runtime.
- Prefer small, trusted, pinned base images.
- Use `.dockerignore`.
- Order Dockerfile instructions to preserve build cache.
- Use absolute paths with `WORKDIR`.
- Prefer exec-form `CMD` and `ENTRYPOINT`.
- Run as a non-root user with `USER` when the workload allows it.
- Add `HEALTHCHECK` when container health matters to operations.
- Keep Dockerfiles readable and stage names meaningful.
- Use Compose-spec files with `services:` and omit the obsolete top-level `version` field.
EOF

  write_file "$ROOT/.roo/rules-tester/30-docker-testing.md" <<'EOF'
# Docker Testing Rules

## Rules

- Build every image in CI.
- Run smoke tests against the built image, not just source-level tests.
- Verify the container starts with the expected command.
- Verify non-root execution where configured.
- Verify `.dockerignore` and `COPY` behavior when context-sensitive files matter.
- Verify health checks for services that declare them.
- Verify Compose configurations start the expected services with the expected ports and environment behavior.
EOF

  write_file "$ROOT/.roo/rules-debug/30-docker-debugging.md" <<'EOF'
# Docker Debugging Rules

## Rules

- Reproduce the failure with the exact build or run command first.
- Separate build failures from runtime failures before changing code.
- Check build context, `.dockerignore`, and `COPY` paths before deeper changes.
- Check cache invalidation and layer ordering when builds become unexpectedly slow.
- Check user, file permissions, working directory, and entrypoint behavior for runtime issues.
- Check signal handling and process startup behavior when containers do not stop cleanly.
- Check healthcheck behavior separately from startup success.
- Rebuild and re-test after every Dockerfile or Compose fix.
EOF

  write_file "$ROOT/.roo/skills-code/write-multi-stage-dockerfile/SKILL.md" <<'EOF'
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
EOF

  write_file "$ROOT/.roo/skills-debug/debug-container-runtime/SKILL.md" <<'EOF'
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
EOF
}

write_python_docker_cross_skill() {
  write_file "$ROOT/.roo/skills-code/containerize-python-service/SKILL.md" <<'EOF'
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
EOF
}

write_common_files

if has_usecase bash; then
  write_bash_files
fi

if has_usecase python; then
  write_python_files
fi

if has_usecase docker; then
  write_docker_files
fi

if has_usecase python && has_usecase docker; then
  write_python_docker_cross_skill
fi

printf '\nGenerated structure:\n'
printf '.roo/\n'
printf '%s\n' "${WRITTEN_FILES[@]}" \
  | sed "s#^$ROOT/.roo/#  #g" \
  | sed "s#^$ROOT/##g" \
  | sort

printf '\nDone.\n'
