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
