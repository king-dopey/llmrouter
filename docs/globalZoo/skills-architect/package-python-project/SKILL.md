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
