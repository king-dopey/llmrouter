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
