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
