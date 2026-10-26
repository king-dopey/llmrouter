# Global Code Diff Safety Rule

The `apply_diff` tool uses `=======` (seven equals signs) as the separator between its SEARCH and REPLACE blocks, and `<<<<<<<` / `>>>>>>>` as the open/close markers. If any of these strings appear inside the SEARCH content, the tool parser misreads them as conflict markers and the diff aborts.

## Required check before every apply_diff call

Before constructing an `apply_diff` call, scan the intended SEARCH content for these exact strings:
- `=======` (seven equals signs)
- `<<<<<<<` (seven less-than signs)
- `>>>>>>>` (seven greater-than signs)

## If any marker string is found: use write_to_file

1. Read the current file content using `codebase_search` or `read_file`.
2. Perform the targeted replacement in the in-memory content.
3. Write the full updated content back with `write_to_file`.

Do not attempt `apply_diff` when the SEARCH block contains a conflict-marker string. The `write_to_file` fallback is not a workaround — it is the correct tool for this case.

## Common file types that trigger this

- YAML files with `=` dividers in multiline strings or comments
- Markdown files with `===` heading underlines or horizontal rules
- Mermaid diagram source with `===` style tokens
- Shell scripts with `=====` separators in output
- Any file that was previously involved in a merge conflict and still contains unresolved markers
