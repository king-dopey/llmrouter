# Global Documentation Diff Safety Rule

Documentation files frequently contain content that conflicts with the `apply_diff` tool's parser — including `===` heading underlines, Mermaid diagram syntax, code blocks containing conflict markers, and shell/YAML output with `=====` separators.

## Rule

Prefer `write_to_file` for all documentation changes. If `apply_diff` is used for a small targeted edit, scan the SEARCH content first.

## Required check before apply_diff

Before constructing any `apply_diff` call, check the SEARCH content for these exact strings:
- `=======` (seven equals signs) — the diff tool's SEARCH/REPLACE separator
- `<<<<<<<` (seven less-than signs) — the diff open marker
- `>>>>>>>` (seven greater-than signs) — the diff close marker

If any of these strings appear in the SEARCH content, do not use `apply_diff`. Use `write_to_file` instead:

1. Read the current file content.
2. Make the targeted replacement in the in-memory content.
3. Write the full updated content with `write_to_file`.

## Why documentation files are high risk

- Markdown headings underlined with `===` are valid syntax and trigger false positives.
- Mermaid diagrams embedded in `.md` files may contain `=` sequences.
- Code blocks showing merge conflict examples contain all three marker strings.
- YAML front matter and shell output blocks frequently contain `=====` dividers.
