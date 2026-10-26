# Global Evidence Minimums

Apply these minimum evidence requirements in tester, reviewer, security-review, and debug outputs.

## Required Evidence

- For each failing check: include command or tool, exit status, and 1-5 key output lines.
- For each HIGH or CRITICAL finding: include exact file and line or a concrete runtime artifact.
- For policy denials: include the denied action and matching policy rule.

## Quality Requirements

- Evidence must be specific and reproducible.
- Avoid vague claims without artifact references.
- If evidence is unavailable, classify as `missing-input` and request the smallest missing artifact.
