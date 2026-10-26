# Global Blocker Taxonomy

Use exactly one blocker tag when blocked or in error state.

## Allowed Tags

- `missing-input`: required user/context input is missing.
- `tool-failure`: required tool invocation failed.
- `env-failure`: environment/runtime dependency unavailable.
- `scope-conflict`: request conflicts with hard constraints.
- `policy-denied`: action denied by safety or policy guard.
- `approval-required`: gated action awaits explicit approval.
- `schema-invalid`: payload or config shape failed validation.

## Classification Guidance

- Prefer the nearest actionable cause.
- Do not stack multiple tags.
- If cause is uncertain, use `env-failure` only when an external dependency is clearly implicated; otherwise use `tool-failure`.
