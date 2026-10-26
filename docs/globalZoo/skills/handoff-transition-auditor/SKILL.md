---
name: handoff-transition-auditor
description: Use this skill to validate handoff payload quality, transition basis clarity, and authoritative next-mode routing across the global workflow pack.
modeSlugs:
  - architect
  - tester
  - reviewer
  - security-review
  - debug
---

# Handoff Transition Auditor

## Purpose

Audit whether a mode artifact contains enough structured information for authoritative handoff validation and safe transition.

## Checklist

1. Confirm `status`, `evidence_ref`, and `operator_action` are present.
2. Confirm `transition_basis` is present when multiple next transitions are possible.
3. Confirm `next_mode` is copied from the validator response rather than guessed manually.
4. Confirm blockers use exactly one allowed blocker tag.

## Outputs

- Handoff contract conformance notes.
- Missing-field or invalid-transition findings.
- Suggested minimal repair steps.