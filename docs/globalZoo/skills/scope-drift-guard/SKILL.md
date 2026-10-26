---
name: scope-drift-guard
description: Use this skill to detect hidden task expansion, role bleed, and mode-scope violations in the reusable global workflow pack.
modeSlugs:
  - architect
  - code
  - reviewer
  - documentation-writer
  - debug
---

# Scope Drift Guard

## Purpose

Keep each mode inside its declared authority and prevent hidden expansion across planning, implementation, review, debugging, and documentation.

## Checklist

1. Confirm the current mode is producing only its required artifact.
2. Confirm no adjacent mode responsibilities have leaked into the current mode.
3. Confirm repository-specific runtime concerns are not being introduced into global artifacts.
4. Confirm recommended next actions stay within the workflow contract.

## Outputs

- Scope drift findings.
- Mode-boundary violations.
- Minimal corrections to restore role purity.