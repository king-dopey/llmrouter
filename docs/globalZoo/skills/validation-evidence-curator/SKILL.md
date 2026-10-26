---
name: validation-evidence-curator
description: Use this skill to enforce strong, reproducible evidence capture in tester, reviewer, security-review, and debug artifacts.
modeSlugs:
  - tester
  - reviewer
  - security-review
  - debug
---

# Validation Evidence Curator

## Purpose

Raise the quality of validation and audit evidence so downstream decisions are reproducible and machine-reviewable.

## Checklist

1. Confirm every failing check includes command or tool, exit status, and key output lines.
2. Confirm every serious finding includes exact file and line or a concrete runtime artifact.
3. Confirm missing evidence is classified as `missing-input` rather than replaced with speculation.
4. Confirm evidence references are short, concrete, and reusable by the next mode.

## Outputs

- Evidence sufficiency findings.
- Missing artifact list.
- Suggested evidence normalization changes.