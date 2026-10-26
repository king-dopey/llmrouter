---
name: global-rulepack-curator
description: Use this skill to maintain the separated global rules hierarchy and keep shared global rules distinct from repo-scoped rule folders.
modeSlugs:
  - architect
  - documentation-writer
  - skill-writer
---

# Global Rulepack Curator

## Purpose

Maintain the global rules hierarchy as a reusable pack and prevent global rules from being mixed with repository-specific rule folders.

## Directory Structure

```
globalZoo/          ← source for the user's global ~/.roo/ folder
  rules/            ← generic rules that apply to every mode in every project
  rules-{slug}/     ← generic mode-scoped rules for every project
  skills/           ← reusable skill playbooks
  prompts/          ← global prompt templates (e.g., enhance.md)

.roo/               ← workspace (project-specific) overrides for this repo
  mcp.json          ← MCP server config for this workspace
  rules-{slug}/     ← project-specific mode rules (e.g., phase2-*.md)
```

**Rule**: `globalZoo/rules/` must never duplicate files that also live in `.roo/rules/`, and `.roo/` must never duplicate content from `globalZoo/`. Shared generic rules belong only in `globalZoo/`; project-specific rules (referencing repo scripts or paths) belong only in `.roo/rules-{slug}/`. The workspace `.roo/rules/` directory should not exist unless a project-level override genuinely differs from the global version.

## Checklist

1. Confirm shared global rules live under `globalZoo/rules/`.
2. Confirm mode-scoped global rules live under `globalZoo/rules-{slug}/`.
3. Confirm project-specific mode rules remain under `.roo/rules-{slug}/`.
4. Confirm no file exists in both `globalZoo/rules/` and `.roo/rules/` — if one does, determine which location is authoritative and remove the duplicate.
5. Confirm `.roo/rules/` does not exist unless there is a genuine project-level override that differs from the global version.
6. Confirm docs and prompt contracts refer to the correct global rule locations.

## Outputs

- Rule hierarchy conformance notes.
- Misfiled rule findings.
- Update list for docs or prompt references.
