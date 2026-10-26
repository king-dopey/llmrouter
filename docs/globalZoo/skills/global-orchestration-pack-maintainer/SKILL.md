---
name: global-orchestration-pack-maintainer
description: Use this skill to maintain the reusable global orchestration pack (modes, rules, and governance contracts) while keeping repository-specific behavior out of global definitions.
modeSlugs:
  - architect
  - reviewer
  - documentation-writer
---

# Global Orchestration Pack Maintainer

## Purpose

Maintain a portable orchestration pack that can be reused across repositories without leaking stack-specific assumptions.

## Scope

- Global mode contracts and handoff conventions.
- Shared cross-mode rules in `globalZoo/rules/`.
- Packaging and versioned release notes for global imports.

## Guardrails

- Keep global artifacts generic and reusable.
- Put stack- or repo-specific behavior in project-scoped `.roo/rules-{slug}/` files.
- Use MCP tools for global agentic governance behavior.

## Checklist

1. Validate required global rule files exist.
2. Confirm handoff schema fields and blocker taxonomy are consistent.
3. Verify no repository paths, hostnames, or service-specific assumptions appear in global mode definitions.
4. Publish concise change notes and expected migration impact.

## Outputs

- Updated global mode/rule artifacts.
- Compatibility notes for importers.
- Risk list for workflow behavior changes.
