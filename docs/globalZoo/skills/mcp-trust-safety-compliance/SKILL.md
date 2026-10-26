---
name: mcp-trust-safety-compliance
description: Use this skill to enforce MCP trust/safety controls: consent before side effects, least privilege, transparent tool intent, and audit-ready evidence.
modeSlugs:
  - architect
  - reviewer
  - security-review
  - documentation-writer
---

# MCP Trust Safety Compliance

## Purpose

Apply trust-and-safety guardrails to MCP tool usage across orchestration workflows.

## Control Objectives

- Explicit consent before side-effecting actions.
- Transparent tool intent before invocation.
- Least-privilege tool selection.
- Auditability of decisions and outcomes.

## Review Checklist

1. Confirm side-effecting actions were consent-gated.
2. Confirm risky actions used policy checks before execution.
3. Confirm logs/artifacts are redacted before sharing.
4. Confirm blockers are classified with shared taxonomy.

## Guardrails

- No hidden retries for privileged actions.
- No secret disclosure in evidence packets.
- Global governance behavior must be MCP-server based.

## Outputs

- Compliance findings with severity.
- Remediation instructions.
- Evidence references for audit traceability.
