# Global Handoff Contract

Every mode handoff artifact must include this payload shape:

- `status`: `PASS` | `FAIL` | `BLOCKED` | `ERROR`
- `blocker_tag`: `missing-input` | `tool-failure` | `env-failure` | `scope-conflict` | `policy-denied` | `approval-required` | `schema-invalid`
- `transition_basis`: mode-specific factual transition basis and routing evidence
- `recommended_next_action`: orchestrator-selected next action or parent return; this is advisory metadata, not an authoritative routing field
- `evidence_ref`: pointer to logs, report, or command output
- `operator_action`: one concrete next action

## Contract Rules

- If `status` is `BLOCKED` or `ERROR`, `blocker_tag` is required.
- The orchestrator decides the next child task; the packet does not encode an authoritative `next_mode` routing decision.
- `transition_basis` must be present whenever a mode has more than one valid next transition.
- `recommended_next_action` must summarize the orchestrator’s likely next step or indicate a parent return when the work is complete.
- `operator_action` must be imperative and single-step.
- Only one blocker tag is allowed per handoff.
