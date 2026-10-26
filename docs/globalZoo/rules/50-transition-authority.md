# Global Transition Authority

Task transitions have a single authority chain.

## Rules

- The orchestrator is the authority for the next child task, based on the completed packet and current workflow state.
- `new_task` is the only mechanism for mode transitions in this workflow. `switch_mode` is not used.
- The handoff packet is result metadata; it does not encode an authoritative `next_mode` routing decision.
- Modes must not guess, override, or pre-commit the next task before the orchestrator chooses it.
- Validation tools never execute transitions themselves.

## The code → debug → code loop

All mode transitions use `new_task` and `attempt_completion` — including the debug diagnosis loop:

1. Code detects a failure it cannot resolve.
2. Code writes the intermediate handoff packet (FAIL state) to the ordered story path.
3. Code calls `new_task(debug)` with the failure context and evidence.
4. Debug diagnoses. Debug calls `attempt_completion` with the diagnosis packet — it does NOT call `new_task(code)`.
5. The code parent receives the diagnosis and calls `new_task(code)` to implement the confirmed fix.
6. The fix code task validates and calls `attempt_completion` back to the code parent with PASS.
7. The code parent proceeds with `new_task(tester)` or `attempt_completion`.

This pattern ensures every transition produces a written artifact, so the session can be resumed from disk after a crash.
