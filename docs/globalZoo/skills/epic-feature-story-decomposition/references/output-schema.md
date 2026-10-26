# Output Schema — Decomposition Artifact (XML)

Emit **exactly one** `<decomposition>` document. No prose before/after it.

## Schema

```xml
<decomposition>
  <request_summary>[1-2 sentence restatement of the ask]</request_summary>

  <sizing_decision>STORY | FEATURE | EPIC | MULTIPLE_EPICS | UNCERTAIN</sizing_decision>
  <sizing_rationale>[why this level, citing the decision table signal]</sizing_rationale>

  <assumptions>
    <assumption>[scope assumption made due to ambiguity]</assumption>
  </assumptions>

  <!-- Omit levels not used. A STORY-only ask has no epic/feature wrappers. -->
  <epic id="EPIC-01" title="[Epic name]">
    <outcome>[business outcome this epic delivers]</outcome>
    <feature id="FEAT-01-01" title="[Feature name]">
      <story id="STORY-01-01-01">
        <title>As a [role], I want [capability], so that [value].</title>
        <acceptance_criteria>
          <scenario type="happy">Given ..., When ..., Then ...</scenario>
          <scenario type="edge">Given ..., When ..., Then ...</scenario>
        </acceptance_criteria>
        <points>3</points>
        <dependencies>none | STORY-xx-xx-xx, FEAT-xx-xx</dependencies>
        <invest>
          <independent>true</independent>
          <negotiable>true</negotiable>
          <valuable>true</valuable>
          <estimable>true</estimable>
          <small>true</small>
          <testable>true</testable>
        </invest>
      </story>
    </feature>
  </epic>

  <backlog>
    <!-- Every story ID in suggested delivery order (dependencies first). -->
    <order>STORY-01-01-01, STORY-01-01-02, STORY-02-01-01, ...</order>
  </backlog>

  <handoff>
    <status>PASS | BLOCKED | ERROR</status>
    <blocker_tag>missing-input | scope-conflict | none</blocker_tag>
    <transition_basis>complete</transition_basis>
    <next_mode>architect</next_mode>
    <selected_story>STORY-01-01-01</selected_story>
    <evidence_ref>decomposition artifact</evidence_ref>
    <operator_action>run Architect on selected_story to produce impl_plan.xml, then proceed down the backlog in order</operator_action>
  </handoff>
</decomposition>
```

## Rules
- `sizing_decision` must match the hierarchy actually emitted (a `STORY` decision
  emits a single `<story>` with no epic/feature wrappers).
- IDs are hierarchical and stable: `EPIC-nn / FEAT-nn-nn / STORY-nn-nn-nn`.
- Every story has ≥2 acceptance scenarios, a point estimate, dependencies, and a
  full `invest` block.
- `backlog.order` lists all stories dependency-first so the pipeline can execute
  them sequentially (one story → one Architect plan → code/test/review loop).
- The `<handoff>` node is **metadata only**; it does not perform the transition.
  The Tech Lead mode creates the next child task via `new_task` for the selected story and routes it to `architect`.

## Minimal Example (Story-only ask)
```xml
<decomposition>
  <request_summary>Add the "payments" source to the existing nightly data extract.</request_summary>
  <sizing_decision>STORY</sizing_decision>
  <sizing_rationale>One capability, one module, fits a single sprint — lowest sufficient level.</sizing_rationale>
  <assumptions><assumption>Existing extract framework and schema conventions are stable.</assumption></assumptions>
  <story id="STORY-00-00-01">
    <title>As a data analyst, I want the nightly extract to include the payments source, so the warehouse reflects payment events.</title>
    <acceptance_criteria>
      <scenario type="happy">Given the extract job runs and the payments source is present, When it completes, Then the output partition contains payments rows matching schema X.</scenario>
      <scenario type="edge">Given the payments source is unavailable, When the job runs, Then it logs a warning and completes with the other sources.</scenario>
    </acceptance_criteria>
    <points>3</points>
    <dependencies>none</dependencies>
    <invest><independent>true</independent><negotiable>true</negotiable><valuable>true</valuable><estimable>true</estimable><small>true</small><testable>true</testable></invest>
  </story>
  <backlog><order>STORY-00-00-01</order></backlog>
  <handoff>
    <status>PASS</status><blocker_tag>none</blocker_tag><transition_basis>complete</transition_basis>
    <next_mode>architect</next_mode><selected_story>STORY-00-00-01</selected_story>
    <evidence_ref>decomposition artifact</evidence_ref>
    <operator_action>run Architect on STORY-00-00-01 to produce impl_plan.xml</operator_action>
  </handoff>
</decomposition>
```
