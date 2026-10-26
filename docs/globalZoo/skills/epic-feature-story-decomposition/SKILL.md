---
name: epic-feature-story-decomposition
description: Decompose a user request into the correct Agile hierarchy (Epic / Feature / Story) sized to the ask, then emit sprint-ready, INVEST-compliant stories with Given-When-Then acceptance criteria. Use when deciding whether an ask is a Story, a Feature, or an Epic, or when breaking larger work into implementable units.
license: MIT
allowed-tools: read, command
metadata:
  category: planning
  consumes: user request, repository context, impl_plan.xml (optional)
  produces: decomposition artifact (XML) + story backlog
---

# Epic / Feature / Story Decomposition

Turn a free-form ask into a correctly-sized Agile hierarchy. The output must be
precise enough that a downstream **Architect** can produce an `impl_plan.xml` for
a single story without further clarification.

## When to Use
- A request is ambiguous in scope (is it a tweak, a feature, or a whole product?).
- You must decide whether to emit a Story, a Feature, or one/multiple Epics.
- You need to split large work into sprint-sized, independently deliverable stories.
- You need acceptance criteria in a testable (Given-When-Then) form.

Do **not** use for: writing implementation code, producing `impl_plan.xml` (that is
Architect's job), or running tests.

## The Hierarchy (SAFe-aligned)

| Level | Scope | Typical size | Deliverable |
| :--- | :--- | :--- | :--- |
| **Epic** | Large strategic initiative, cross-team, multi-sprint | Program/quarter | Epic statement + outcome + capability list |
| **Feature** | Customer-facing capability, one team, several stories | Sprint to release | Feature description + story set |
| **Story** | Smallest independently valuable, testable unit | Fits one sprint | Story + acceptance criteria |

Use the **lowest level that fully satisfies the ask**. Never inflate a Story into
an Epic. See `references/sizing-rubric.md` for the decision procedure and worked
examples (e.g., "new eCommerce site" → multiple Epics; "add a source to an
existing data extract" → one Story).

## Story Format (standard)

```
As a [role],
I want [capability / goal],
so that [measurable business value].
```

Every story must carry:
- **ID** — hierarchical, e.g. `EPIC-01 / FEAT-01-02 / STORY-01-02-03`
- **Acceptance criteria** — at least 2 Given-When-Then scenarios per story (happy path + one edge/error).
- **Story points** — relative estimate (Fibonacci 1,2,3,5,8,13) or a S/M/L label.
- **Dependencies** — IDs of stories/features that must land first.
- **INVEST check** — confirm Independent, Negotiable, Valuable, Estimable, Small, Testable.

## INVEST Quality Gate (mandatory per story)
| Criterion | Check |
| :--- | :--- |
| **I**ndependent | No hidden coupling; can be built/delivered alone |
| **N**egotiable | Describes *what/why*, not rigid *how* |
| **V**aluable | Delivers observable value to a user/stakeholder |
| **E**stimable | Team can size it with available context |
| **S**mall | Fits one sprint; split if not |
| **T**estable | Has concrete, verifiable acceptance criteria |

A story failing INVEST must be split or reworded before output. See
`references/story-splitting.md` for splitting techniques (by user workflow step,
data variation, rule/variation, I/O channel, business rule, placeholder).

## Sizing Decision Procedure
1. Estimate the number of distinct user-facing capabilities in the ask.
2. If it is **one capability** that fits a sprint → emit a **Story** (or small Feature with 1–3 stories).
3. If it is **several capabilities** for one team/release → emit a **Feature** with its stories.
4. If it spans **multiple teams, systems, or quarters** → emit **Epic(s)**, each decomposed into Features, each into Stories.
5. Prefer **vertical slices** (a thin end-to-end slice per story) over horizontal layers (all DB, then all API, then all UI).

## Output Contract
Produce exactly one **Decomposition Artifact** in XML (see `references/output-schema.md`).
- Root: `<decomposition>` with `<request_summary>`, `<sizing_decision>`, `<assumptions>`.
- One or more `<epic>` → `<feature>` → `<story>` nodes, each story carrying acceptance
  criteria, points, dependencies, and an `invest` checklist.
- A `<backlog>` table listing every story ID in suggested delivery order.
- A trailing `<handoff>` node (metadata only) with `next_mode: architect` and
  `transition_basis: complete`, so the Tech Lead mode can route the first/selected
  story to Architect for an `impl_plan.xml`.

## Hard Rules
- Single purpose: decompose and size. Do not write code or plans.
- Emit the lowest sufficient level; do not pad the hierarchy.
- Every story must be INVEST-compliant with ≥2 Given-When-Then criteria.
- No silent assumptions: state scope assumptions explicitly in `<assumptions>`.
- If the ask is too ambiguous to size, emit `sizing_decision = UNCERTAIN` with the
  specific missing inputs, rather than guessing.

## Progressive Disclosure
- `references/sizing-rubric.md` — decision table + worked examples (eCommerce vs. data-extract).
- `references/story-splitting.md` — INVEST deep-dive and splitting techniques.
- `references/output-schema.md` — full XML schema with a complete example.
