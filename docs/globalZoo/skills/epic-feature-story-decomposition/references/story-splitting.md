# Story Splitting & INVEST Deep-Dive

A story that fails INVEST must be split or reworded before it is emitted.

## INVEST, in practice

| Criterion | Failing smell | Fix |
| :--- | :--- | :--- |
| **I**ndependent | Story A can't ship until B's internals exist | Re-scope into vertical slices; expose a stable interface |
| **N**egotiable | Story dictates exact UI pixels / tech | Describe the outcome, leave design to implementation |
| **V**aluable | "Refactor module X" with no user benefit | Attach the user/stakeholder value or mark as technical debt with rationale |
| **E**stimable | "Make it faster" (no target) | Add a measurable target (e.g., p95 < 200 ms) |
| **S**mall | Spans multiple sprints | Split (techniques below) |
| **T**estable | "Looks good" | Write Given-When-Then with concrete values |

## Splitting Techniques

Pick the technique that matches the *reason* the story is too big.

1. **By user workflow step** — split the journey into sequential steps.
   - "Complete checkout" → "Add to cart", "Enter shipping", "Pay", "Confirm".
2. **By data variation** — one story per data type/shape.
   - "Import contacts" → "Import CSV", "Import vCard", "Import from CRM API".
3. **By business rule / variation** — one story per rule or variant.
   - "Apply discounts" → "Percentage discount", "Fixed-amount discount", "Buy-X-get-Y".
4. **By I/O channel** — one story per input/output path.
   - "Notify user" → "Email notification", "In-app banner", "SMS fallback".
5. **By role / persona** — one story per distinct user type when behavior differs.
   - "Manage orders" → "Customer views order", "Admin cancels order".
6. **Happy path first, then edges** — deliver the core flow, then error/edge handling.
   - "Login" (happy) → "Login with invalid credentials" → "Login lockout after N failures".
7. **Placeholder / stub** — ship a working thin version, replace the stub in a later story.
   - "Payment: card only" → "Payment: add PayPal".

## Acceptance Criteria (Given-When-Then)
Every story needs **at least two** scenarios: one happy path and at least one
edge/error path. Keep them concrete and value-bound.

```
AC1 (happy):  Given [context], When [action], Then [measurable outcome].
AC2 (edge):   Given [boundary/bad input], When [action], Then [graceful behavior].
```

Example:
```
STORY: As a shopper, I want to apply a promo code at checkout, so I can get a discount.
AC1: Given a valid code "SAVE10" in the cart, When I click Apply, Then the total
     decreases by 10% and a confirmation shows.
AC2: Given an expired code, When I click Apply, Then an error "Code expired" shows
     and the total is unchanged.
```

## Sizing (story points)
Use relative Fibonacci (1,2,3,5,8,13) or S/M/L. If a story estimates > 8 points,
it is too big — apply a splitting technique. Points are relative within the team,
not absolute hours.
