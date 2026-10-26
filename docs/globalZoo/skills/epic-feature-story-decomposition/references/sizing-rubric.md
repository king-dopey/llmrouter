# Sizing Rubric & Worked Examples

Goal: pick the **lowest hierarchy level** that fully satisfies the ask.

## Decision Table

| Signal in the ask | Likely level | Why |
| :--- | :--- | :--- |
| Single behavior change, one module, < 1 sprint | **Story** | Smallest unit; independently valuable & testable |
| One customer-facing capability, several related stories, 1 team | **Feature** | Groups stories into one deliverable capability |
| Multiple capabilities / teams / systems, multi-sprint, strategic | **Epic** (possibly several) | Cross-cutting initiative needing decomposition |
| Whole new product / platform | **Multiple Epics** | Each major domain becomes its own Epic |

## Worked Examples

### Example A — "Create a new eCommerce web site for a new business"
This is a **whole product** → **multiple Epics**, each with Features, each with Stories.

```
EPIC-01  Catalog & Product Management
  FEAT-01-01  Product listing & detail pages
    STORY-01-01-01  As a shopper, I want to browse products by category, so I can find what I need.
    STORY-01-01-02  As a shopper, I want to view a product detail page with price/stock, so I can decide to buy.
  FEAT-01-02  Product search & filters
    STORY-01-02-01  As a shopper, I want to search and filter products, so I can narrow results quickly.
EPIC-02  Cart & Checkout
  FEAT-02-01  Shopping cart
    STORY-02-01-01  As a shopper, I want to add/remove items in a cart, so I can review my order.
  FEAT-02-02  Payment & order confirmation
    STORY-02-02-01  As a shopper, I want to pay securely and get an order confirmation, so my purchase completes.
EPIC-03  Accounts & Orders
  FEAT-03-01  Customer accounts
    STORY-03-01-01  As a customer, I want to register and log in, so I can track my orders.
  FEAT-03-02  Order history
    STORY-03-02-01  As a customer, I want to view past orders, so I can re-order or check status.
EPIC-04  Admin & Operations (inventory, shipping, analytics) ...
```

> Note: this is a *decomposition*, not an implementation plan. Each Story is then
> handed (one at a time) to the Architect for its own `impl_plan.xml`.

### Example B — "Add a new source to the existing data extract"
This is **one capability, one module, < 1 sprint** → **a single Story** (or a tiny Feature).

```
STORY-00-00-01  As a data analyst, I want the nightly extract to include the
                "payments" source, so the warehouse reflects payment events.
  AC1: Given the extract job runs, When the payments source is present,
       Then the output partition contains payments rows with schema X.
  AC2: Given the payments source is down, When the job runs,
       Then the job logs a warning and completes with the other sources.
  Points: 3   Dependencies: none
```

### Example C — "Add a dark mode to the settings page"
**One capability, one team** → **Feature** with 2–3 Stories.

```
FEAT-01-01  Dark Mode
  STORY-01-01-01  As a user, I want a light/dark toggle in settings, so I can reduce eye strain.
  STORY-01-01-02  As a user, I want my theme to persist across sessions, so I don't reset it each visit.
  STORY-01-01-03  As a user, I want the theme to follow my OS preference by default, so it feels native.
```

## Vertical Slices Rule
Prefer stories that deliver an **end-to-end thin slice** (data + API + UI for one
capability) over horizontal layers. A "vertical slice" story is independently
demonstrable and testable; a horizontal layer ("build all the DB tables") is not.
