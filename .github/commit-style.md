# Commit Message Style

This repo uses **Conventional Commits**. Commit messages are part of the
change record for an operationally sensitive LAN service, so they should
make the *operational impact* obvious at a glance.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

- `feat`     — new user-visible capability (new route, new policy rule,
               new model wired in, new env var, etc.)
- `fix`      — bug fix in existing behavior
- `docs`     — documentation only (including `README.md` updates that
               accompany no code change)
- `refactor` — code change with no behavior change
- `perf`     — performance improvement with no behavior change
- `test`     — tests only
- `chore`    — tooling, deps, housekeeping
- `build`    — build system, Dockerfiles, image pins
- `ci`       — CI configuration

## Scopes (suggested, not exhaustive)

- `router`   — FastAPI router code
- `policy`   — `router/model_policy.yml` or policy decision logic
- `compose`  — `docker-compose.yml`, profiles, service wiring
- `env`      — `.env.example` and environment variable surface
- `docs`     — `README.md` and other documentation
- `tests`    — tests under `router/tests/`

Add new scopes as the repo grows; keep them lowercase and short.

## Subject line

- Imperative mood ("add", "fix", "remove" — not "added" / "adds").
- Lowercase.
- No trailing period.
- ≤ 72 characters.

## Body

Optional but strongly encouraged for anything non-trivial. Explain
**what** changed and **why**, not how. Wrap around 100 characters.

Call out explicitly, when applicable:

- Any change to exposed ports, bind addresses, or network posture.
- Any change to env var names, defaults, or required-vs-optional status.
- Any change to model IDs, `keep_alive`, `think` policy, or tool matching.
- Any change to the OpenAI-compatible request/response contract.
- Whether `README.md` was updated to match (it should be).

## Footer

- `Refs #N` / `Closes #N` for issue tracking.
- `BREAKING CHANGE: <description>` for anything that breaks an existing
  client config (LibreChat base URL, model IDs, header names, env var
  names, policy file schema, etc.).

## Examples

```
feat(policy): add tool-pattern matching for retrieval-style tools

Extend the policy file's tool list so additional retrieval/scrape style
tool calls disable thinking, matching existing fetch/browse behavior.
Updates README.md policy section and adds a unit test covering the new
match and the negative case.
```

```
fix(router): apply header override before size heuristic

Explicit override headers were ignored on large prompts because the
size check ran first. Reorder resolution so overrides win unconditionally.
README.md precedence section updated to match.
```

```
chore(env): document new optional threshold variable

Adds the variable to .env.example with a default and a short comment.
README.md operational section updated. No code change.
```

```
feat(router)!: rename OpenAI-compatible base path

BREAKING CHANGE: clients (including LibreChat) must update their base
URL. README.md and .env.example updated; migration note added.
```
