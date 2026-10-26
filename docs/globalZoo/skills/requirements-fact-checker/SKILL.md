---
name: requirements-fact-checker
description: Use this skill to validate stated technical values (timeouts, config keys, env vars, file paths, resource limits) against codebase evidence and authoritative documentation before accepting them as requirements. Apply in Tech Lead before decomposing, and in Architect before planning any change to an external system.
modeSlugs:
  - tech-lead
  - architect
---

# Requirements Fact Checker

## Purpose

Prevent architectural drift caused by accepting user-stated technical values as verified facts without confirmation. Every stated configuration key, numeric threshold, environment variable name, or file path must be confirmed before it is decomposed into stories or included in an implementation plan.

## When to Apply

- **Tech Lead**: during step 2b (Validate Requirements), before sizing the ask.
- **Architect**: during step 2 (Gather Context and Verify Schema), before synthesizing the plan.

Apply whenever the request contains any of the following:
- A specific numeric value (timeout, memory limit, CPU count, port, file size)
- A configuration key name (e.g., `proxy_read_timeout`, `systemprompt`, `EXECUTION_TIMEOUT`)
- An environment variable name
- A file path claimed to exist in the repository
- A claim about how an external system (LibreChat, nginx, Docker, a cloud service) is configured

## Checklist

### 1. Configuration Keys
- Search the repository's existing config files for the key (`codebase_search`).
- If not found in the repo, check official documentation. If the documentation URL is well-known (e.g., `https://www.librechat.ai/docs/configuration/...`), fetch it directly: `curl -sL <url> -o /tmp/<slug>.html` and verify with `wc -c`. If the URL is unknown, use `web_search` with the system name + key name + "configuration"; use the summary if available, or fetch the returned URL with `curl -sL <url> -o /tmp/<slug>.html` if the summary is insufficient.
- **Pass**: key confirmed valid in the documented schema.
- **Fail**: key not found — record as UNCERTAIN, add as `blocker_tag = missing-input`, do not include in plan.

### 2. Numeric Values (timeouts, limits, sizes)
- Check the codebase for existing values of the same type.
- Check official documentation for documented defaults, maximum supported values, and recommended settings. If the documentation URL is known, fetch it directly with `curl -sL <url> -o /tmp/<slug>.html`. If the URL is unknown, use `web_search_and_summarize`; use the summary if available, or fetch the returned URL with `curl` if the summary is insufficient.
- **Pass**: stated value is consistent with documented range and existing codebase values.
- **Conflict**: stated value differs from documented default or known good value — record both in `<assumptions>`, use the verified value, flag the discrepancy explicitly.
- **Unverifiable**: no authoritative source found — record as UNCERTAIN, note the required verification step.

### 3. File Paths
- Verify the path exists in the repository using `codebase_search` or `search_files`.
- **Pass**: path confirmed to exist.
- **Fail**: path not found — record as UNCERTAIN; do not treat the file as a dependency without evidence.

### 4. Environment Variable Names
- Search the codebase for existing usage of the variable name.
- Check official documentation for the correct name.
- **Pass**: name confirmed in codebase or docs.
- **Fail**: name not found — record as UNCERTAIN with the canonical name from docs if available.

### 5. External System Architecture Claims
- Before accepting a claim about how an external system works (e.g., "LibreChat reads agents from librechat.yaml"), verify it by searching the repository's configuration and the system's official documentation.
- If the claim is wrong, note the correct architecture in `<assumptions>` and adjust the story or plan accordingly.

## Outputs

- `<assumptions>` entries for each verified value (confirmed, conflicting, or UNCERTAIN).
- Blocker items for any value that cannot be confirmed and must not proceed to implementation.
- A documented source for each confirmed value (file path in repo or URL from web search).

## Hard Rules

- Never accept a user-stated configuration key as valid without evidence.
- Never carry a numeric threshold into a story or plan without verifying it against authoritative sources.
- A value classified as UNCERTAIN must not appear as a confirmed implementation detail — it must be a blocker or a clearly flagged assumption awaiting resolution.
- If `web_search_and_summarize` returns pages and snippets but the summary is unavailable or removes necessary context, use `curl -sL <url> -o /tmp/<slug>.html` on the most relevant URL returned by the tool — verify with `wc -c` before reading. If you already have the specific documentation URL, go directly to curl without searching. Do not loop, do not abandon the check.
