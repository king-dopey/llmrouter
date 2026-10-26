# Global Tool Minimization

Keep the active tool surface concise and purpose-driven.

## Rules

- Do not introduce a new tool when an existing tool already covers the same intent.
- Prefer a single authoritative tool for one decision point.
- Treat repo runtime probes, CI helpers, and operator diagnostics as non-baseline AI task tools.
- If a tool is optional or deferred, keep it out of the default mode flow.

## Web Research and Direct Fetch

Two distinct scenarios require different tools:

### Scenario A — URL unknown: search first

When you need to find information and do not yet have the specific URL:

1. Run `web_search` (quick factual lookup) or `web_search_and_summarize` (multi-source synthesis).
2. If a usable summary is returned, use it.
3. If the summary is unavailable or removes required detail, identify the most relevant URL from the returned results and run `curl -sL <url> -o /tmp/<slug>.html` to fetch the full page content.
4. Do not loop the original search. Do not abandon the verification step because a summary was unavailable.

### Scenario B — URL known: fetch directly

When you already have the specific URL (e.g., a well-known official documentation page, a URL returned by a previous search in the same session, or a URL stated in the task context) and need the full page content:

1. Skip `web_search_and_summarize` — there is nothing to discover.
2. Fetch the page directly: `curl -sL <url> -o /tmp/<slug>.html`.
3. Verify the download before reading: `wc -c /tmp/<slug>.html` (non-zero byte count confirms success).
4. Read the file for schema keys, configuration structure, or full technical detail.

Use Scenario B whenever the summary of a known page would strip structural detail needed for schema verification (e.g., YAML/JSON config keys, required field names, enumerated values). Use Scenario A when the URL itself still needs to be found.
