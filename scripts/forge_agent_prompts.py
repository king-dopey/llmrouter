#!/usr/bin/env python3
"""
forge_agent_prompts.py

Local-only agent prompt forge for the Thor stack.

What it does
------------
- Downloads reference pages and extracts text
- Reads local repo skill/context files
- Builds per-agent source packs
- Calls the local OpenAI-compatible endpoint at http://ask:4000/v1
- Uses qwen3.6 to generate production-grade agent prompts
- Uses a cross-family local reviewer to audit them
- Revises once using qwen3.6
- Writes final prompt files
- Writes a manual LibreChat UI request for each agent if needed

Behavior fixes
--------------
- request timeout defaults to 600s
- resumable: skips agents with existing final outputs
- reuses existing draft/review/revised files
- retries same reviewer on HTTP 500
- falls back to cross-family alternate reviewer only after same-model retries fail
- never uses qwen to review its own generated prompt
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests
from bs4 import BeautifulSoup


DEFAULT_API_BASE = "http://ask:4000/v1"
DEFAULT_API_KEY = "local"

GENERATOR_MODEL = "qwen3.6:35b-a3b-q8_0"
DEFAULT_REVIEWER_MODEL = "north-mini-code-1.0:q8_0"
ALT_REVIEWER_MODEL = "devstral-small-2:24b-instruct-2512-q8_0"
DEFAULT_REQUEST_TIMEOUT = 600

DEFAULT_LOCAL_GLOBS = [
    "AGENTS.md",
    ".roo/**/*.md",
    ".ai/**/*.md",
    "docs/**/*.md",
    "prompts/**/*.md",
]

SOURCES = [
    {
        "name": "librechat_agents",
        "url": "https://www.librechat.ai/docs/features/agents",
        "groups": ["librechat", "agents", "all"],
    },
    {
        "name": "librechat_subagents",
        "url": "https://www.librechat.ai/docs/features/subagents",
        "groups": ["librechat", "agents", "all"],
    },
    {
        "name": "librechat_agents_api",
        "url": "https://www.librechat.ai/docs/features/agents_api",
        "groups": ["librechat", "agents", "all"],
    },
    {
        "name": "librechat_web_search",
        "url": "https://www.librechat.ai/docs/features/web_search",
        "groups": ["librechat", "search", "all"],
    },
    {
        "name": "anthropic_prompting",
        "url": "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
        "groups": ["prompting", "all"],
    },
    {
        "name": "google_prompting",
        "url": "https://ai.google.dev/gemini-api/docs/prompting-intro",
        "groups": ["prompting", "all"],
    },
    {
        "name": "ollama_qwen36",
        "url": "https://ollama.com/library/qwen3.6",
        "groups": ["models", "qwen", "planning", "all"],
    },
    {
        "name": "ollama_qwen3_coder_next",
        "url": "https://ollama.com/library/qwen3-coder-next",
        "groups": ["models", "qwen", "implementation", "all"],
    },
    {
        "name": "ollama_gemma4",
        "url": "https://ollama.com/library/gemma4",
        "groups": ["models", "gemma", "planning", "all"],
    },
    {
        "name": "ollama_devstral_small_2",
        "url": "https://ollama.com/library/devstral-small-2",
        "groups": ["models", "devstral", "implementation", "review", "all"],
    },
    {
        "name": "ollama_north_mini_code",
        "url": "https://ollama.com/library/north-mini-code-1.0",
        "groups": ["models", "north", "review", "all"],
    },
    {
        "name": "ollama_granite_guardian",
        "url": "https://ollama.com/library/granite4.1-guardian",
        "groups": ["models", "judge", "review", "all"],
    },
    {
        "name": "qwen3_blog",
        "url": "https://qwenlm.github.io/blog/qwen3/",
        "groups": ["qwen", "prompting", "all"],
    },
    {
        "name": "qwen3_coder_blog",
        "url": "https://qwenlm.github.io/blog/qwen3-coder/",
        "groups": ["qwen", "coding", "implementation", "all"],
    },
]

MODEL_DIALECTS = {
    "qwen3.6:35b-a3b-q8_0": {
        "style": "xml_contract",
        "notes": [
            "Use explicit XML-style sections.",
            "Make artifact contracts extremely explicit.",
            "Use assumptions, failure policy, and final checks.",
            "Keep instructions ordered and unambiguous.",
        ],
    },
    "gemma4:31b-it-q4_K_M": {
        "style": "markdown_rewrite",
        "notes": [
            "Use clean markdown sections.",
            "Focus on rewrite quality, precision, and ambiguity removal.",
            "Avoid excessive XML nesting.",
        ],
    },
    "qwen3-coder-next:q4_K_M": {
        "style": "execution_packet",
        "notes": [
            "Keep prompt terse and executable.",
            "Focus on files, tasks, constraints, commands, done criteria.",
            "Avoid reflective/self-critique instructions.",
        ],
    },
    "devstral-small-2:24b-instruct-2512-q8_0": {
        "style": "runbook_checklist",
        "notes": [
            "Use runbook/checklist form.",
            "Be command-heavy and validation-heavy.",
            "Include rollback and anti-overreach rules.",
        ],
    },
    "north-mini-code-1.0:q8_0": {
        "style": "schema_tool_review",
        "notes": [
            "Use schema-first instructions.",
            "Be explicit about tool policy and decision logic.",
            "Make review criteria and output schema strict.",
        ],
    },
    "granite4.1-guardian:8b-q6_K": {
        "style": "binary_judge",
        "notes": [
            "Use strict pass/fail or accept/reject contract.",
            "Keep output highly constrained.",
        ],
    },
}

AGENTS: List[Dict[str, Any]] = [
    {
        "name": "planning_prompt_parent",
        "target_model": "qwen3.6:35b-a3b-q8_0",
        "purpose": "Produce the final planning prompt artifact only.",
        "artifact": "Planning prompt artifact",
        "agent_type": "librechat_parent",
        "source_groups": ["librechat", "prompting", "planning", "qwen", "models"],
        "skills": [
            "repo architecture constraints",
            "module boundaries",
            "dependency/version policy",
            "coding conventions",
            "security constraints",
            "validation command inventory",
            "documentation/update policy",
        ],
        "tools": ["subagent", "web_search"],
    },
    {
        "name": "planning_prompt_drafter",
        "target_model": "qwen3.6:35b-a3b-q8_0",
        "purpose": "Draft the first planning prompt artifact only.",
        "artifact": "Planning prompt draft",
        "agent_type": "librechat_subagent",
        "source_groups": ["prompting", "planning", "qwen", "models"],
        "skills": [
            "repo architecture constraints",
            "module boundaries",
            "coding conventions",
            "validation command inventory",
        ],
        "tools": ["web_search"],
    },
    {
        "name": "planning_prompt_refiner",
        "target_model": "gemma4:31b-it-q4_K_M",
        "purpose": "Refine one planning prompt draft into a stronger final planning prompt only.",
        "artifact": "Refined planning prompt",
        "agent_type": "librechat_subagent",
        "source_groups": ["prompting", "planning", "gemma", "models"],
        "skills": [
            "scope control rules",
            "constraint handling",
            "ambiguity removal",
            "handoff quality rules",
        ],
        "tools": ["web_search"],
    },
    {
        "name": "planner",
        "target_model": "qwen3.6:35b-a3b-q8_0",
        "purpose": "Produce the implementation plan artifact only.",
        "artifact": "Implementation plan",
        "agent_type": "zoo_mode",
        "source_groups": ["prompting", "planning", "qwen", "models"],
        "skills": [
            "repo architecture constraints",
            "test strategy",
            "deployment/rollback policy",
            "coding conventions",
        ],
        "tools": ["repo_read"],
    },
    {
        "name": "implementation_prompt_parent",
        "target_model": "qwen3.6:35b-a3b-q8_0",
        "purpose": "Produce the final implementation prompt artifact only.",
        "artifact": "Implementation prompt artifact",
        "agent_type": "librechat_parent",
        "source_groups": ["librechat", "prompting", "implementation", "qwen", "models"],
        "skills": [
            "repo architecture constraints",
            "coding conventions",
            "validation command inventory",
            "dependency/version policy",
            "security constraints",
            "documentation/update policy",
        ],
        "tools": ["subagent", "web_search"],
    },
    {
        "name": "implementation_prompt_drafter",
        "target_model": "qwen3-coder-next:q4_K_M",
        "purpose": "Draft the first implementation prompt artifact only.",
        "artifact": "Implementation prompt draft",
        "agent_type": "librechat_subagent",
        "source_groups": ["prompting", "implementation", "qwen", "coding", "models"],
        "skills": [
            "file targeting rules",
            "scope control rules",
            "test command inventory",
            "coding conventions",
        ],
        "tools": ["web_search"],
    },
    {
        "name": "implementation_prompt_refiner",
        "target_model": "devstral-small-2:24b-instruct-2512-q8_0",
        "purpose": "Refine one implementation prompt draft into a narrower, safer, testable implementation prompt only.",
        "artifact": "Refined implementation prompt",
        "agent_type": "librechat_subagent",
        "source_groups": ["prompting", "implementation", "devstral", "models"],
        "skills": [
            "validation command inventory",
            "rollback/fallback policy",
            "anti-overreach rules",
            "file targeting rules",
        ],
        "tools": ["web_search"],
    },
    {
        "name": "implementer",
        "target_model": "qwen3-coder-next:q4_K_M",
        "purpose": "Apply approved code changes only.",
        "artifact": "Code changes and concise change log",
        "agent_type": "zoo_mode",
        "source_groups": ["implementation", "qwen", "coding", "models"],
        "skills": [
            "coding conventions",
            "test command inventory",
            "dependency/version policy",
            "security constraints",
            "documentation/update policy",
        ],
        "tools": ["repo_read", "repo_write", "command"],
    },
    {
        "name": "tester",
        "target_model": "qwen3.6:35b-a3b-q8_0",
        "purpose": "Validate the current implementation only.",
        "artifact": "Test report",
        "agent_type": "zoo_mode",
        "source_groups": ["implementation", "planning", "qwen", "models"],
        "skills": [
            "test command inventory",
            "integration test policy",
            "smoke test policy",
            "failure capture standards",
        ],
        "tools": ["repo_read", "command"],
    },
    {
        "name": "reviewer",
        "target_model": "north-mini-code-1.0:q8_0",
        "purpose": "Review completed work and recommend exactly one next direction only.",
        "artifact": "Review report",
        "agent_type": "zoo_mode",
        "source_groups": ["review", "north", "devstral", "models", "prompting"],
        "skills": [
            "acceptance criteria policy",
            "scope drift detection",
            "hidden regression checklist",
            "test sufficiency checklist",
        ],
        "tools": ["repo_read", "command", "web_search"],
    },
    {
        "name": "acceptance_judge",
        "target_model": "granite4.1-guardian:8b-q6_K",
        "purpose": "Return a binary acceptance judgment only.",
        "artifact": "Binary acceptance judgment",
        "agent_type": "zoo_mode",
        "source_groups": ["review", "judge", "models"],
        "skills": [
            "acceptance criteria policy",
            "binary pass/fail policy",
        ],
        "tools": [],
    },
    {
        "name": "diagnoser",
        "target_model": "qwen3.6:35b-a3b-q8_0",
        "purpose": "Produce a root-cause diagnosis packet only.",
        "artifact": "Diagnosis packet",
        "agent_type": "zoo_mode",
        "source_groups": ["review", "implementation", "planning", "qwen", "models"],
        "skills": [
            "root-cause analysis policy",
            "minimal fix strategy policy",
            "failure evidence capture",
        ],
        "tools": ["repo_read", "command", "web_search"],
    },
    {
        "name": "orchestrator",
        "target_model": "qwen3.6:35b-a3b-q8_0",
        "purpose": "Delegate approved artifacts into isolated child tasks only.",
        "artifact": "Delegation packets",
        "agent_type": "zoo_mode",
        "source_groups": ["librechat", "prompting", "planning", "qwen", "models"],
        "skills": [
            "workflow transition rules",
            "artifact handoff contracts",
            "mode routing rules",
        ],
        "tools": ["new_task"],
    },
]


def slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_")


def ensure_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "base": base,
        "raw": base / "raw",
        "text": base / "text",
        "packs": base / "packs",
        "manual": base / "manual",
        "drafts": base / "drafts",
        "reviews": base / "reviews",
        "revised": base / "revised",
        "final": base / "final",
        "manifest": base / "manifest",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def fetch_url(url: str, timeout: int = 45) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "thor-prompt-forge/1.0"})
    r.raise_for_status()
    return r.text


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "img", "meta", "link"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


def read_text_if_exists(path: Path) -> Optional[str]:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_local_context(globs_list: List[str], max_chars_per_file: int) -> List[Dict[str, str]]:
    found = []
    seen = set()
    for pattern in globs_list:
        for path in glob.glob(pattern, recursive=True):
            p = Path(path)
            if not p.is_file():
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                text = trim_text(text.strip(), max_chars_per_file)
                if text:
                    found.append({
                        "name": p.name,
                        "path": str(p),
                        "text": text,
                        "groups": ["local", "skills", "repo", "all"],
                    })
            except Exception:
                pass
    return found


def save_downloaded_sources(dirs: Dict[str, Path], max_chars_per_source: int) -> List[Dict[str, str]]:
    manifest = []
    for src in SOURCES:
        try:
            html = fetch_url(src["url"])
            text = trim_text(html_to_text(html), max_chars_per_source)

            raw_path = dirs["raw"] / f"{slugify(src['name'])}.html"
            txt_path = dirs["text"] / f"{slugify(src['name'])}.txt"

            write_text(raw_path, html)
            write_text(txt_path, text)

            manifest.append({
                "name": src["name"],
                "url": src["url"],
                "text_path": str(txt_path),
                "groups": src["groups"],
                "kind": "remote",
            })
            print(f"[ok] downloaded {src['name']}")
        except Exception as e:
            print(f"[warn] failed {src['name']}: {e}")
    return manifest


def build_pack(
    agent: Dict[str, Any],
    all_sources: List[Dict[str, Any]],
    local_sources: List[Dict[str, Any]],
    max_pack_chars: int,
) -> str:
    wanted = set(agent["source_groups"])
    sections = []

    for src in all_sources:
        if wanted.intersection(set(src["groups"])) or "all" in src["groups"]:
            try:
                text = Path(src["text_path"]).read_text(encoding="utf-8")
                section = (
                    f"# SOURCE: {src['name']}\n"
                    f"# URL: {src['url']}\n"
                    f"# GROUPS: {', '.join(src['groups'])}\n\n"
                    f"{text}\n"
                )
                sections.append(section)
            except Exception:
                pass

    for src in local_sources:
        section = (
            f"# LOCAL CONTEXT: {src['name']}\n"
            f"# PATH: {src['path']}\n"
            f"# GROUPS: {', '.join(src['groups'])}\n\n"
            f"{src['text']}\n"
        )
        sections.append(section)

    pack = "\n\n" + ("\n\n" + ("-" * 80) + "\n\n").join(sections)
    return trim_text(pack, max_pack_chars)


def choose_reviewer_model(target_model: str) -> str:
    if target_model == DEFAULT_REVIEWER_MODEL:
        return ALT_REVIEWER_MODEL
    return DEFAULT_REVIEWER_MODEL


def choose_reviewer_fallback(primary_model: str) -> Optional[str]:
    if primary_model == DEFAULT_REVIEWER_MODEL:
        return ALT_REVIEWER_MODEL
    if primary_model == ALT_REVIEWER_MODEL:
        return DEFAULT_REVIEWER_MODEL
    return None


def is_retryable_http_500(exc: Exception) -> bool:
    if not isinstance(exc, requests.exceptions.HTTPError):
        return False
    response = getattr(exc, "response", None)
    if response is None:
        return False
    return response.status_code == 500


def api_chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.15,
    max_tokens: int = 7000,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def api_chat_with_retries(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retry_count: int,
    retry_sleep: int,
) -> str:
    last_exc = None
    for attempt in range(1, retry_count + 1):
        try:
            if attempt > 1:
                print(f"[retry] model={model} attempt={attempt}/{retry_count}")
            return api_chat(
                api_base=api_base,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            last_exc = e
            if not is_retryable_http_500(e):
                raise
            if attempt < retry_count:
                print(f"[warn] retryable 500 from model {model}: {e}")
                time.sleep(retry_sleep)
    raise last_exc


PROMPT_ARCHITECT_SYSTEM = textwrap.dedent("""
You are Prompt Forge.

Your job is to write a production-grade SYSTEM PROMPT for exactly one target agent.

You are not solving the user's coding task.
You are designing the target agent's operating instructions.

Hard requirements:
1. The target agent must have exactly one purpose.
2. The target agent must produce exactly one primary artifact.
3. The final prompt must be deployable as-is.
4. The final prompt must be model-aware.
5. The final prompt must include explicit web-search policy if the target agent can use web search.
6. The final prompt must include skill-context policy.
7. The final prompt must include output contract and stop conditions.
8. The final prompt must not mix planning, coding, testing, review, diagnosis, or orchestration unless the target agent's single purpose is exactly that step.

Mandatory sections to include in the generated agent prompt:
- identity
- single_purpose
- primary_artifact
- allowed_inputs
- skill_context
- repository_context_policy
- web_search_policy
- tool_usage_policy
- process
- quality_bar
- hard_rules
- failure_policy
- output_contract
- stop_conditions
- final_checks

Model-awareness rules:
- If the target model is planning/reasoning-oriented, use explicit structured sections and a strong quality bar.
- If the target model is execution-oriented and non-thinking, make the prompt terse, ordered, and directly executable.
- If the target model is runbook-oriented, use checklist/runbook style.
- If the target model is review-oriented, make decision criteria and output schema strict.
- If the target model is a binary judge, keep output contract minimal and deterministic.

Web-search rules to embed when relevant:
- Use web search only for external, non-repo, current facts that materially affect task quality.
- Never use web search for repository-local facts.
- Use web search for current library/framework behavior, official docs, standards, API changes, package/version facts, CVEs/advisories, or upstream deprecations only when needed.
- Summarize only the facts needed for the artifact.
- Fold searched facts into constraints, validation, or review criteria instead of narrating research.

Skill-context rules to embed:
- Treat local skill/context documents as durable instructions.
- Prioritize architecture constraints, module boundaries, coding conventions, validation commands, deployment/rollback constraints, security constraints, and documentation/update policy when relevant.
- Use only the skill context needed for the current artifact.
- Do not restate all skill context if only a subset applies.

Do not output commentary about the prompt.
Do not explain your reasoning.
Output the final system prompt only.
""").strip()


def build_generation_request(agent: Dict[str, Any], pack_text: str) -> str:
    dialect = MODEL_DIALECTS.get(agent["target_model"], {"style": "structured", "notes": []})
    notes = "\n".join(f"- {n}" for n in dialect["notes"])

    if agent["agent_type"] == "librechat_parent":
        subagent_clause = """
Additional requirement:
- This is a LibreChat parent agent.
- It must explicitly instruct itself to call its designated subagents in sequence when performing its single purpose.
- It must not do the subagent's work directly if delegation is part of its contract.
"""
    elif agent["agent_type"] == "librechat_subagent":
        subagent_clause = """
Additional requirement:
- This is a LibreChat subagent.
- It must not behave like a parent/orchestrator.
- It must perform only its own subtask and return its own artifact.
"""
    else:
        subagent_clause = """
Additional requirement:
- This is a Zoo mode prompt.
- It must focus on the single workflow step assigned to this mode.
"""

    tools = ", ".join(agent["tools"]) if agent["tools"] else "none"
    skills = "\n".join(f"- {s}" for s in agent["skills"])

    return textwrap.dedent(f"""
    Write the full production-grade SYSTEM PROMPT for this target agent.

    Target agent name:
    - {agent['name']}

    Target model:
    - {agent['target_model']}

    Single purpose:
    - {agent['purpose']}

    Primary artifact:
    - {agent['artifact']}

    Agent class:
    - {agent['agent_type']}

    Tools available:
    - {tools}

    Required skill-context topics:
    {skills}

    Prompt dialect requirements for this target model:
    - style: {dialect['style']}
    {notes if notes else "- no additional notes"}

    Required behavior:
    - Make the target agent single-purpose.
    - Make the target agent produce exactly one primary artifact.
    - Include a strong context-minimization policy.
    - Include an explicit web-search policy if and only if web_search is in available tools.
    - Include an explicit tool-usage policy.
    - Include explicit blocker behavior.
    - Include explicit ambiguity handling.
    - Include explicit stop conditions.
    - Include a strict output contract.
    - Include rules for scope control, no hidden task expansion, and no silent assumption changes.
    - Include enough detail for real production use.
    - The prompt must be long, explicit, and operational, not a short role blurb.

    {subagent_clause.strip()}

    Use the source pack below as reference material.
    Synthesize from it.
    Do not quote it verbatim unless needed.
    Do not mention the source pack in the final prompt.

    SOURCE PACK START
    {pack_text}
    SOURCE PACK END
    """).strip()


PROMPT_REVIEW_SYSTEM = textwrap.dedent("""
You are Prompt Contract Reviewer.

You review one generated agent system prompt against its target contract.

Return strict JSON only with this schema:
{
  "decision": "PASS" | "REVISE",
  "summary": "short summary",
  "missing_sections": ["..."],
  "contract_violations": ["..."],
  "model_misalignment": ["..."],
  "tool_policy_issues": ["..."],
  "web_search_policy_issues": ["..."],
  "skill_context_issues": ["..."],
  "revision_instructions": ["..."]
}

Review dimensions:
- single purpose purity
- single artifact purity
- prompt completeness
- model/dialect fit
- skill-context handling
- web-search usage policy
- tool policy
- coding-workflow discipline
- stop conditions
- output contract clarity

Do not rewrite the prompt.
Do not explain outside JSON.
""").strip()


def build_review_request(agent: Dict[str, Any], candidate_prompt: str) -> str:
    dialect = MODEL_DIALECTS.get(agent["target_model"], {"style": "structured"})
    return textwrap.dedent(f"""
    Review this generated system prompt.

    Target agent:
    - {agent['name']}

    Target model:
    - {agent['target_model']}

    Expected single purpose:
    - {agent['purpose']}

    Expected primary artifact:
    - {agent['artifact']}

    Agent class:
    - {agent['agent_type']}

    Expected tools:
    - {", ".join(agent['tools']) if agent['tools'] else "none"}

    Required skill-context topics:
    - {"; ".join(agent['skills'])}

    Expected dialect:
    - {dialect['style']}

    Candidate prompt:
    {candidate_prompt}
    """).strip()


PROMPT_REVISION_SYSTEM = textwrap.dedent("""
You are Prompt Forge Revision.

You will receive:
- a target agent contract
- a previously generated system prompt
- a JSON review report

Your task:
- revise the prompt to satisfy the review
- preserve the agent's single purpose
- preserve the agent's single artifact
- keep the prompt deployable
- keep the prompt model-aware

Output the revised final system prompt only.
Do not output commentary.
""").strip()


def build_revision_request(agent: Dict[str, Any], previous_prompt: str, review_json: str) -> str:
    return textwrap.dedent(f"""
    Revise this generated system prompt.

    Target agent:
    - {agent['name']}

    Target model:
    - {agent['target_model']}

    Required single purpose:
    - {agent['purpose']}

    Required primary artifact:
    - {agent['artifact']}

    Tools:
    - {", ".join(agent['tools']) if agent['tools'] else "none"}

    Required skill-context topics:
    - {"; ".join(agent['skills'])}

    Previous prompt:
    {previous_prompt}

    Review JSON:
    {review_json}
    """).strip()


def build_manual_request(agent: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""
    Use model: {GENERATOR_MODEL}

    Attach the matching pack file for this agent.

    Task:
    Generate the full production-grade SYSTEM PROMPT for the target agent below.

    Target agent:
    - {agent['name']}

    Target model:
    - {agent['target_model']}

    Single purpose:
    - {agent['purpose']}

    Primary artifact:
    - {agent['artifact']}

    Agent class:
    - {agent['agent_type']}

    Available tools:
    - {", ".join(agent['tools']) if agent['tools'] else "none"}

    Required skill-context topics:
    - {"; ".join(agent['skills'])}

    Requirements:
    - The prompt must be deployable as-is.
    - The prompt must define exactly one purpose.
    - The prompt must define exactly one primary artifact.
    - The prompt must include: identity, single_purpose, primary_artifact, allowed_inputs,
      skill_context, repository_context_policy, web_search_policy (if relevant),
      tool_usage_policy, process, quality_bar, hard_rules, failure_policy, output_contract,
      stop_conditions, final_checks.
    - The prompt must be long and operational, not a short role blurb.
    - The prompt must be formatted for the target model's strengths.
    - Do not output commentary.
    - Output the final system prompt only.
    """).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-base", default=DEFAULT_API_BASE)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--out", default="./generated_agent_prompts")
    p.add_argument("--generator-model", default=GENERATOR_MODEL)
    p.add_argument("--reviewer-model", default=DEFAULT_REVIEWER_MODEL)
    p.add_argument("--local-glob", action="append", default=[])
    p.add_argument("--chars-per-source", type=int, default=18000)
    p.add_argument("--chars-per-local-file", type=int, default=18000)
    p.add_argument("--max-pack-chars", type=int, default=140000)
    p.add_argument("--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    p.add_argument("--retry-count", type=int, default=3)
    p.add_argument("--retry-sleep", type=int, default=20)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    dirs = ensure_dirs(out)

    local_globs = list(DEFAULT_LOCAL_GLOBS)
    if args.local_glob:
        local_globs.extend(args.local_glob)

    remote_manifest = save_downloaded_sources(dirs, args.chars_per_source)
    write_text(dirs["manifest"] / "remote_manifest.json", json.dumps(remote_manifest, indent=2))

    local_sources = load_local_context(local_globs, args.chars_per_local_file)
    write_text(dirs["manifest"] / "local_sources.json", json.dumps(local_sources, indent=2))

    if args.dry_run:
        print("Dry run complete.")
        return 0

    final_manifest = []

    for agent in AGENTS:
        print(f"\n=== {agent['name']} ===")

        pack_path = dirs["packs"] / f"{agent['name']}_pack.md"
        manual_path = dirs["manual"] / f"{agent['name']}_request.md"
        draft_path = dirs["drafts"] / f"{agent['name']}.txt"
        review_path = dirs["reviews"] / f"{agent['name']}.json"
        review_err_path = dirs["reviews"] / f"{agent['name']}.error.txt"
        revised_path = dirs["revised"] / f"{agent['name']}.txt"
        final_path = dirs["final"] / f"{agent['name']}.txt"

        if final_path.exists() and not args.force:
            print(f"[skip] {agent['name']}: final already exists")
            final_manifest.append({
                "agent": agent["name"],
                "target_model": agent["target_model"],
                "generator_model": args.generator_model,
                "reviewer_model": choose_reviewer_model(agent["target_model"]),
                "pack": str(pack_path),
                "manual_request": str(manual_path),
                "draft": str(draft_path),
                "review": str(review_path),
                "final": str(final_path),
            })
            continue

        if not pack_path.exists() or args.force:
            pack_text = build_pack(agent, remote_manifest, local_sources, args.max_pack_chars)
            write_text(pack_path, pack_text)
        else:
            pack_text = read_text_if_exists(pack_path) or ""
            print(f"[reuse] {agent['name']}: pack")

        if not manual_path.exists() or args.force:
            write_text(manual_path, build_manual_request(agent))
        else:
            print(f"[reuse] {agent['name']}: manual request")

        draft = read_text_if_exists(draft_path)
        if draft is None or args.force:
            gen_messages = [
                {"role": "system", "content": PROMPT_ARCHITECT_SYSTEM},
                {"role": "user", "content": build_generation_request(agent, pack_text)},
            ]
            draft = api_chat_with_retries(
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.generator_model,
                messages=gen_messages,
                temperature=0.18,
                max_tokens=9000,
                timeout=args.request_timeout,
                retry_count=args.retry_count,
                retry_sleep=args.retry_sleep,
            )
            write_text(draft_path, draft)
        else:
            print(f"[reuse] {agent['name']}: draft")

        reviewer_model = choose_reviewer_model(agent["target_model"])
        review_raw = read_text_if_exists(review_path)
        if review_raw is None or args.force:
            review_messages = [
                {"role": "system", "content": PROMPT_REVIEW_SYSTEM},
                {"role": "user", "content": build_review_request(agent, draft)},
            ]

            try:
                review_raw = api_chat_with_retries(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=reviewer_model,
                    messages=review_messages,
                    temperature=0.10,
                    max_tokens=3000,
                    timeout=args.request_timeout,
                    retry_count=args.retry_count,
                    retry_sleep=args.retry_sleep,
                )
            except Exception as first_exc:
                fallback_model = choose_reviewer_fallback(reviewer_model)
                if fallback_model:
                    print(f"[warn] reviewer {reviewer_model} failed for {agent['name']}, trying fallback reviewer {fallback_model}")
                    try:
                        review_raw = api_chat_with_retries(
                            api_base=args.api_base,
                            api_key=args.api_key,
                            model=fallback_model,
                            messages=review_messages,
                            temperature=0.10,
                            max_tokens=3000,
                            timeout=args.request_timeout,
                            retry_count=args.retry_count,
                            retry_sleep=args.retry_sleep,
                        )
                        reviewer_model = fallback_model
                    except Exception as second_exc:
                        err = (
                            f"Primary reviewer failed: {reviewer_model}\n{repr(first_exc)}\n\n"
                            f"Fallback reviewer failed: {fallback_model}\n{repr(second_exc)}\n"
                        )
                        write_text(review_err_path, err)
                        print(f"[error] review failed for {agent['name']}; saved {review_err_path}")
                        raise
                else:
                    err = f"Reviewer failed: {reviewer_model}\n{repr(first_exc)}\n"
                    write_text(review_err_path, err)
                    print(f"[error] review failed for {agent['name']}; saved {review_err_path}")
                    raise

            write_text(review_path, review_raw)
        else:
            print(f"[reuse] {agent['name']}: review")

        final_prompt = draft
        try:
            review_obj = json.loads(review_raw)
        except Exception:
            review_obj = {
                "decision": "REVISE",
                "revision_instructions": ["Reviewer output was not valid JSON; revise conservatively."],
            }

        if review_obj.get("decision") == "REVISE":
            existing_revised = read_text_if_exists(revised_path)
            if existing_revised is None or args.force:
                rev_messages = [
                    {"role": "system", "content": PROMPT_REVISION_SYSTEM},
                    {"role": "user", "content": build_revision_request(agent, draft, review_raw)},
                ]
                final_prompt = api_chat_with_retries(
                    api_base=args.api_base,
                    api_key=args.api_key,
                    model=args.generator_model,
                    messages=rev_messages,
                    temperature=0.12,
                    max_tokens=9000,
                    timeout=args.request_timeout,
                    retry_count=args.retry_count,
                    retry_sleep=args.retry_sleep,
                )
                write_text(revised_path, final_prompt)
            else:
                print(f"[reuse] {agent['name']}: revised")
                final_prompt = existing_revised

        write_text(final_path, final_prompt)

        final_manifest.append({
            "agent": agent["name"],
            "target_model": agent["target_model"],
            "generator_model": args.generator_model,
            "reviewer_model": reviewer_model,
            "pack": str(pack_path),
            "manual_request": str(manual_path),
            "draft": str(draft_path),
            "review": str(review_path),
            "final": str(final_path),
        })

    write_text(dirs["manifest"] / "final_manifest.json", json.dumps(final_manifest, indent=2))

    print(f"\nDone. Final prompts: {dirs['final']}")
    print(f"Manual UI requests: {dirs['manual']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())