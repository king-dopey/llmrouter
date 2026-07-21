#!/usr/bin/env python3
"""
emit_agent_configs.py

Reads generated final prompt files and emits:
- custom_modes.yml
- mode_model_bindings.yml
- librechat_agent_blueprints.yml
- librechat_model_specs.yml
- create_agents.md

Usage:
  python emit_agent_configs.py \
    --prompts-dir ./generated_agent_prompts/final \
    --out ./emitted_agent_configs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Any


PROMPT_FILES = {
    "planning_prompt_parent": "planning_prompt_parent.txt",
    "planning_prompt_drafter": "planning_prompt_drafter.txt",
    "planning_prompt_refiner": "planning_prompt_refiner.txt",
    "planner": "planner.txt",
    "implementation_prompt_parent": "implementation_prompt_parent.txt",
    "implementation_prompt_drafter": "implementation_prompt_drafter.txt",
    "implementation_prompt_refiner": "implementation_prompt_refiner.txt",
    "implementer": "implementer.txt",
    "tester": "tester.txt",
    "reviewer": "reviewer.txt",
    "acceptance_judge": "acceptance_judge.txt",
    "diagnoser": "diagnoser.txt",
    "orchestrator": "orchestrator.txt",
}

# Finalized local-only model selections
MODELS = {
    "planning_prompt_parent": "qwen3.6:35b-a3b-q8_0",
    "planning_prompt_drafter": "qwen3.6:35b-a3b-q8_0",
    "planning_prompt_refiner": "gemma4:31b-it-q4_K_M",
    "planner": "qwen3.6:35b-a3b-q8_0",
    "implementation_prompt_parent": "qwen3.6:35b-a3b-q8_0",
    "implementation_prompt_drafter": "qwen3-coder-next:q4_K_M",
    "implementation_prompt_refiner": "devstral-small-2:24b-instruct-2512-q8_0",
    "implementer": "qwen3-coder-next:q4_K_M",
    "tester": "qwen3.6:35b-a3b-q8_0",
    "reviewer": "north-mini-code-1.0:q8_0",
    "acceptance_judge": "granite4.1-guardian:8b-q6_K",
    "diagnoser": "qwen3.6:35b-a3b-q8_0",
    "orchestrator": "qwen3.6:35b-a3b-q8_0",
}

MODEL_PARAMS = {
    "qwen3.6:35b-a3b-q8_0": {
        "temperature": 0.18,
        "maxContextTokens": 262144,
        "maxOutputTokens": 8192,
    },
    "gemma4:31b-it-q4_K_M": {
        "temperature": 0.20,
        "maxContextTokens": 131072,
        "maxOutputTokens": 6144,
    },
    "qwen3-coder-next:q4_K_M": {
        "temperature": 0.10,
        "maxContextTokens": 262144,
        "maxOutputTokens": 8192,
    },
    "devstral-small-2:24b-instruct-2512-q8_0": {
        "temperature": 0.15,
        "maxContextTokens": 131072,
        "maxOutputTokens": 6144,
    },
    "north-mini-code-1.0:q8_0": {
        "temperature": 0.10,
        "maxContextTokens": 131072,
        "maxOutputTokens": 6144,
    },
    "granite4.1-guardian:8b-q6_K": {
        "temperature": 0.0,
        "maxContextTokens": 65536,
        "maxOutputTokens": 2048,
    },
}

# Zoo modes -> prompt file + tool groups
ZOO_MODES = [
    {
        "slug": "planning-prompt",
        "name": "Planning Prompt",
        "description": "Produce only the planning prompt artifact.",
        "prompt_key": "planning_prompt_parent",
        "groups": ["read", "mcp"],
    },
    {
        "slug": "planner",
        "name": "Planner",
        "description": "Produce only the implementation plan artifact.",
        "prompt_key": "planner",
        "groups": [
            "read",
            ["edit", {"fileRegex": r"\.md$", "description": "Markdown files only"}],
            "mcp",
        ],
    },
    {
        "slug": "implementation-prompt",
        "name": "Implementation Prompt",
        "description": "Produce only the implementation prompt artifact.",
        "prompt_key": "implementation_prompt_parent",
        "groups": ["read", "mcp"],
    },
    {
        "slug": "implementer",
        "name": "Implementer",
        "description": "Apply approved code changes only.",
        "prompt_key": "implementer",
        "groups": ["read", "edit", "command", "mcp"],
    },
    {
        "slug": "tester",
        "name": "Tester",
        "description": "Validate the current implementation only.",
        "prompt_key": "tester",
        "groups": ["read", "command", "mcp"],
    },
    {
        "slug": "reviewer",
        "name": "Reviewer",
        "description": "Review completed work and recommend exactly one next direction only.",
        "prompt_key": "reviewer",
        "groups": ["read", "command", "mcp"],
    },
    {
        "slug": "acceptance-judge",
        "name": "Acceptance Judge",
        "description": "Return a binary acceptance judgment only.",
        "prompt_key": "acceptance_judge",
        "groups": ["read", "mcp"],
    },
    {
        "slug": "diagnoser",
        "name": "Diagnoser",
        "description": "Produce a root-cause diagnosis packet only.",
        "prompt_key": "diagnoser",
        "groups": ["read", "command", "mcp"],
    },
    {
        "slug": "orchestrator",
        "name": "Orchestrator",
        "description": "Delegate approved artifacts into isolated child tasks only.",
        "prompt_key": "orchestrator",
        "groups": ["mcp"],
    },
]

# LibreChat agents only for prompt generation/refinement
LIBRECHAT_AGENTS = [
    {
        "key": "planning_prompt_drafter",
        "name": "Planning Prompt Drafter",
        "description": "Drafts the first planning prompt artifact only.",
        "skills": [
            "repo-architecture-constraints",
            "module-boundaries",
            "coding-conventions",
            "validation-command-inventory",
        ],
        "tools": ["web_search"],
        "subagents": None,
    },
    {
        "key": "planning_prompt_refiner",
        "name": "Planning Prompt Refiner",
        "description": "Refines one planning prompt draft into a stronger final planning prompt only.",
        "skills": [
            "scope-control-rules",
            "constraint-handling",
            "ambiguity-removal",
            "handoff-quality-rules",
        ],
        "tools": ["web_search"],
        "subagents": None,
    },
    {
        "key": "planning_prompt_parent",
        "name": "Planning Prompt Parent",
        "description": "Produces the final planning prompt artifact only.",
        "skills": [
            "repo-architecture-constraints",
            "module-boundaries",
            "dependency-version-policy",
            "coding-conventions",
            "security-constraints",
            "validation-command-inventory",
            "documentation-update-policy",
        ],
        "tools": ["web_search", "subagent"],
        "subagents": [
            "agent_planning_prompt_drafter",
            "agent_planning_prompt_refiner",
        ],
    },
    {
        "key": "implementation_prompt_drafter",
        "name": "Implementation Prompt Drafter",
        "description": "Drafts the first implementation prompt artifact only.",
        "skills": [
            "file-targeting-rules",
            "scope-control-rules",
            "test-command-inventory",
            "coding-conventions",
        ],
        "tools": ["web_search"],
        "subagents": None,
    },
    {
        "key": "implementation_prompt_refiner",
        "name": "Implementation Prompt Refiner",
        "description": "Refines one implementation prompt draft into a narrower, safer, testable prompt only.",
        "skills": [
            "validation-command-inventory",
            "rollback-fallback-policy",
            "anti-overreach-rules",
            "file-targeting-rules",
        ],
        "tools": ["web_search"],
        "subagents": None,
    },
    {
        "key": "implementation_prompt_parent",
        "name": "Implementation Prompt Parent",
        "description": "Produces the final implementation prompt artifact only.",
        "skills": [
            "repo-architecture-constraints",
            "coding-conventions",
            "validation-command-inventory",
            "dependency-version-policy",
            "security-constraints",
            "documentation-update-policy",
        ],
        "tools": ["web_search", "subagent"],
        "subagents": [
            "agent_implementation_prompt_drafter",
            "agent_implementation_prompt_refiner",
        ],
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts-dir", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def read_prompt(prompts_dir: Path, key: str) -> str:
    path = prompts_dir / PROMPT_FILES[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def yaml_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else pad for line in text.splitlines())


def yaml_block(text: str, spaces: int = 0) -> str:
    pad = " " * spaces
    return "|\n" + "\n".join(pad + "  " + line for line in text.splitlines())


def render_groups(groups: List[Any], level: int = 4) -> str:
    lines = []
    pad = " " * level
    for g in groups:
        if isinstance(g, str):
            lines.append(f"{pad}- {g}")
        else:
            tool_name, opts = g
            lines.append(f"{pad}- - {tool_name}")
            for k, v in opts.items():
                lines.append(f"{pad}  - {k}: {v}")
    return "\n".join(lines)


def emit_custom_modes(prompts_dir: Path, out_dir: Path):
    lines = ["customModes:"]
    for mode in ZOO_MODES:
        prompt = read_prompt(prompts_dir, mode["prompt_key"])
        lines.extend([
            f"  - slug: {mode['slug']}",
            f"    name: {yaml_quote(mode['name'])}",
            f"    description: {yaml_quote(mode['description'])}",
            "    roleDefinition: >",
            f"      {mode['description']}",
            "    whenToUse: >",
            f"      Use this mode only when the next required workflow artifact matches its single purpose.",
            "    customInstructions: |",
        ])
        for line in prompt.splitlines():
            lines.append(f"      {line}")
        lines.append("    groups:")
        lines.append(render_groups(mode["groups"], level=6))
    (out_dir / "custom_modes.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_mode_model_bindings(out_dir: Path):
    mapping = {
        "planning-prompt": {
            "provider": "librechat-agents",
            "model_or_agent": "agent_planning_prompt_parent",
        },
        "planner": {
            "provider": "thor-router",
            "model_or_agent": MODELS["planner"],
        },
        "implementation-prompt": {
            "provider": "librechat-agents",
            "model_or_agent": "agent_implementation_prompt_parent",
        },
        "implementer": {
            "provider": "thor-router",
            "model_or_agent": MODELS["implementer"],
        },
        "tester": {
            "provider": "thor-router",
            "model_or_agent": MODELS["tester"],
        },
        "reviewer": {
            "provider": "thor-router",
            "model_or_agent": MODELS["reviewer"],
        },
        "acceptance-judge": {
            "provider": "thor-router",
            "model_or_agent": MODELS["acceptance_judge"],
        },
        "diagnoser": {
            "provider": "thor-router",
            "model_or_agent": MODELS["diagnoser"],
        },
        "orchestrator": {
            "provider": "thor-router",
            "model_or_agent": MODELS["orchestrator"],
        },
    }

    lines = ["modeBindings:"]
    for slug, cfg in mapping.items():
        lines.extend([
            f"  {slug}:",
            f"    provider: {cfg['provider']}",
            f"    model_or_agent: {yaml_quote(cfg['model_or_agent'])}",
        ])
    (out_dir / "mode_model_bindings.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_agent_blueprints(prompts_dir: Path, out_dir: Path):
    lines = ["agents:"]
    for a in LIBRECHAT_AGENTS:
        prompt = read_prompt(prompts_dir, a["key"])
        model = MODELS[a["key"]]
        params = MODEL_PARAMS[model]

        lines.extend([
            f"  - key: agent_{a['key']}",
            f"    name: {yaml_quote(a['name'])}",
            f"    description: {yaml_quote(a['description'])}",
            f"    model: {yaml_quote(model)}",
            f"    temperature: {params['temperature']}",
            f"    maxContextTokens: {params['maxContextTokens']}",
            f"    maxOutputTokens: {params['maxOutputTokens']}",
            f"    tools:",
        ])
        for tool in a["tools"]:
            lines.append(f"      - {tool}")

        lines.extend([
            "    skills_enabled: true",
            "    skills:",
        ])
        for skill in a["skills"]:
            lines.append(f"      - {skill}")

        if a["subagents"]:
            lines.extend([
                "    subagents:",
                "      enabled: true",
                "      allowSelf: false",
                "      agent_ids:",
            ])
            for sub in a["subagents"]:
                lines.append(f"        - {sub}")
        else:
            lines.extend([
                "    subagents:",
                "      enabled: false",
                "      allowSelf: false",
                "      agent_ids: []",
            ])

        lines.append("    instructions: |")
        for line in prompt.splitlines():
            lines.append(f"      {line}")

    (out_dir / "librechat_agent_blueprints.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_model_specs(out_dir: Path):
    lines = [
        "modelSpecs:",
        "  list:",
    ]

    for a in LIBRECHAT_AGENTS:
        key = f"agent_{a['key']}"
        lines.extend([
            f"    - name: {yaml_quote(key)}",
            f"      label: {yaml_quote(a['name'])}",
            f"      description: {yaml_quote(a['description'])}",
            "      skills:",
        ])
        for skill in a["skills"]:
            lines.append(f"        - {yaml_quote(skill)}")

        if a["subagents"]:
            lines.extend([
                "      subagents:",
                "        enabled: true",
                "        allowSelf: false",
                "        agent_ids:",
            ])
            for sub in a["subagents"]:
                lines.append(f"          - {yaml_quote(sub)}")
        else:
            lines.extend([
                "      subagents:",
                "        enabled: false",
                "        allowSelf: false",
                "        agent_ids: []",
            ])

        lines.extend([
            "      preset:",
            "        endpoint: 'agents'",
            f"        agent_id: 'REPLACE_WITH_{key.upper()}_ID'",
        ])

    (out_dir / "librechat_model_specs.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def emit_create_guide(out_dir: Path):
    text = """
# LibreChat Agent Creation Guide

Create these persisted agents in this order so parent agents can reference child agents:

1. Planning Prompt Drafter
2. Planning Prompt Refiner
3. Implementation Prompt Drafter
4. Implementation Prompt Refiner
5. Planning Prompt Parent
6. Implementation Prompt Parent

For each agent:
- Name: use the blueprint name
- Description: use the blueprint description
- Model: use the blueprint model
- Instructions: paste the full `instructions` block from `librechat_agent_blueprints.yml`
- Skills: enable skills and assign the listed skill IDs
- Tools: enable the listed tools
- Web Search: enable for every prompt agent
- Subagents:
  - for child agents: disabled
  - for parent agents: enabled, allowSelf=false
  - add the exact child agent IDs after the child agents are created

After creating them, update:
- `librechat_model_specs.yml`
- `mode_model_bindings.yml`

Replace:
- REPLACE_WITH_AGENT_PLANNING_PROMPT_PARENT_ID
- REPLACE_WITH_AGENT_IMPLEMENTATION_PROMPT_PARENT_ID
- etc.

Zoo/Roo side:
- put `custom_modes.yml` into the global modes location
- bind `planning-prompt` to the LibreChat parent agent `agent_planning_prompt_parent`
- bind `implementation-prompt` to the LibreChat parent agent `agent_implementation_prompt_parent`
- bind the remaining modes to Thor router models using `mode_model_bindings.yml`
""".lstrip()
    (out_dir / "create_agents.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    prompts_dir = Path(args.prompts_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    emit_custom_modes(prompts_dir, out_dir)
    emit_mode_model_bindings(out_dir)
    emit_agent_blueprints(prompts_dir, out_dir)
    emit_model_specs(out_dir)
    emit_create_guide(out_dir)

    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()