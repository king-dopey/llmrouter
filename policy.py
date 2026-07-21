import os
import re
from dataclasses import dataclass
from typing import Any

DEFAULT_DISABLE_TOOL_PATTERNS = [
    "web_search",
    "search",
    "browser",
    "browse",
    "serp",
    "http_get",
    "http",
    "fetch",
    "scrape",
]

DEFAULT_ENABLE_TOOL_PATTERNS = [
    "file_search",
    "retrieval",
    "rag",
    "openweather",
    "weather",
]

DEFAULT_SUMMARY_PATTERNS = [
    "summarize",
    "tl;dr",
    "tldr",
    "bullet summary",
    "key takeaways",
]

DEFAULT_CHAR_THRESHOLD = 12000


@dataclass(frozen=True)
class ThinkPolicyConfig:
    disable_tool_patterns: list[str]
    enable_tool_patterns: list[str]
    summary_patterns: list[str]
    char_threshold: int


def _parse_csv_env(name: str, defaults: list[str]) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return defaults
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def load_think_policy_config() -> ThinkPolicyConfig:
    threshold_raw = os.getenv("DISABLE_THINK_CHAR_THRESHOLD", str(DEFAULT_CHAR_THRESHOLD)).strip()
    try:
        char_threshold = max(1, int(threshold_raw))
    except ValueError:
        char_threshold = DEFAULT_CHAR_THRESHOLD

    return ThinkPolicyConfig(
        disable_tool_patterns=_parse_csv_env("DISABLE_THINK_TOOL_PATTERNS", DEFAULT_DISABLE_TOOL_PATTERNS),
        enable_tool_patterns=DEFAULT_ENABLE_TOOL_PATTERNS,
        summary_patterns=_parse_csv_env("DISABLE_THINK_SUMMARY_PATTERNS", DEFAULT_SUMMARY_PATTERNS),
        char_threshold=char_threshold,
    )


def parse_think_override(header_value: str | None) -> bool | None:
    if header_value is None:
        return None

    value = header_value.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("X-Ollama-Think must be 'true' or 'false'")


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _matches_pattern_by_tokens(value: str, pattern: str) -> bool:
    value_tokens = _tokenize(value)
    pattern_tokens = _tokenize(pattern)

    if not value_tokens or not pattern_tokens:
        return False

    if len(pattern_tokens) == 1:
        return pattern_tokens[0] in value_tokens

    window = len(pattern_tokens)
    for idx in range(0, len(value_tokens) - window + 1):
        if value_tokens[idx : idx + window] == pattern_tokens:
            return True
    return False


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def _collect_tool_names(body: dict[str, Any]) -> list[str]:
    names: list[str] = []

    for tool in body.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function", {}) if isinstance(tool.get("function"), dict) else {}
        tool_name = fn.get("name") or tool.get("name")
        if tool_name:
            names.append(str(tool_name))

    for msg in body.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue

        msg_name = msg.get("name")
        if msg.get("role") in {"tool", "function"} and msg_name:
            names.append(str(msg_name))

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
                fn_name = fn.get("name") or call.get("name")
                if fn_name:
                    names.append(str(fn_name))

    return names


def _combined_message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += len(_extract_text(msg.get("content", "")))
    return total


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return _extract_text(msg.get("content", ""))
    return ""


def _has_recent_tool_activity(messages: list[dict[str, Any]], lookback: int = 8) -> bool:
    for msg in reversed(messages[-lookback:]):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in {"tool", "function"}:
            return True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return True
    return False


def should_enable_think(
    body: dict[str, Any],
    override: bool | None,
    config: ThinkPolicyConfig,
    default_think: bool = True,
) -> bool:
    if override is not None:
        return override

    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    combined_chars = _combined_message_chars(messages)
    if combined_chars >= config.char_threshold:
        return False

    tool_names = _collect_tool_names(body)
    for tool_name in tool_names:
        if any(_matches_pattern_by_tokens(tool_name, p) for p in config.enable_tool_patterns):
            continue
        if any(_matches_pattern_by_tokens(tool_name, p) for p in config.disable_tool_patterns):
            return False

    last_user = _last_user_text(messages).lower()
    summary_requested = any(pattern in last_user for pattern in config.summary_patterns)
    very_long_turn = combined_chars >= max(1, config.char_threshold // 2)
    if summary_requested and (_has_recent_tool_activity(messages) or very_long_turn):
        return False

    return default_think