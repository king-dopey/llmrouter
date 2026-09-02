import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CHAR_THRESHOLD = 12000


@dataclass(frozen=True)
class ThinkPolicyConfig:
    char_threshold: int


def load_think_policy_config() -> ThinkPolicyConfig:
    threshold_raw = os.getenv("DISABLE_THINK_CHAR_THRESHOLD", str(DEFAULT_CHAR_THRESHOLD)).strip()
    try:
        char_threshold = max(1, int(threshold_raw))
    except ValueError:
        char_threshold = DEFAULT_CHAR_THRESHOLD

    return ThinkPolicyConfig(char_threshold=char_threshold)


def parse_think_override(header_value: str | None) -> str | None:
    """Parse the X-Ollama-Think header value.
    
    Args:
        header_value: The header value to parse
        
    Returns:
        str | None: One of "true", "false", "tool_call_true", or None if header not present
        
    Raises:
        ValueError: If the header value is not one of the valid options
    """
    if header_value is None:
        return None

    value = header_value.strip().lower()
    if value in {"true", "false", "tool_call_true"}:
        return value
    raise ValueError("X-Ollama-Think must be 'true', 'false', or 'tool_call_true'")


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


def _combined_message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += len(_extract_text(msg.get("content", "")))
    return total


def _detect_tool_call_failure(
    assistant_content: str,
    tool_names: list[str],
    has_structured_tool_calls: bool
) -> bool:
    """Detect when assistant text mentions a tool but no structured tool_calls present.
    
    Returns True if:
    - has_structured_tool_calls is False AND
    - assistant_content mentions any tool name from tool_names
    
    Args:
        assistant_content: The assistant's text response
        tool_names: List of available tool names from the request
        has_structured_tool_calls: Whether structured tool_calls were returned
    
    Returns:
        bool: True if failure detected (tool mentioned but no structured calls)
    """
    if has_structured_tool_calls:
        return False
    
    if not assistant_content or not tool_names:
        return False
    
    # Check if any tool name appears in assistant content
    content_lower = assistant_content.lower()
    for tool_name in tool_names:
        if tool_name.lower() in content_lower:
            return True
    
    return False


def should_enable_think(
    body: dict[str, Any],
    override: str | None,
    config: ThinkPolicyConfig,
    default_think: bool = True,
    *,
    model_name: str = "",
    tool_call_think: bool | None = None,
) -> bool:
    """Determine whether thinking should be enabled for this request.
    
    Rule hierarchy (evaluated top-to-bottom, first match wins):
    1. Model with tool_call_think: false + tools present → think: false (SAFETY INVARIANT — NO OVERRIDES ALLOWED)
    2. Model with tools present:
       a. X-Ollama-Think: tool_call_true → think: true
       b. X-Ollama-Think: true/false → think: false (deny by default)
       c. Explicit tool_call_think policy exists → use it
       d. No explicit policy → think: false (FAIL-SAFE)
    3. No tools present → apply char threshold heuristic, then default
    
    Args:
        body: The request body containing messages and tools
        override: Client override from X-Ollama-Think header ("true", "false", "tool_call_true", or None)
        config: ThinkPolicyConfig with char_threshold
        default_think: Default think value when no rules match
        model_name: The model name from the request (keyword-only, unused but kept for API compatibility)
        tool_call_think: Model's tool_call_think config value (keyword-only)
        
    Returns:
        bool: Whether thinking should be enabled for this request
    """
    has_tools = bool(body.get("tools"))
    
    # Rule 1: Models with tool_call_think: false and tools present — SAFETY INVARIANT, NO OVERRIDES
    if has_tools and tool_call_think is False:
        logger.warning(
            "Safety invariant enforced: think disabled for tool request",
            extra={
                "event_type": "safety_invariant",
                "model": model_name,
                "override": override,
                "tool_call_think": tool_call_think,
                "decision": False,
                "reason": "safety_invariant",
                "has_tools": has_tools
            }
        )
        return False
    
    # Rule 2: Models with tools present — explicit policy only, fail-safe to false
    if has_tools:
        # Check for valid override value
        if override == "tool_call_true":
            logger.info(
                "Valid override accepted: tool_call_true",
                extra={
                    "event_type": "override_accepted",
                    "model": model_name,
                    "override": override,
                    "decision": True,
                    "reason": "valid_override"
                }
            )
            return True
        elif override is not None:
            # Invalid override value with tools → deny by default
            logger.warning(
                "Invalid override denied for tool request",
                extra={
                    "event_type": "override_denied",
                    "model": model_name,
                    "override": override,
                    "decision": False,
                    "reason": "invalid_override"
                }
            )
            return False
        
        # No override: use explicit policy or fail-safe
        if tool_call_think is not None:
            logger.info(
                "Explicit policy applied",
                extra={
                    "event_type": "policy_applied",
                    "model": model_name,
                    "tool_call_think": tool_call_think,
                    "decision": tool_call_think,
                    "reason": "explicit_policy"
                }
            )
            return tool_call_think
        else:
            # No explicit policy → fail-safe to false
            logger.warning(
                "Fail-safe activated: no explicit policy for tool request",
                extra={
                    "event_type": "fail_safe",
                    "model": model_name,
                    "decision": False,
                    "reason": "no_explicit_policy"
                }
            )
            return False
    
    # Rule 3: No tools present — apply char threshold heuristic, then default
    if override is not None:
        decision = override == "true"
        logger.info(
            "Override applied for non-tool request",
            extra={
                "event_type": "override_applied",
                "model": model_name,
                "override": override,
                "decision": decision,
                "reason": "client_override"
            }
        )
        return decision
    
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        messages = []
    
    combined_chars = _combined_message_chars(messages)
    if combined_chars >= config.char_threshold:
        logger.info(
            "Character threshold exceeded",
            extra={
                "event_type": "threshold_exceeded",
                "model": model_name,
                "combined_chars": combined_chars,
                "threshold": config.char_threshold,
                "decision": False,
                "reason": "char_threshold"
            }
        )
        return False
    
    logger.debug(
        "Default think value applied",
        extra={
            "event_type": "default_applied",
            "model": model_name,
            "decision": default_think,
            "reason": "default"
        }
    )
    return default_think