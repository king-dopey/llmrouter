from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any

import tokenizer

logger = logging.getLogger("router.headroom")

try:
    from headroom import compress as _headroom_compress  # type: ignore
except ImportError:  # pragma: no cover - exercised through dependency-failure tests
    _headroom_compress = None


@dataclass(frozen=True)
class HeadroomPolicy:
    model: str
    max_num_ctx: int
    reserved_output_tokens: int
    safety_headroom_tokens: int


@dataclass
class HeadroomCheckResult:
    rejected: bool
    trimmed: bool
    messages: list[dict[str, Any]]
    prompt_tokens: int
    usable_prompt_budget: int
    trim_reason: str | None = None
    error_response: dict[str, Any] | None = None
    tokens_before: int = 0
    tokens_after: int = 0
    transforms_applied: list[str] = field(default_factory=list)

def resolve_headroom(model_name: str, policy_entry: dict[str, Any]) -> HeadroomPolicy:
    options = policy_entry.get("options") or {}
    return HeadroomPolicy(
        model=model_name,
        max_num_ctx=int(options.get("num_ctx", 65536)),
        reserved_output_tokens=int(policy_entry.get("reserved_output_tokens", 2048)),
        safety_headroom_tokens=int(policy_entry.get("safety_headroom_tokens", 2048)),
    )


def _usable_prompt_budget(policy: HeadroomPolicy) -> int:
    return max(
        0,
        policy.max_num_ctx
        - policy.reserved_output_tokens
        - policy.safety_headroom_tokens,
    )


def _token_count(messages: list[dict[str, Any]], model: str) -> int:
    return int(tokenizer.count_prompt_tokens(messages, model))


def _extract_query(messages: list[dict[str, Any]]) -> str | None:
    """Extract the user's query from messages for relevance scoring.
    
    Iterates through messages in reverse order to find the last user message.
    Returns the content as a string, or None if extraction fails.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Multi-modal content: extract text parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                if text_parts:
                    return " ".join(text_parts)
            # If content is dict or other type, skip this message
            # and continue looking for a valid user message
    return None


def _normalize_output(
    compressed: Any,
    original: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(compressed, list):
        return compressed
    messages = getattr(compressed, "messages", None)
    if isinstance(messages, list):
        return messages
    if isinstance(compressed, dict) and isinstance(compressed.get("messages"), list):
        return compressed["messages"]
    logger.warning(
        "router.headroom: unexpected compression result type=%s; using original messages",
        type(compressed).__name__,
    )
    return original


def _telemetry(compressed: Any) -> tuple[int, int, list[str]]:
    if isinstance(compressed, dict):
        return (
            int(compressed.get("tokens_before", 0) or 0),
            int(compressed.get("tokens_after", 0) or 0),
            list(compressed.get("transforms_applied", []) or []),
        )
    return (
        int(getattr(compressed, "tokens_before", 0) or 0),
        int(getattr(compressed, "tokens_after", 0) or 0),
        list(getattr(compressed, "transforms_applied", []) or []),
    )


def _deserialize_tool_arguments(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tool call arguments from JSON strings back to dicts for Ollama.
    
    Headroom requires all content as strings, so we serialize arguments before compression.
    This function reverses that serialization so Ollama receives the expected dict format.
    """
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        norm_msg = dict(msg)
        
        # Convert tool call arguments from JSON strings to dicts
        if "tool_calls" in norm_msg and isinstance(norm_msg["tool_calls"], list):
            converted_tool_calls = []
            for tc in norm_msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                
                # If arguments is a JSON string, parse it back to a dict
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                        if isinstance(parsed, (dict, list)):
                            args = parsed
                    except json.JSONDecodeError:
                        logger.warning(
                            "router.headroom: failed to parse tool arguments as JSON: %r",
                            args[:200],
                        )
                
                converted_tc = dict(tc)
                if fn is not None:
                    converted_fn = dict(fn) if isinstance(fn, dict) else {}
                    converted_fn["arguments"] = args
                    converted_tc["function"] = converted_fn
                converted_tool_calls.append(converted_tc)
            norm_msg["tool_calls"] = converted_tool_calls
        
        out.append(norm_msg)
    
    return out

def _sanitize_messages_for_ollama(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove any Headroom-specific metadata from messages before sending to Ollama.
    
    Headroom's compression system may inject internal fields (compression hashes,
    context tracking IDs, etc.) into message objects. These must be stripped before
    sending to Ollama and returning to the client to prevent artifacts in the response.
    """
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        # Only keep standard OpenAI/Ollama message fields
        sanitized_msg = {}
        
        # Required fields
        if "role" in msg:
            sanitized_msg["role"] = msg["role"]
        if "content" in msg:
            sanitized_msg["content"] = msg["content"]
        
        # Optional standard fields
        if "name" in msg:
            sanitized_msg["name"] = msg["name"]
        if "tool_call_id" in msg:
            sanitized_msg["tool_call_id"] = msg["tool_call_id"]
        if "images" in msg:
            sanitized_msg["images"] = msg["images"]
        if "tool_calls" in msg:
            # Keep tool_calls but ensure they're clean
            sanitized_tool_calls = []
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                sanitized_tc = {}
                if "id" in tc:
                    sanitized_tc["id"] = tc["id"]
                if "type" in tc:
                    sanitized_tc["type"] = tc["type"]
                if "function" in tc:
                    fn = tc["function"]
                    if isinstance(fn, dict):
                        sanitized_fn = {}
                        if "name" in fn:
                            sanitized_fn["name"] = fn["name"]
                        if "arguments" in fn:
                            sanitized_fn["arguments"] = fn["arguments"]
                        sanitized_tc["function"] = sanitized_fn
                sanitized_tool_calls.append(sanitized_tc)
            sanitized_msg["tool_calls"] = sanitized_tool_calls
        
        out.append(sanitized_msg)
    
    return out

def _compress(
    messages: list[dict[str, Any]],
    model_name: str,
    policy_entry: dict[str, Any],
    headroom_query: str | None = None,
) -> tuple[list[dict[str, Any]], int, int, list[str]]:
    if _headroom_compress is None:
        raise RuntimeError(
            "HEADROOM_ENABLED=1 but the headroom compression API is unavailable"
        )

    compress_kwargs: dict[str, Any] = {"model": model_name}
    
    if headroom_query is not None:
        compress_kwargs["headroom_query"] = headroom_query
    
    try:
        # Call headroom.compress() with valid parameters only
        result = _headroom_compress(messages, **compress_kwargs)
        
        normalized_compressed = _normalize_output(result, messages)
        
        # Sanitize messages to remove any Headroom-specific metadata
        sanitized_messages = _sanitize_messages_for_ollama(normalized_compressed)
        
        # Deserialize tool arguments back to dicts for Ollama compatibility
        deserialized_messages = _deserialize_tool_arguments(sanitized_messages)
        
        # Extract telemetry from result
        tokens_before = int(getattr(result, "tokens_before", 0) or 0)
        tokens_after = int(getattr(result, "tokens_after", 0) or 0)
        transforms = list(getattr(result, "transforms_applied", []) or [])
        
        logger.info(
            "router.headroom: compression model=%s before=%s after=%s transforms=%s query=%s",
            model_name,
            tokens_before or "unknown",
            tokens_after or "unknown",
            transforms,
            headroom_query[:50] if headroom_query else "none",
        )
        
        return deserialized_messages, tokens_before, tokens_after, transforms
    except TypeError as e:
        logger.warning("router.headroom: TypeError in compress, retrying with minimal params: %s", e)
        try:
            result = _headroom_compress(messages, model=model_name)
            normalized_compressed = _normalize_output(result, messages)
            sanitized_messages = _sanitize_messages_for_ollama(normalized_compressed)
            deserialized_messages = _deserialize_tool_arguments(sanitized_messages)
            return deserialized_messages, int(getattr(result, "tokens_before", 0) or 0), int(getattr(result, "tokens_after", 0) or 0), list(getattr(result, "transforms_applied", []) or [])
        except Exception as e2:
            logger.error("router.headroom: compression failed after fallback: %s", e2)
            raise RuntimeError(f"Headroom compression failed: {e2}") from e2
    except Exception as e:
        logger.error(f"Headroom compression failed with exception: {e}", exc_info=True)
        raise RuntimeError(f"Headroom compression failed: {e}") from e


def check_and_trim(
    messages: list[dict[str, Any]],
    model_name: str,
    policy_entry: dict[str, Any],
) -> HeadroomCheckResult:
    """Compress one finalized prompt and enforce the model's post-compression limit."""
    policy = resolve_headroom(model_name, policy_entry)
    budget = _usable_prompt_budget(policy)
    initial_tokens = _token_count(messages, model_name)

    if not messages:
        return HeadroomCheckResult(
            rejected=False,
            trimmed=False,
            messages=[],
            prompt_tokens=0,
            usable_prompt_budget=budget,
        )

    if os.getenv("HEADROOM_ENABLED", "1").lower() in {"0", "false", "off", "no"}:
        return HeadroomCheckResult(
            rejected=False,
            trimmed=False,
            messages=messages,
            prompt_tokens=initial_tokens,
            usable_prompt_budget=budget,
            tokens_before=initial_tokens,
            tokens_after=initial_tokens,
        )

    # Extract query for relevance scoring if enabled
    headroom_query = None
    if os.getenv("HEADROOM_RELEVANCE_ENABLED", "1").lower() not in {"0", "false", "off", "no"}:
        headroom_query = _extract_query(messages)
        if headroom_query:
            logger.debug("router.headroom: extracted query for relevance scoring: %s", headroom_query[:50])
        else:
            logger.debug("router.headroom: no query extracted, skipping relevance scoring")

    # DIAGNOSTIC LOGGING START
    logger.info(f"HEADROOM_CHECK_AND_TRIM_START model={model_name} message_count={len(messages)}")
    # DIAGNOSTIC LOGGING END

    compressed, tokens_before, tokens_after, transforms = _compress(
        messages,
        model_name,
        policy_entry,
        headroom_query=headroom_query,
    )
    final_tokens = _token_count(compressed, model_name)
    rejected = final_tokens > budget

    logger.info(
        "router.headroom: compression model=%s before=%s after=%s transforms=%s",
        model_name,
        tokens_before or initial_tokens,
        tokens_after or final_tokens,
        transforms,
    )

    error_response = None
    if rejected:
        error_response = {
            "error": {
                "type": "context_length_exceeded",
                "message": "Request exceeds the model context window after compression",
                "details": {
                    "model": model_name,
                    "max_num_ctx": policy.max_num_ctx,
                    "reserved_output_tokens": policy.reserved_output_tokens,
                    "safety_headroom_tokens": policy.safety_headroom_tokens,
                    "usable_prompt_budget": budget,
                    "final_prompt_tokens": final_tokens,
                    "over_by_tokens": final_tokens - budget,
                },
            }
        }

    return HeadroomCheckResult(
        rejected=rejected,
        trimmed=compressed != messages,
        messages=compressed,
        prompt_tokens=final_tokens,
        usable_prompt_budget=budget,
        trim_reason="compressed" if compressed != messages else None,
        error_response=error_response,
        tokens_before=tokens_before or initial_tokens,
        tokens_after=tokens_after or final_tokens,
        transforms_applied=transforms,
    )
