from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any

import tokenizer

from headroom.compress import CompressConfig  # type: ignore

logger = logging.getLogger("router.headroom")

try:
    from headroom import compress as _headroom_compress  # type: ignore
except ImportError:  # pragma: no cover - exercised through dependency-failure tests
    _headroom_compress = None

try:
    from headroom.transforms import (  # type: ignore
        ContentType as _HeadroomContentType,
        LogCompressor as _HeadroomLogCompressor,
        SearchCompressor as _HeadroomSearchCompressor,
        TextCompressor as _HeadroomTextCompressor,
        detect_content_type as _headroom_detect_content_type,
    )
except ImportError:  # pragma: no cover - optional SDK extras
    _HeadroomContentType = None
    _HeadroomLogCompressor = None
    _HeadroomSearchCompressor = None
    _HeadroomTextCompressor = None
    _headroom_detect_content_type = None

# Module-level singleton — system messages must NEVER be compressed.
# This preserves Zoo Code's tool definitions, usage instructions, and protocol.
_HEADROOM_CONFIG = CompressConfig(
    compress_system_messages=False,  # CRITICAL: prevents corrupting tool-calling protocol
    compress_user_messages=False,     # Already default, explicit for clarity
    protect_recent=4,                 # Protect last N turns (already default)
)


def normalize_policy_entry(policy_entry: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize model policy entries for the current runtime.

    Policy fields that do not affect the implemented behavior are stripped
    immediately so the configuration reflects the real system instead of legacy
    names or compatibility aliases.
    """
    if not isinstance(policy_entry, dict):
        return {}

    normalized = dict(policy_entry)
    normalized.pop("allow_auto_pull", None)
    normalized.pop("recovery_strategy", None)
    normalized.pop("trim_strategy", None)

    normalized["proactive_summarization_enabled"] = bool(
        normalized.get("proactive_summarization_enabled", False)
    )
    threshold_value = normalized.get("proactive_summarization_threshold_tokens", 0)
    try:
        normalized["proactive_summarization_threshold_tokens"] = int(threshold_value)
    except (TypeError, ValueError):
        normalized["proactive_summarization_threshold_tokens"] = 0

    return normalized


def _env_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "on", "yes"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


RECOVERY_ENABLE_STRONGER_COMPRESSION = _env_enabled(
    "HEADROOM_RECOVERY_ENABLE_STRONGER_COMPRESSION",
    True,
)
RECOVERY_ENABLE_TEXT_COMPRESSION = _env_enabled(
    "HEADROOM_RECOVERY_ENABLE_TEXT_COMPRESSION",
    True,
)
RECOVERY_ENABLE_SEGMENT_STRIPPING = _env_enabled(
    "HEADROOM_RECOVERY_ENABLE_SEGMENT_STRIPPING",
    True,
)
RECOVERY_MIN_TOKEN_DELTA = _env_int("HEADROOM_RECOVERY_MIN_TOKEN_DELTA", 32)
RECOVERY_MIN_STRIP_TOKEN_DELTA = _env_int("HEADROOM_RECOVERY_MIN_STRIP_TOKEN_DELTA", 1)
RECOVERY_MIN_SEGMENTS_TO_KEEP = _env_int("HEADROOM_RECOVERY_MIN_SEGMENTS_TO_KEEP", 1)

# ---------------------------------------------------------------------------
# Recovery trim strategy (applied only when upstream reports context overflow)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadroomPolicy:
    model: str
    max_num_ctx: int
    reserved_output_tokens: int
    safety_headroom_tokens: int
    proactive_summarization_enabled: bool = False
    proactive_summarization_threshold_tokens: int = 0


@dataclass
class HeadroomCheckResult:
    rejected: bool = False
    trimmed: bool = False
    messages: list[dict[str, Any]] | None = None
    prompt_tokens: int = 0
    usable_prompt_budget: int = 0
    trim_reason: str | None = None
    error_response: dict[str, Any] | None = None
    tokens_before: int = 0
    tokens_after: int = 0
    transforms_applied: list[str] = field(default_factory=list)


@dataclass
class RecoveryAttempt:
    stage: str
    tokens_before: int
    tokens_after: int
    changed: bool
    reason: str


@dataclass
class RecoveryResult:
    messages: list[dict[str, Any]]
    changed: bool
    success_candidate: bool
    attempts: list[RecoveryAttempt]
    final_stage: str


@dataclass
class ConversationSegment:
    start_index: int
    end_index: int
    roles: list[str]
    tool_call_ids: set[str]
    token_estimate: int = 0


def resolve_headroom(model_name: str, policy_entry: dict[str, Any]) -> HeadroomPolicy:
    normalized = normalize_policy_entry(policy_entry)
    options = normalized.get("options") or {}
    return HeadroomPolicy(
        model=model_name,
        max_num_ctx=int(options.get("num_ctx", 65536)),
        reserved_output_tokens=int(normalized.get("reserved_output_tokens", 2048)),
        safety_headroom_tokens=int(normalized.get("safety_headroom_tokens", 2048)),
        proactive_summarization_enabled=bool(normalized.get("proactive_summarization_enabled", False)),
        proactive_summarization_threshold_tokens=int(normalized.get("proactive_summarization_threshold_tokens", 0) or 0),
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


def _validate_tool_integrity(original_messages: list[dict], compressed_messages: list[dict]) -> bool:
    """Verify tool_calls structure survived compression intact.

    Returns True if all messages maintain structural integrity, False otherwise.
    When returning False, callers should fall back to original messages.
    """
    if len(original_messages) != len(compressed_messages):
        return False

    for orig, comp in zip(original_messages, compressed_messages):
        # Assistant messages with tool_calls must still have them
        orig_tc = orig.get("tool_calls")
        comp_tc = comp.get("tool_calls")
        if bool(orig_tc) != bool(comp_tc):
            return False

        # System message content must be unchanged (guaranteed by Fix A, but verify)
        if orig.get("role") == "system":
            if orig.get("content") != comp.get("content"):
                logger.warning(
                    "router.headroom: system message altered during compression; "
                    "falling back to original"
                )
                return False

        # Tool result messages must still have tool_call_id
        if orig.get("role") == "tool":
            if not comp.get("tool_call_id"):
                logger.warning(
                    "router.headroom: tool message lost tool_call_id during compression; "
                    "falling back to original"
                )
                return False

    return True


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


def _preserve_tool_arguments(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve original tool arguments before headroom compression.
    
    Returns a tuple of (messages_for_compression, preserved_arguments_map).
    The preserved_arguments_map maps message indices to their original tool arguments.
    """
    preserved_map: dict[str, Any] = {}
    messages_for_compression = []
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            messages_for_compression.append(msg)
            continue
        
        norm_msg = dict(msg)
        
        if "tool_calls" in norm_msg and isinstance(norm_msg["tool_calls"], list):
            # Store original tool arguments
            original_tool_calls = []
            for tc in norm_msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                original_tool_calls.append({
                    "id": tc.get("id"),
                    "type": tc.get("type"),
                    "function": {
                        "name": fn.get("name"),
                        "arguments": args  # Keep original (dict or string)
                    }
                })
            
            if original_tool_calls:
                preserved_map[f"msg_{i}_tool_calls"] = original_tool_calls
        
        messages_for_compression.append(norm_msg)
    
    return messages_for_compression, preserved_map


def _restore_tool_arguments(
    messages: list[dict[str, Any]],
    preserved_map: dict[str, Any]
) -> list[dict[str, Any]]:
    """Restore original tool arguments after headroom compression.
    
    This ensures that tool arguments are not corrupted by headroom compression.
    """
    restored_messages = []
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            restored_messages.append(msg)
            continue
        
        norm_msg = dict(msg)
        key = f"msg_{i}_tool_calls"
        
        if key in preserved_map:
            # Restore original tool arguments
            norm_msg["tool_calls"] = preserved_map[key]
        
        restored_messages.append(norm_msg)
    
    return restored_messages


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


# ---------------------------------------------------------------------------
# Trim strategy enforcement (recovery path only — Section 8.4)
# ---------------------------------------------------------------------------
# CRITICAL: Tool output messages ("tool" role) are NEVER dropped in isolation.
# Removing them causes infinite tool-call loops because the LLM has no state about
# whether its tool calls succeeded or failed. When dropping a turn that contains
# assistant tool_calls, ALL corresponding "tool" result messages must be removed
# as an atomic unit to prevent orphaned tool_call_id references.


def _collect_tool_ids(message: dict[str, Any]) -> set[str]:
    """Extract all tool_call_ids from an assistant message's tool_calls."""
    ids: set[str] = set()
    for tc in (message.get("tool_calls") or []):
        if isinstance(tc, dict):
            call_id = tc.get("id")
            if call_id:
                ids.add(call_id)
    return ids


def _content_type_name(detection: Any) -> str:
    content_type = getattr(detection, "content_type", None)
    name = getattr(content_type, "name", None)
    if isinstance(name, str):
        return name
    return str(content_type or "")


def _compress_large_text_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply Headroom SDK text compression utilities in linear message order.

    This stage is best-effort and capability-gated: if optional SDK components are
    unavailable, it returns the original messages unchanged.
    """
    if (
        _HeadroomSearchCompressor is None
        or _HeadroomLogCompressor is None
        or _HeadroomTextCompressor is None
        or _headroom_detect_content_type is None
    ):
        return messages

    out: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        norm_msg = dict(msg)
        content = norm_msg.get("content")
        if not isinstance(content, str) or len(content) < 1000:
            out.append(norm_msg)
            continue

        try:
            detection = _headroom_detect_content_type(content)
            ctype_name = _content_type_name(detection)

            if "SEARCH" in ctype_name:
                result = _HeadroomSearchCompressor().compress(content, context="context overflow recovery")
            elif "BUILD" in ctype_name or "LOG" in ctype_name:
                result = _HeadroomLogCompressor().compress(content)
            else:
                result = _HeadroomTextCompressor().compress(content, context="context overflow recovery")

            compressed = getattr(result, "compressed", None)
            if isinstance(compressed, str) and len(compressed) < len(content):
                norm_msg["content"] = compressed
        except Exception:
            # Compression errors are intentionally non-fatal in recovery path.
            pass

        out.append(norm_msg)

    return out


def _material_reduction(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    model_name: str,
    min_delta: int,
) -> tuple[bool, int, int]:
    before_tokens = _token_count(before, model_name)
    after_tokens = _token_count(after, model_name)
    changed = after != before and (before_tokens - after_tokens) >= min_delta
    return changed, before_tokens, after_tokens


def _build_segments(messages: list[dict[str, Any]]) -> list[ConversationSegment]:
    """Create linear, chronological segments anchored at user turns."""
    if not messages:
        return []

    start_idx = 1 if messages and messages[0].get("role") == "system" else 0
    if start_idx >= len(messages):
        return []

    user_indices = [
        i for i in range(start_idx, len(messages))
        if isinstance(messages[i], dict) and messages[i].get("role") == "user"
    ]
    if not user_indices:
        user_indices = [start_idx]

    segments: list[ConversationSegment] = []
    for idx, seg_start in enumerate(user_indices):
        if idx + 1 < len(user_indices):
            seg_end = user_indices[idx + 1] - 1
        else:
            seg_end = len(messages) - 1

        seg_roles: list[str] = []
        seg_tool_ids: set[str] = set()
        for pos in range(seg_start, seg_end + 1):
            msg = messages[pos]
            if not isinstance(msg, dict):
                continue
            seg_roles.append(str(msg.get("role", "")))
            seg_tool_ids.update(_collect_tool_ids(msg))

        segments.append(
            ConversationSegment(
                start_index=seg_start,
                end_index=seg_end,
                roles=seg_roles,
                tool_call_ids=seg_tool_ids,
            )
        )
    return segments


def _strip_oldest_segments(
    messages: list[dict[str, Any]],
    budget: int,
    model_name: str,
    min_segments_to_keep: int = 1,
) -> list[dict[str, Any]]:
    """Last-resort stripping: remove oldest complete segments linearly."""

    def _segment_is_removable(
        all_messages: list[dict[str, Any]],
        segment: ConversationSegment,
    ) -> bool:
        segment_call_ids: set[str] = set()
        segment_tool_result_ids: set[str] = set()

        for idx in range(segment.start_index, segment.end_index + 1):
            msg = all_messages[idx]
            if not isinstance(msg, dict):
                continue
            segment_call_ids.update(_collect_tool_ids(msg))
            if msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id")
                if isinstance(tc_id, str) and tc_id:
                    segment_tool_result_ids.add(tc_id)

        # If this segment contains tool results for call IDs not created inside
        # the segment, removing it would orphan assistant tool_calls elsewhere.
        if any(tc_id not in segment_call_ids for tc_id in segment_tool_result_ids):
            return False

        # If call IDs created in this segment have tool results outside the segment,
        # removing it would orphan those tool messages.
        for idx, msg in enumerate(all_messages):
            if segment.start_index <= idx <= segment.end_index:
                continue
            if not isinstance(msg, dict) or msg.get("role") != "tool":
                continue
            tc_id = msg.get("tool_call_id")
            if isinstance(tc_id, str) and tc_id in segment_call_ids:
                return False

        return True

    result = list(messages)
    while _token_count(result, model_name) > budget:
        segments = _build_segments(result)
        if len(segments) <= min_segments_to_keep:
            break

        oldest = segments[0]
        if not _segment_is_removable(result, oldest):
            # Strict oldest-first behavior: do not skip ahead.
            break
        del result[oldest.start_index:oldest.end_index + 1]

    return result


def _compress_with_config(
    messages: list[dict[str, Any]],
    model_name: str,
    config: CompressConfig,
) -> tuple[list[dict[str, Any]], int, int, list[str]]:
    if _headroom_compress is None:
        raise RuntimeError(
            "HEADROOM_ENABLED=1 but the headroom compression API is unavailable"
        )

    messages_for_compression, preserved_tool_args = _preserve_tool_arguments(messages)

    try:
        result = _headroom_compress(
            messages_for_compression,
            model=model_name,
            config=config,
        )

        normalized_compressed = _normalize_output(result, messages_for_compression)
        sanitized_messages = _sanitize_messages_for_ollama(normalized_compressed)
        restored_messages = _restore_tool_arguments(sanitized_messages, preserved_tool_args)
        deserialized_messages = _deserialize_tool_arguments(restored_messages)

        tokens_before = int(getattr(result, "tokens_before", 0) or 0)
        tokens_after = int(getattr(result, "tokens_after", 0) or 0)
        transforms = list(getattr(result, "transforms_applied", []) or [])

        if not _validate_tool_integrity(messages_for_compression, deserialized_messages):
            logger.warning(
                "router.headroom: tool_calls integrity check FAILED; "
                "falling back to original uncompressed messages"
            )
            return messages, tokens_before, tokens_before, ["integrity_guard:fallback"]

        return deserialized_messages, tokens_before, tokens_after, transforms
    except TypeError as e:
        logger.warning("router.headroom: TypeError in compress, retrying with minimal params: %s", e)
        try:
            result = _headroom_compress(messages_for_compression, model=model_name)
            normalized_compressed = _normalize_output(result, messages_for_compression)
            sanitized_messages = _sanitize_messages_for_ollama(normalized_compressed)
            restored_messages = _restore_tool_arguments(sanitized_messages, preserved_tool_args)
            deserialized_messages = _deserialize_tool_arguments(restored_messages)
            return deserialized_messages, int(getattr(result, "tokens_before", 0) or 0), int(getattr(result, "tokens_after", 0) or 0), list(getattr(result, "transforms_applied", []) or [])
        except Exception as e2:
            logger.error("router.headroom: compression failed after fallback: %s", e2)
            raise RuntimeError(f"Headroom compression failed: {e2}") from e2
    except Exception as e:
        logger.error(f"Headroom compression failed with exception: {e}", exc_info=True)
        raise RuntimeError(f"Headroom compression failed: {e}") from e


def apply_overflow_recovery(
    messages: list[dict[str, Any]],
    model_name: str,
    policy_entry: dict[str, Any],
) -> RecoveryResult:
    """Run the single real overflow recovery flow.

    The router always starts with non-dropping compression steps and only falls
    through to segment stripping if every safe reduction step still leaves the
    request above budget.
    """
    policy = resolve_headroom(model_name, policy_entry)
    budget = _usable_prompt_budget(policy)

    current = list(messages)
    attempts: list[RecoveryAttempt] = []
    changed_any = False
    final_stage = "none"

    current_tokens = _token_count(current, model_name)
    if current_tokens <= budget:
        return RecoveryResult(
            messages=current,
            changed=False,
            success_candidate=True,
            attempts=[
                RecoveryAttempt(
                    stage="precheck",
                    tokens_before=current_tokens,
                    tokens_after=current_tokens,
                    changed=False,
                    reason="already_within_budget",
                )
            ],
            final_stage="precheck",
        )

    if RECOVERY_ENABLE_STRONGER_COMPRESSION and _headroom_compress is not None:
        stronger_profiles = [
            CompressConfig(compress_system_messages=False, compress_user_messages=False, protect_recent=2),
            CompressConfig(compress_system_messages=False, compress_user_messages=True, protect_recent=1),
        ]
        for idx, profile in enumerate(stronger_profiles, start=1):
            candidate, _, _, _ = _compress_with_config(current, model_name, profile)
            stage_changed, before_tokens, after_tokens = _material_reduction(
                current,
                candidate,
                model_name,
                RECOVERY_MIN_TOKEN_DELTA,
            )
            attempts.append(
                RecoveryAttempt(
                    stage=f"stronger_headroom_compression_{idx}",
                    tokens_before=before_tokens,
                    tokens_after=after_tokens,
                    changed=stage_changed,
                    reason="material_reduction" if stage_changed else "below_min_delta_or_no_change",
                )
            )
            if not stage_changed:
                continue

            current = candidate
            changed_any = True
            final_stage = f"stronger_headroom_compression_{idx}"
            if after_tokens <= budget:
                return RecoveryResult(
                    messages=current,
                    changed=changed_any,
                    success_candidate=True,
                    attempts=attempts,
                    final_stage=final_stage,
                )

    if RECOVERY_ENABLE_TEXT_COMPRESSION:
        candidate = _compress_large_text_blocks(current)
        stage_changed, before_tokens, after_tokens = _material_reduction(
            current,
            candidate,
            model_name,
            RECOVERY_MIN_TOKEN_DELTA,
        )
        attempts.append(
            RecoveryAttempt(
                stage="large_text_block_compression",
                tokens_before=before_tokens,
                tokens_after=after_tokens,
                changed=stage_changed,
                reason="material_reduction" if stage_changed else "sdk_unavailable_or_below_min_delta",
            )
        )
        if stage_changed:
            current = candidate
            changed_any = True
            final_stage = "large_text_block_compression"
            if after_tokens <= budget:
                return RecoveryResult(
                    messages=current,
                    changed=changed_any,
                    success_candidate=True,
                    attempts=attempts,
                    final_stage=final_stage,
                )

    if RECOVERY_ENABLE_SEGMENT_STRIPPING:
        candidate = _strip_oldest_segments(
            current,
            budget,
            model_name,
            min_segments_to_keep=max(1, RECOVERY_MIN_SEGMENTS_TO_KEEP),
        )
        stage_changed, before_tokens, after_tokens = _material_reduction(
            current,
            candidate,
            model_name,
            RECOVERY_MIN_STRIP_TOKEN_DELTA,
        )
        attempts.append(
            RecoveryAttempt(
                stage="last_resort_segment_stripping",
                tokens_before=before_tokens,
                tokens_after=after_tokens,
                changed=stage_changed,
                reason="last_resort_strip" if stage_changed else "no_removable_segments",
            )
        )
        if stage_changed:
            current = candidate
            changed_any = True
            final_stage = "last_resort_segment_stripping"

    return RecoveryResult(
        messages=current,
        changed=changed_any,
        success_candidate=_token_count(current, model_name) <= budget,
        attempts=attempts,
        final_stage=final_stage or "none",
    )


def _compress(
    messages: list[dict[str, Any]],
    model_name: str,
    policy_entry: dict[str, Any],
) -> tuple[list[dict[str, Any]], int, int, list[str]]:
    return _compress_with_config(messages, model_name, _HEADROOM_CONFIG)


@dataclass
class ProactiveSummaryResult:
    triggered: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    reason: str | None = None


def maybe_apply_proactive_summarization(
    messages: list[dict[str, Any]],
    model_name: str,
    policy_entry: dict[str, Any],
) -> ProactiveSummaryResult:
    """Run a non-dropping proactive compression/summarization pass before upstream calls.

    The threshold is evaluated against the prompt token count before generation and
    uses the same input-budget logic as the router context budget:
      effective_input_budget = num_ctx - reserved_output_tokens - safety_headroom_tokens
    If the current prompt size reaches the configured threshold, this function runs
    stronger Headroom-native compression stages without any segment stripping or
    message deletion.
    """
    normalized = normalize_policy_entry(policy_entry)
    if not bool(normalized.get("proactive_summarization_enabled", False)):
        return ProactiveSummaryResult(
            triggered=False,
            messages=list(messages),
            tokens_before=_token_count(messages, model_name),
            tokens_after=_token_count(messages, model_name),
            reason="disabled",
        )

    threshold_tokens = int(normalized.get("proactive_summarization_threshold_tokens", 0) or 0)
    if threshold_tokens <= 0:
        return ProactiveSummaryResult(
            triggered=False,
            messages=list(messages),
            tokens_before=_token_count(messages, model_name),
            tokens_after=_token_count(messages, model_name),
            reason="threshold_not_configured",
        )

    current_tokens = _token_count(messages, model_name)
    if current_tokens < threshold_tokens:
        return ProactiveSummaryResult(
            triggered=False,
            messages=list(messages),
            tokens_before=current_tokens,
            tokens_after=current_tokens,
            reason="below_threshold",
        )

    current = list(messages)
    attempts: list[str] = []
    before_tokens = current_tokens

    stronger_config = CompressConfig(
        compress_system_messages=False,
        compress_user_messages=False,
        protect_recent=2,
    )
    candidate, _, _, transforms = _compress_with_config(current, model_name, stronger_config)
    attempts.extend(transforms)
    changed, _, after_tokens = _material_reduction(current, candidate, model_name, 1)
    if changed:
        current = candidate

    text_candidate = _compress_large_text_blocks(current)
    changed_text, _, after_text_tokens = _material_reduction(current, text_candidate, model_name, 1)
    if changed_text:
        current = text_candidate
        attempts.extend(["stage_c_text_compression"])
    else:
        attempts.extend(["stage_c_text_compression:no_material_change"])

    if current == messages:
        return ProactiveSummaryResult(
            triggered=True,
            messages=list(messages),
            tokens_before=before_tokens,
            tokens_after=before_tokens,
            reason="threshold_reached_no_material_reduction",
        )

    return ProactiveSummaryResult(
        triggered=True,
        messages=current,
        tokens_before=before_tokens,
        tokens_after=_token_count(current, model_name),
        reason="threshold_reached",
    )


def check_and_trim(
    messages: list[dict[str, Any]],
    model_name: str,
    policy_entry: dict[str, Any],
) -> HeadroomCheckResult:
    """Compress one finalized prompt and enforce the model's post-compression limit.

    Compaction (headroom.compress) runs on EVERY request unconditionally — this is
    HEADROOM's core purpose. Overflow recovery is a separate bounded retry path
    triggered only when upstream Ollama returns a context-size error.
    """
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
            messages=_deserialize_tool_arguments(messages),
            prompt_tokens=initial_tokens,
            usable_prompt_budget=budget,
            tokens_before=initial_tokens,
            tokens_after=initial_tokens,
        )

    logger.info(
        "router.headroom: check_and_trim model=%s message_count=%d",
        model_name, len(messages),
    )

    # ALWAYS compress — compaction is unconditional (HEADROOM's core purpose)
    compressed, tokens_before, tokens_after, transforms = _compress(
        messages,
        model_name,
        policy_entry,
    )
    final_tokens = _token_count(compressed, model_name)

    logger.info(
        "router.headroom: compression model=%s before=%s after=%s transforms=%s",
        model_name,
        tokens_before or initial_tokens,
        tokens_after or final_tokens,
        transforms,
    )

    return HeadroomCheckResult(
        rejected=False,  # Never reject at compaction stage — recovery handles overflow
        trimmed=compressed != messages,
        messages=compressed,
        prompt_tokens=final_tokens,
        usable_prompt_budget=budget,
        trim_reason="compressed" if compressed != messages else None,
        error_response=None,  # No error response here — upstream errors handled in app.py
        tokens_before=tokens_before or initial_tokens,
        tokens_after=tokens_after or final_tokens,
        transforms_applied=transforms,
    )


def is_context_overflow_error(status_code: int, response_text: str) -> bool:
    """Detect if an upstream error is a context-length overflow.

    Ollama returns HTTP 400 with messages like:
    - 'context length exceeded'
    - 'prompt too long'
    - 'input length exceeds'
    
    Returns True when the error indicates context overflow, False otherwise.
    """
    if status_code != 400:
        return False
    text_lower = response_text.lower()
    indicators = ("context length", "prompt too long", "input length exceeds", "context_length")
    return any(ind in text_lower for ind in indicators)
