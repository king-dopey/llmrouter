import os
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import yaml

logger = logging.getLogger("router.tokenizer")


try:
    import tiktoken  # type: ignore
    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover - environment may not have tiktoken
    tiktoken = None
    _HAS_TIKTOKEN = False


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # try common keys
        for k in ("text", "content"):
            if isinstance(content.get(k), str):
                return content[k]
            if isinstance(content.get(k), list):
                return "".join(_text_from_content(p) for p in content[k])
        return str(content)
    if isinstance(content, list):
        return "".join(_text_from_content(p) for p in content)
    return str(content)


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            parts.append(str(m))
            continue
        parts.append(_text_from_content(m.get("content")))
    return "\n".join(parts)


def _load_tokenizer_map() -> Dict[str, str]:
    # Preferred: environment variable TOKENIZER_MAP as JSON
    raw = os.getenv("TOKENIZER_MAP")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {k: str(v) for k, v in parsed.items()}
        except Exception:
            logger.warning("router.tokenizer: TOKENIZER_MAP is not valid JSON")

    # Fallback: read router/model_policy.yml tokenizers: section
    policy_path = os.path.join(os.path.dirname(__file__), "model_policy.yml")
    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
        tokenizers = parsed.get("tokenizers")
        if isinstance(tokenizers, dict):
            return {k: str(v) for k, v in tokenizers.items()}
    except Exception:
        pass

    return {}


_TOKENIZER_MAP = _load_tokenizer_map()


def _make_tiktoken_counter(enc_name: str) -> Optional[Callable[[str], int]]:
    if not _HAS_TIKTOKEN:
        logger.debug("router.tokenizer: tiktoken not available; cannot use %s", enc_name)
        return None
    try:
        enc = tiktoken.get_encoding(enc_name)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.exception("router.tokenizer: failed to get tiktoken encoding %s", enc_name)
            return None

    def _count(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text))

    return _count


def _estimate_tokens(text: str) -> int:
    """Return a deterministic, conservative estimate without external side effects."""
    if not text:
        return 0
    byte_count = len(text.encode("utf-8"))
    return max(1, (byte_count + 2) // 3)


def _get_counter_for_model(model: str) -> Callable[[str], int]:
    # If model explicitly maps to the approximate sentinel, use it.
    choice = _TOKENIZER_MAP.get(model)
    if choice is None:
        # Support simple prefix wildcards in map keys, e.g. qwen3*: tiktoken:cl100k_base
        for pat, mapped in _TOKENIZER_MAP.items():
            if pat.endswith("*") and model.startswith(pat[:-1]):
                choice = mapped
                break
    if choice:
        if choice.strip().lower() == "approximate":
            return _estimate_tokens
        if choice.startswith("tiktoken:"):
            enc_name = choice.split(":", 1)[1]
            counter = _make_tiktoken_counter(enc_name)
            if counter:
                return counter
            logger.warning("router.tokenizer: requested tiktoken encoding %s unavailable, falling back to approximate", enc_name)

    # Heuristic: prefer tiktoken if available with a sensible default encoding
    if _HAS_TIKTOKEN:
        counter = _make_tiktoken_counter("cl100k_base")
        if counter:
            return counter

    # final fallback
    return _estimate_tokens


def count_prompt_tokens(messages: List[Dict[str, Any]], model: str) -> int:
    """Count tokens for a list of messages (prompt portion)."""
    text = _messages_to_text(messages)
    counter = _get_counter_for_model(model)
    try:
        return int(counter(text))
    except Exception:
        logger.exception("router.tokenizer: counter failed; falling back to estimation")
        return _estimate_tokens(text)


def count_completion_tokens(text: str, model: str) -> int:
    """Count tokens for completion text."""
    counter = _get_counter_for_model(model)
    try:
        return int(counter(text or ""))
    except Exception:
        logger.exception("router.tokenizer: completion counter failed; falling back to estimation")
        return _estimate_tokens(text or "")


__all__ = ["count_prompt_tokens", "count_completion_tokens"]
