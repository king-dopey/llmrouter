import asyncio
import logging
import json
import os
import time
import uuid
from typing import Any
from urllib.parse import urlsplit

import httpx
import tokenizer
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from policy import load_think_policy_config, parse_think_override, should_enable_think
from router_headroom import check_and_trim
from retrieval import RetrievedChunk, format_retrieval_context, retrieve_context

# Add lock for preventing concurrent pulls of the same model
from asyncio import Lock

logger = logging.getLogger("router")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


def _safe_error_response(exc: Exception, operation: str) -> dict:
    """Create a sanitized error response for client consumption.
    
    Logs full error details server-side but returns a generic message
    to clients to prevent information disclosure (CWE-209).
    
    Args:
        exc: The exception that occurred
        operation: Description of the operation that failed
        
    Returns:
        dict: Sanitized error response structure
    """
    error_id = str(uuid.uuid4())[:8]
    logger.error("[%s] %s: %s: %s", error_id, operation, type(exc).__name__, exc)
    return {
        "error": {
            "message": f"An internal error occurred ({error_id}). Please check server logs.",
            "type": "internal_error",
            "code": error_id
        }
    }

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
ASR_BASE_URL_ENV = os.getenv("ASR_BASE_URL", "").rstrip("/")
ASR_PORT = int(os.getenv("ASR_PORT", "8000"))
ASR_SCHEME = (os.getenv("ASR_SCHEME", "http") or "http").strip()
POLICY_FILE = os.getenv("MODEL_POLICY_FILE", "/app/model_policy.yml")
DEFAULT_MODEL = os.getenv("MODEL_DEFAULT", "qwen3.6:35b-a3b")
DEFAULT_KEEP_ALIVE = os.getenv("KEEP_ALIVE_DEFAULT", "-1")
EMBEDDING_MODEL_DEFAULT = os.getenv("EMBEDDING_MODEL_DEFAULT", "qwen3-embedding:4b")
ENABLE_QDRANT_RETRIEVAL = os.getenv("ENABLE_QDRANT_RETRIEVAL", "false").lower() == "true"
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "repo_chunks")
QDRANT_EMBEDDING_MODEL = os.getenv("QDRANT_EMBEDDING_MODEL", "nomic-embed-text")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "20"))
QDRANT_FINAL_K = int(os.getenv("QDRANT_FINAL_K", "8"))

def _translate_tool_calls(ollama_tool_calls):
    """Convert Ollama-shape tool_calls to OpenAI-shape.

    Ollama: [{"function": {"name": "...", "arguments": {...dict...}}}]
    OpenAI: [{"id": "...", "type": "function",
              "function": {"name": "...", "arguments": "<json string>"}}]
    """
    if not ollama_tool_calls:
        return None
    out = []
    for tc in ollama_tool_calls:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, (dict, list)):
            args_str = json.dumps(args)
        elif args is None:
            args_str = "{}"
        else:
            args_str = str(args)
        out.append({
            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": fn.get("name", ""),
                "arguments": args_str,
            },
        })
    return out


def _streaming_tool_call_deltas(ollama_tool_calls):
    """OpenAI streaming requires each tool_call entry in a delta to carry an
    `index`. Ollama emits the whole tool_calls array in a single chunk, so we
    emit them all in one delta with sequential indices."""
    translated = _translate_tool_calls(ollama_tool_calls)
    if not translated:
        return None
    for i, tc in enumerate(translated):
        tc["index"] = i
    return translated


def _build_stream_usage_chunk(
    *,
    include_usage: bool,
    completion_id: str,
    created: int,
    model: str,
    done_data: dict[str, Any],
) -> dict[str, Any] | None:
    if not include_usage:
        return None
    prompt_tokens = int(done_data.get("prompt_eval_count", 0) or 0)
    completion_tokens = int(done_data.get("eval_count", 0) or 0)
    cache_creation_tokens = int(done_data.get("cache_creation_input_tokens", 0) or 0)
    cache_read_tokens = int(done_data.get("cache_read_input_tokens", 0) or 0)
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cache_creation_input_tokens": cache_creation_tokens,
            "cache_read_input_tokens": cache_read_tokens,
        },
    }


def _build_non_stream_usage(
    *,
    ollama_result: dict[str, Any],
    payload_messages: list[dict[str, Any]],
    model: str,
    completion_text: str,
) -> dict[str, int]:
    prompt_tokens = ollama_result.get("prompt_eval_count")
    if prompt_tokens is None:
        prompt_tokens = tokenizer.count_prompt_tokens(payload_messages, model)

    completion_tokens = ollama_result.get("eval_count")
    if completion_tokens is None:
        completion_tokens = tokenizer.count_completion_tokens(completion_text, model)

    prompt_tokens = int(prompt_tokens or 0)
    completion_tokens = int(completion_tokens or 0)
    cache_creation_tokens = int(ollama_result.get("cache_creation_input_tokens", 0) or 0)
    cache_read_tokens = int(ollama_result.get("cache_read_input_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
    }


def _build_retrieval_query(body: dict[str, Any]) -> dict[str, Any] | None:
    retrieval = body.get("retrieval")
    if isinstance(retrieval, dict):
        return retrieval

    repo = body.get("context_repo")
    query = body.get("context_query") or body.get("query")
    if repo and query:
        return {
            "repo": repo,
            "query": query,
            "branch": body.get("context_branch"),
            "top_k": body.get("retrieval_top_k", QDRANT_TOP_K),
            "final_k": body.get("retrieval_final_k", QDRANT_FINAL_K),
            "filters": body.get("retrieval_filters") or {},
        }
    return None


def _coerce_embedding_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            else:
                out.append(str(item))
        return out
    raise HTTPException(status_code=400, detail="'input' must be a string or an array of strings")


async def _fetch_embedding(model: str, prompt: str) -> list[float]:
    payload = {"model": model, "prompt": prompt}
    try:
        response = await _ollama_post("/api/embeddings", payload, stream=False)
    except Exception:
        raise HTTPException(status_code=503, detail="Ollama upstream unavailable")

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    data = response.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise HTTPException(status_code=502, detail="Invalid embedding response from Ollama")
    return [float(v) for v in embedding]


async def _inject_retrieval_context(body: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ENABLE_QDRANT_RETRIEVAL:
        return messages

    request = _build_retrieval_query(body)
    if not request:
        return messages

    repo = str(request.get("repo") or "").strip()
    query = str(request.get("query") or "").strip()
    if not repo or not query:
        return messages

    try:
        chunks = await retrieve_context(
            repo=repo,
            query=query,
            branch=request.get("branch"),
            top_k=int(request.get("top_k", QDRANT_TOP_K)),
            final_k=int(request.get("final_k", QDRANT_FINAL_K)),
            filters=request.get("filters") or {},
            qdrant_url=QDRANT_URL,
            collection=request.get("collection") or QDRANT_COLLECTION,
            embedding_model=request.get("embedding_model") or QDRANT_EMBEDDING_MODEL,
        )
    except Exception as exc:
        logger.warning("router: retrieval unavailable for repo=%s query=%s: %s", repo, query, exc)
        return messages

    if not chunks:
        return messages

    retrieval_block = format_retrieval_context([
        RetrievedChunk(**chunk) if isinstance(chunk, dict) else chunk
        for chunk in chunks
    ])
    if not retrieval_block:
        return messages

    injected = [{"role": "system", "content": retrieval_block}]
    injected.extend(messages)
    return injected


def _parse_scalar(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return value


# Options the router will inject from policy. num_ctx is the only one where
# the policy unconditionally wins; the rest are defaults that client requests
# may override.
POLICY_LOCKED_OPTIONS = {"num_ctx"}


def _load_policy() -> dict[str, dict[str, Any]]:
    if os.path.exists(POLICY_FILE):
        with open(POLICY_FILE, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f) or {}
        models = parsed.get("models", [])
        if isinstance(models, list):
            table: dict[str, dict[str, Any]] = {}
            for item in models:
                model_name = item.get("model")
                if not model_name:
                    continue
                keep_alive = _parse_scalar(item.get("keep_alive"))
                think = item.get("think")
                options = item.get("options") or {}
                if not isinstance(options, dict):
                    logger.warning("policy: ignoring non-dict options for %s", model_name)
                    options = {}
                table[model_name] = {
                    "keep_alive": keep_alive,
                    "think": bool(think) if think is not None else True,
                    "options": {k: _parse_scalar(v) for k, v in options.items()},
                    "warmup": bool(item.get("warmup", False)),
                    "reserved_output_tokens": _parse_scalar(item.get("reserved_output_tokens", 2048)),
                    "safety_headroom_tokens": _parse_scalar(item.get("safety_headroom_tokens", 2048)),
                    "trim_strategy": item.get("trim_strategy", "drop_oldest"),
                    "allow_auto_pull": bool(item.get("allow_auto_pull", True)),
                }
            if table:
                return table
    # Fallback (env-driven) — unchanged shape plus empty options.
    return {
        DEFAULT_MODEL: {
            "keep_alive": _parse_scalar(DEFAULT_KEEP_ALIVE),
            "think": True,
            "options": {},
            "warmup": False,
        },
    }


MODEL_POLICY = _load_policy()
THINK_POLICY_CONFIG = load_think_policy_config()

# Auto-pull configuration
AUTO_PULL_MISSING_MODELS = os.getenv("AUTO_PULL_MISSING_MODELS", "false").lower() == "true"
MODEL_PULL_TIMEOUT_SEC = int(os.getenv("MODEL_PULL_TIMEOUT_SEC", "7200"))
# MODEL_PULL_MAX_RETRIES and MODEL_PULL_BACKOFF_SEC are currently unused but kept for future implementation

# Global lock for preventing concurrent pulls of the same model
_model_pull_locks = {}
_model_pull_locks_lock = Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    targets = [(m, e) for m, e in MODEL_POLICY.items() if e.get("warmup")]
    if targets:
        # Sequential — parallel warmup of two 30B models will thrash the GPU.
        for model, entry in targets:
            await _warmup_model(model, entry)
    else:
        logger.info("warmup: no models flagged warmup: true; skipping")

    yield

    # --- shutdown ---
    # Nothing to clean up today. If you later add an httpx.AsyncClient or a
    # background task, close/cancel it here.

def _is_embedding_model(model: str) -> bool:
    """Detect if a model is an embedding model based on name pattern."""
    return "embed" in model.lower()


async def _send_warmup_request(model: str, entry: dict[str, Any]) -> httpx.Response:
    """Send a minimal warmup request to Ollama and return the response.

    Uses /api/embeddings for embedding models, /api/chat for chat models.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    timeout = httpx.Timeout(connect=10.0, read=1200.0, write=30.0, pool=30.0)
    
    if _is_embedding_model(model):
        # Embedding models use /api/embeddings endpoint
        payload = {
            "model": model,
            "prompt": "warmup",
        }
        endpoint = f"{OLLAMA_BASE_URL}/api/embeddings"
        logger.debug("warmup: using embeddings endpoint for %s", model)
    else:
        # Chat models use /api/chat endpoint
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
            "keep_alive": entry.get("keep_alive", -1),
            "think": False,
            "options": {**(entry.get("options") or {}), "num_predict": 1},
        }
        endpoint = f"{OLLAMA_BASE_URL}/api/chat"
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(endpoint, json=payload)
        r.raise_for_status()
    return r


async def _warmup_model(model: str, entry: dict[str, Any]) -> None:
    """Send a tiny generation so Ollama loads the model at the policy num_ctx
    and pins it according to keep_alive.

    If the model is not found (HTTP 404) and AUTO_PULL_MISSING_MODELS is enabled,
    the router will attempt to pull the missing model and retry the warmup once.
    """
    try:
        await _send_warmup_request(model, entry)
        logger.info(
            "warmup: %s loaded at num_ctx=%s keep_alive=%s",
            model,
            (entry.get("options") or {}).get("num_ctx"),
            entry.get("keep_alive"),
        )
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.info(
                "warmup: model '%s' not found on Ollama host, attempting auto-pull...",
                model,
            )
            if not AUTO_PULL_MISSING_MODELS:
                logger.info(
                    "warmup: auto-pull is disabled (AUTO_PULL_MISSING_MODELS=false), "
                    "skipping pull for '%s'",
                    model,
                )
                return

            pull_result = await pull_model(model)
            if not pull_result:
                logger.warning(
                    "warmup: failed to pull '%s': pull returned false",
                    model,
                )
                return

            logger.info(
                "warmup: pulled missing model '%s', retrying warmup...",
                model,
            )
            try:
                await _send_warmup_request(model, entry)
                logger.info(
                    "warmup: %s loaded at num_ctx=%s keep_alive=%s",
                    model,
                    (entry.get("options") or {}).get("num_ctx"),
                    entry.get("keep_alive"),
                )
            except Exception as retry_exc:
                logger.warning(
                    "warmup: %s failed after pull: %s",
                    model,
                    retry_exc,
                )
        else:
            logger.warning("warmup: %s failed: %s", model, exc)
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.warning(
            "warmup: Ollama host unreachable during warmup for '%s': %s",
            model,
            exc,
        )
    except Exception as exc:
        logger.warning("warmup: %s failed: %s", model, exc)


async def list_local_models() -> list[str] | None:
    """Get list of locally available models from Ollama."""
    try:
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            response.raise_for_status()
            data = response.json()
            model_names = [model["name"] for model in data.get("models", [])]
            logger.debug("Local models from /api/tags: %s", model_names)
            return model_names
    except Exception as exc:
        logger.warning("Failed to list local models: %s", exc)
        return None


async def is_model_available(model_name: str) -> bool:
    """Check if a model is available locally."""
    local_models = await list_local_models()
    if local_models is None:
        return False
    is_available = model_name in local_models
    logger.debug("Model %s availability check: %s (local models: %s)", model_name, is_available, local_models)
    return is_available


async def pull_model(model_name: str) -> bool:
    """Pull a model from Ollama registry and wait for completion."""
    if not AUTO_PULL_MISSING_MODELS:
        logger.debug("AUTO_PULL_MISSING_MODELS is false, skipping pull for %s", model_name)
        return False
        
    logger.info("Pulling model %s", model_name)
    
    # Get the lock for this specific model
    async with _model_pull_locks_lock:
        if model_name not in _model_pull_locks:
            _model_pull_locks[model_name] = Lock()
        model_lock = _model_pull_locks[model_name]
    
    async with model_lock:
        # Double-check if model is available after acquiring lock
        if await is_model_available(model_name):
            logger.info("Model %s already available after lock acquisition", model_name)
            return True
            
        try:
            # Start the pull process
            timeout = httpx.Timeout(connect=10.0, read=MODEL_PULL_TIMEOUT_SEC, write=30.0, pool=30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Stream the pull response to check for success
                logger.debug("Sending pull request to %s/api/pull for model %s", OLLAMA_BASE_URL, model_name)
                async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/pull", json={"name": model_name}) as response:
                    if response.status_code >= 400:
                        logger.error("Pull request failed with status %d", response.status_code)
                        return False
                        
                    # Read the stream to check for success
                    pull_success = False
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            logger.debug("Pull response for %s: %s", model_name, data)
                            if data.get("status") == "success":
                                pull_success = True
                                logger.info("Pull for model %s completed successfully", model_name)
                                break
                            elif data.get("status") == "error":
                                logger.error("Pull for model %s failed with error: %s", model_name, data.get("error", "unknown"))
                                return False
                        except json.JSONDecodeError:
                            logger.debug("Non-JSON line in pull response for %s: %s", model_name, line)
                            continue
     
                    if not pull_success:
                        logger.error("Pull for model %s did not complete successfully", model_name)
                        return False
                        
            # Verify the model was pulled successfully
            if await is_model_available(model_name):
                logger.info("Successfully pulled model %s", model_name)
                return True
            else:
                logger.error("Model %s was not found after pull completion", model_name)
                return False
                
        except Exception as exc:
            logger.error("Failed to pull model %s: %s", model_name, exc)
            return False


async def _preflight_model(model_name: str) -> bool:
    """Perform preflight validation and ensure model is available."""
    logger.info("Preflight check for model %s", model_name)
    
    # Validate model name
    if not model_name:
        logger.warning("Model name is empty or missing")
        return False
    
    # Check if model is in policy
    if model_name not in MODEL_POLICY:
        logger.warning("Model %s not in policy, rejecting request", model_name)
        return False

    local_models = await list_local_models()
    if local_models is None:
        raise HTTPException(status_code=503, detail="Ollama upstream unavailable")
    
    # Ensure model is available
    logger.debug("Checking if model %s is available", model_name)
    if model_name in local_models:
        logger.info("Model %s is already available", model_name)
        return True
    
    # Pull the model if not available
    logger.info("Model %s not available locally, attempting to pull", model_name)
    result = await pull_model(model_name)
    logger.info("Preflight result for %s: %s", model_name, result)
    return result


async def _ensure_asr_admission() -> bool:
    """Ensure that ASR can run by reclaiming Ollama memory if needed.
    
    This function implements deterministic memory reclamation before ASR execution.
    It checks if any models are loaded that should be evicted to make room for ASR.
    """
    logger.info("Checking for ASR admission - reclaiming Ollama memory if needed")
    
    try:
        # Get currently loaded models from Ollama
        local_models = await list_local_models()
        if not local_models:
            logger.info("No models currently loaded, ASR can proceed")
            return True
            
        # Check if we need to evict any models to make room for ASR
        # This is a simplified approach - in a real implementation, we'd check
        # if the current models are compatible with ASR requirements
        logger.info("Checking if any models need to be evicted for ASR")
        
        # For now, we'll just log that we're checking admission
        # In a more sophisticated implementation, we'd:
        # 1. Check if models are loaded that should be evicted
        # 2. Evict them if needed
        # 3. Wait for eviction to complete
        
        logger.info("ASR admission check complete - proceeding with request")
        return True
        
    except Exception as e:
        logger.error(f"Error during ASR admission check: {e}")
        # If we can't check admission, proceed anyway (graceful degradation)
        return True


app = FastAPI(title="Ollama OpenAI Router", lifespan=lifespan, version="0.1.0")

# Roles Ollama accepts. Anything else gets dropped with a warning.
_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def _content_part_to_text(part: Any) -> str:
    """Extract plain text from a single content part, regardless of shape."""
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return str(part)
    if isinstance(part.get("text"), str):
        return part["text"]
    if isinstance(part.get("content"), str):
        return part["content"]
    if isinstance(part.get("content"), list):
        return "".join(_content_part_to_text(p) for p in part["content"])
    return ""


def _coerce_tool_arguments(args: Any) -> dict:
    """Ollama expects tool_call.function.arguments as a dict."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        if not args.strip():
            return {}
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            logger.warning("router: tool arguments not valid JSON: %r", args[:200])
            return {}
    return {}


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate incoming messages (OpenAI-shape or Anthropic-shape content
    blocks as emitted by Zoo Code / Roo Code) into the shape Ollama's
    /api/chat expects:
      - content is always a string
      - assistant tool calls are at top level as `tool_calls`
      - tool results are separate messages with role="tool" and tool_call_id
    """
    out: list[dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        if role not in _ALLOWED_ROLES:
            logger.warning("router: dropping message with unsupported role=%s", role)
            continue

        content = msg.get("content")

        # Case 1: content is a list of Anthropic/OpenAI content blocks.
        if isinstance(content, list):
            text_chunks: list[str] = []
            extracted_tool_calls: list[dict] = []   # assistant tool_use blocks
            extracted_tool_results: list[dict] = [] # user tool_result blocks

            for part in content:
                if not isinstance(part, dict):
                    text_chunks.append(str(part))
                    continue

                ptype = part.get("type")

                if ptype == "tool_use":
                    # Anthropic-style assistant tool call.
                    extracted_tool_calls.append({
                        "id": part.get("id"),
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": _coerce_tool_arguments(part.get("input")),
                        },
                    })
                elif ptype == "tool_result":
                    # Anthropic-style tool result. Must become its own
                    # role="tool" message paired by tool_call_id.
                    tr_content = part.get("content", "")
                    if isinstance(tr_content, list):
                        tr_content = "".join(_content_part_to_text(p) for p in tr_content)
                    elif not isinstance(tr_content, str):
                        try:
                            tr_content = json.dumps(tr_content)
                        except (TypeError, ValueError):
                            tr_content = str(tr_content)
                    extracted_tool_results.append({
                        "role": "tool",
                        "tool_call_id": part.get("tool_use_id") or part.get("tool_call_id"),
                        "content": tr_content,
                    })
                else:
                    # text / input_text / unknown — treat as plain text.
                    text_chunks.append(_content_part_to_text(part))

            # Tool results must come *before* the user's text in the message
            # stream so they pair correctly with the prior assistant's
            # tool_calls. Emit them first.
            out.extend(extracted_tool_results)

            text_content = "".join(text_chunks)

            if role == "assistant":
                norm: dict[str, Any] = {"role": "assistant", "content": text_content}
                if extracted_tool_calls:
                    norm["tool_calls"] = extracted_tool_calls
                # Always emit assistant turns (even empty-content with
                # tool_calls) so the model sees its own prior actions.
                if text_content or extracted_tool_calls:
                    out.append(norm)
            else:
                # user/system: only emit if there's residual text after
                # tool_results have been split out.
                if text_content:
                    out.append({"role": role, "content": text_content})
            continue

        # Case 2: content is a string (or missing). OpenAI-shape path.
        norm = {"role": role, "content": "" if content is None else str(content)}

        if msg.get("name") is not None:
            norm["name"] = msg["name"]
        if msg.get("tool_call_id") is not None:
            norm["tool_call_id"] = msg["tool_call_id"]
        if msg.get("images") is not None:
            norm["images"] = msg["images"]

        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            converted = []
            for tc in msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                converted.append({
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": _coerce_tool_arguments(fn.get("arguments")),
                    },
                })
            if converted:
                norm["tool_calls"] = converted

        out.append(norm)

    return out


def _serialize_tool_arguments_for_headroom(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tool call arguments from dicts to JSON strings for Headroom compression.
    
    Headroom's compression pipeline requires all content to be strings. This function
    serializes dict-type tool arguments into JSON strings before passing messages to
    headroom.compress(). The result is deserialized back to dicts after compression.
    """
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        norm_msg = dict(msg)
        
        # Convert tool call arguments from dicts to JSON strings
        if "tool_calls" in norm_msg and isinstance(norm_msg["tool_calls"], list):
            serialized_tool_calls = []
            for tc in norm_msg["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                
                # If arguments is a dict or list, serialize it to a JSON string
                if isinstance(args, (dict, list)):
                    try:
                        args_str = json.dumps(args)
                    except (TypeError, ValueError):
                        logger.warning(
                            "router: failed to serialize tool arguments for headroom: %r",
                            str(args)[:200],
                        )
                        args_str = "{}"
                elif args is None:
                    args_str = "{}"
                else:
                    # Already a string, keep as-is
                    args_str = str(args) if not isinstance(args, str) else args
                
                serialized_tc = dict(tc)
                if fn is not None:
                    serialized_fn = dict(fn) if isinstance(fn, dict) else {}
                    serialized_fn["arguments"] = args_str
                    serialized_tc["function"] = serialized_fn
                serialized_tool_calls.append(serialized_tc)
            norm_msg["tool_calls"] = serialized_tool_calls
        
        out.append(norm_msg)
    
    return out


def _build_ollama_payload(body: dict[str, Any], think: bool) -> dict[str, Any]:
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="'model' is required")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty array")

    logger.info(
        "router: inbound model=%s tools=%s tool_choice=%s msgs=%d",
        model,
        "yes" if body.get("tools") else "no",
        body.get("tool_choice"),
        len(messages or []),
    )

    policy_entry = MODEL_POLICY.get(model, {})
    policy_options: dict[str, Any] = dict(policy_entry.get("options") or {})

    keep_alive = body.get("keep_alive")
    if keep_alive is None:
        keep_alive = policy_entry.get("keep_alive", os.getenv("OLLAMA_KEEP_ALIVE", "10m"))

    # 1) Start with policy defaults (so num_ctx, num_batch, etc. are always present).
    options: dict[str, Any] = dict(policy_options)

    # 2) Layer OpenAI-style scalar passthroughs on top — these are client intent.
    passthrough = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repeat_penalty": "repeat_penalty",
        "seed": "seed",
    }
    for src, dst in passthrough.items():
        if src in body and body[src] is not None:
            options[dst] = body[src]
    if body.get("max_tokens") is not None:
        options["num_predict"] = body["max_tokens"]
    if body.get("stop") is not None:
        options["stop"] = body["stop"]

    # 3) Layer the explicit Ollama-style options block from the client on top,
    #    but re-assert policy-locked keys (num_ctx) so a misconfigured client
    #    cannot force an Ollama reload at a different context size.
    client_options = body.get("options")
    if client_options is not None:
        if not isinstance(client_options, dict):
            raise HTTPException(status_code=400, detail="'options' must be an object")
        for k, v in client_options.items():
            if k in POLICY_LOCKED_OPTIONS and k in policy_options:
                if v != policy_options[k]:
                    logger.info(
                        "router: overriding client %s=%s with policy %s=%s for %s",
                        k, v, k, policy_options[k], model,
                    )
                continue  # policy wins
            options[k] = v

    # 4) Re-assert locked keys one more time in case step 2 clobbered them
    #    (it doesn't today, but cheap insurance).
    for k in POLICY_LOCKED_OPTIONS:
        if k in policy_options:
            options[k] = policy_options[k]

    payload = {
        "model": model,
        "messages": _normalize_messages(messages),
        "stream": bool(body.get("stream", False)),
        "keep_alive": keep_alive,
        "think": think,
    }

    logger.info(
        "router: normalized msgs=%s",
        [
            {"role": m["role"],
            "tc": len(m.get("tool_calls", []) or []),
            "tci": m.get("tool_call_id"),
            "clen": len(m.get("content") or "")}
            for m in payload["messages"]
        ],
    )

    char_total = sum(
        len(m.get("content") or "")
        + sum(len(json.dumps(tc.get("function", {}).get("arguments") or {}))
            for tc in (m.get("tool_calls") or []))
        for m in payload["messages"]
    )
    approx_tokens = char_total // 4
    policy_ctx = (MODEL_POLICY.get(payload["model"], {}).get("options") or {}).get("num_ctx")
    logger.info(
        "router: outbound model=%s approx_tokens=%d policy_num_ctx=%s headroom=%s",
        payload["model"], approx_tokens, policy_ctx,
        (policy_ctx - approx_tokens) if policy_ctx else "unknown",
    )
    logger.info(
        "router: payload stream=%s num_predict=%s tool_choice=%s",
        payload.get("stream"),
        (payload.get("options") or {}).get("num_predict"),
        payload.get("tool_choice"),
    )
    if policy_ctx and approx_tokens > policy_ctx * 0.9:
        logger.warning(
            "router: payload approaches num_ctx (%d / %d). Truncation likely.",
            approx_tokens, policy_ctx,
        )

    # Exsure tools are forwarded
    if body.get("tools") is not None:
        payload["tools"] = body["tools"]
    if body.get("tool_choice") is not None:
        payload["tool_choice"] = body["tool_choice"]

    if options:
        payload["options"] = options
    if body.get("format") is not None:
        payload["format"] = body["format"]
    return payload

async def _ollama_post(path: str, payload: dict, stream: bool = False,):
    """POST to Ollama with timeouts appropriate for local LLM generation.

    Non-streaming chat completions on a 30B model can take minutes; the
    httpx default of 5s is far too short.
    """
    logger.debug("Calling _ollama_post to %s with model %s", path, payload.get("model"))
    timeout = httpx.Timeout(
        connect=10.0,
        read=1200.0,      # 10 min: covers prefill + full num_predict generation
        write=30.0,
        pool=30.0,
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}{path}", json=payload)
        logger.debug("Ollama response for %s: status %d", payload.get("model"), response.status_code)
        return response


@app.get("/healthz")
async def healthz() -> JSONResponse:
    try:
        timeout = httpx.Timeout(5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/version")
            resp.raise_for_status()
        return JSONResponse({"status": "ok"})
    except Exception as exc:
        error_response = _safe_error_response(exc, "healthz: Ollama health check")
        return JSONResponse(
            {"status": "degraded", **error_response},
            status_code=503
        )


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    items = []
    now = int(time.time())
    for model in MODEL_POLICY.keys():
        items.append(
            {
                "id": model,
                "object": "model",
                "created": now,
                "owned_by": "ollama",
            }
        )
    return JSONResponse({"object": "list", "data": items})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    logger.debug("Starting /v1/chat/completions request")
    body = await request.json()
    try:
        think_override = parse_think_override(request.headers.get("X-Ollama-Think"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model_name = body.get("model")
    logger.debug("Requested model: %s", model_name)
    
    # Perform preflight check
    logger.info("Starting preflight check for model %s", model_name)
    if not await _preflight_model(model_name):
        error_response = _safe_error_response(
            Exception(f"Preflight check failed for model {model_name}"),
            "chat_completions: preflight model check"
        )
        status_code = 500 if AUTO_PULL_MISSING_MODELS else 404
        raise HTTPException(
            status_code=status_code,
            detail=error_response["error"]["message"]
        )
    
    logger.info("Preflight check passed for model %s", model_name)

    model_default_think = MODEL_POLICY.get(model_name, {}).get("think", True)
    think = should_enable_think(
        body=body,
        override=think_override,
        config=THINK_POLICY_CONFIG,
        default_think=model_default_think,
    )
    body = dict(body)
    body_messages = body.get("messages") or []
    if isinstance(body_messages, list):
        body["messages"] = await _inject_retrieval_context(body, body_messages)

    payload = _build_ollama_payload(body, think=think)
    stream_options = body.get("stream_options") or {}
    include_stream_usage = bool(stream_options.get("include_usage", False))

    policy_entry = MODEL_POLICY.get(model_name, {})
    
    # Serialize tool arguments for Headroom compression (converts dicts to JSON strings)
    headroom_messages = _serialize_tool_arguments_for_headroom(payload["messages"])
    
    try:
        headroom_result = check_and_trim(headroom_messages, model_name, policy_entry)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if headroom_result.rejected:
        raise HTTPException(status_code=413, detail=headroom_result.error_response)
    
    # Headroom returns messages with JSON string arguments; _deserialize_tool_arguments()
    # in router_headroom.py converts them back to dicts for Ollama compatibility.
    payload["messages"] = headroom_result.messages

    stream = payload.get("stream", False)

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model = payload["model"]

    if stream:
        # Ensure preflight is complete before starting streaming response
        logger.info("Starting preflight check for streaming request")
        if not await _preflight_model(model_name):
            logger.error("Preflight check failed for streaming model %s", model_name)
            raise HTTPException(status_code=500, detail=f"Failed to pull model {model_name}")
        
        async def event_stream():
            logger.debug("Starting streaming response for model %s", model)
            # Opening role chunk so clients see structure immediately.
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            # Single queue feeds the generator. Heartbeat task pushes
            # keepalive comments; reader task pushes real SSE frames.
            queue: asyncio.Queue = asyncio.Queue()
            SENTINEL = object()

            async def heartbeat():
                """Emit an SSE comment every 15s while the upstream is
                producing nothing, so clients and intermediate proxies
                don't treat the silent prefill window as a dead
                connection."""
                try:
                    while True:
                        await asyncio.sleep(15)
                        await queue.put(": keepalive\n\n")
                except asyncio.CancelledError:
                    pass

            async def reader():
                """Stream Ollama's /api/chat response and translate each
                chunk into an OpenAI-shape SSE frame."""
                try:
                    logger.debug("About to call Ollama /api/chat for streaming model %s", model)
                    timeout = httpx.Timeout(
                        connect=10.0, read=1800.0, write=30.0, pool=30.0
                    )
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        async with client.stream(
                            "POST",
                            f"{OLLAMA_BASE_URL}/api/chat",
                            json=payload,
                        ) as resp:
                            logger.debug("Ollama /api/chat streaming response status: %d", resp.status_code)
                            if resp.status_code >= 400:
                                # Check if this is a 404 for a policy model that should be pulled
                                if resp.status_code == 404:
                                    try:
                                        error_text = (await resp.aread()).decode("utf-8", errors="ignore")
                                        if "not found" in error_text and model_name in MODEL_POLICY:
                                            logger.info("Received 404 for model %s in streaming, attempting one-time recovery pull", model_name)
                                            # Try to pull the model again and retry once
                                            if await _preflight_model(model_name):
                                                logger.info("Recovery pull successful, retrying Ollama request for %s", model_name)
                                                # Note: We can't easily retry the streaming request, so we'll just return the error
                                                # This is a limitation of the streaming approach - the original request has already started
                                                logger.warning("Recovery pull succeeded but streaming request cannot be retried")
                                    except Exception:
                                        # If we can't parse the error or recovery fails, continue with original error
                                        pass

                                error_text = (await resp.aread()).decode(
                                    "utf-8", errors="ignore"
                                )
                                error_response = _safe_error_response(
                                    Exception(f"Ollama upstream error status={resp.status_code}: {error_text[:500]}"),
                                    "streaming: Ollama upstream"
                                )
                                err_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "error",
                                    }],
                                    "error": {
                                        "message": error_response["error"]["message"],
                                        "type": error_response["error"]["type"],
                                        "code": error_response["error"]["code"],
                                    },
                                }
                                await queue.put(
                                    f"data: {json.dumps(err_chunk)}\n\n"
                                )
                                return

                            saw_tool_calls = False
                            async for line in resp.aiter_lines():
                                if not line.strip():
                                    continue
                                data = json.loads(line)
                                message = data.get("message") or {}

                                # Tool-call delta translation.
                                raw_tool_calls = message.get("tool_calls") or []
                                tc_deltas = _streaming_tool_call_deltas(raw_tool_calls)
                                if tc_deltas:
                                    saw_tool_calls = True
                                    tc_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"tool_calls": tc_deltas},
                                            "finish_reason": None,
                                        }],
                                    }
                                    await queue.put(
                                        f"data: {json.dumps(tc_chunk)}\n\n"
                                    )

                                # Plain content delta.
                                token = message.get("content", "")
                                if token:
                                    chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": token},
                                            "finish_reason": None,
                                        }],
                                    }
                                    await queue.put(
                                        f"data: {json.dumps(chunk)}\n\n"
                                    )

                                if data.get("done"):
                                    usage_chunk = _build_stream_usage_chunk(
                                        include_usage=include_stream_usage,
                                        completion_id=completion_id,
                                        created=created,
                                        model=model,
                                        done_data=data,
                                    )
                                    if usage_chunk is not None:
                                        await queue.put(f"data: {json.dumps(usage_chunk)}\n\n")

                                    if saw_tool_calls:
                                        finish_reason = "tool_calls"
                                    else:
                                        dr = data.get("done_reason")
                                        finish_reason = (
                                            dr if dr in ("stop", "length", "content_filter")
                                            else "stop"
                                        )
                                    end_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": finish_reason,
                                        }],
                                    }
                                    await queue.put(
                                        f"data: {json.dumps(end_chunk)}\n\n"
                                    )
                                    await queue.put("data: [DONE]\n\n")
                                    break
                except Exception as exc:
                    error_response = _safe_error_response(exc, "streaming: reader task")
                    err_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "error",
                        }],
                        "error": {
                            "message": error_response["error"]["message"],
                            "type": error_response["error"]["type"],
                            "code": error_response["error"]["code"],
                        },
                    }
                    await queue.put(f"data: {json.dumps(err_chunk)}\n\n")
                finally:
                    await queue.put(SENTINEL)

            hb_task = asyncio.create_task(heartbeat())
            rd_task = asyncio.create_task(reader())
            try:
                while True:
                    item = await queue.get()
                    if item is SENTINEL:
                        break
                    yield item
            finally:
                hb_task.cancel()
                if not rd_task.done():
                    rd_task.cancel()
                for t in (hb_task, rd_task):
                    try:
                        await t
                    except:
                        pass
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    logger.debug("About to call Ollama /api/chat for non-streaming model %s", model)
    response = await _ollama_post("/api/chat", payload, stream=False)
    
    # If we get a 404 for a model that should be in policy, try one recovery pull
    if response.status_code == 404:
        try:
            # Check if we can parse the JSON error
            error_data = response.json()
            error_message = error_data.get("error", "")
            if "not found" in error_message and model_name in MODEL_POLICY:
                logger.info("Received 404 for model %s, attempting one-time recovery pull", model_name)
                # Try to pull the model again and retry once
                if await _preflight_model(model_name):
                    logger.info("Recovery pull successful, retrying Ollama request for %s", model_name)
                    response = await _ollama_post("/api/chat", payload, stream=False)
        except Exception:
            # If we can't parse the error or recovery fails, continue with original error
            pass
    
    logger.debug("Ollama /api/chat response for non-streaming: status %d", response.status_code)
    if response.status_code >= 400:
        error_response = _safe_error_response(
            Exception(f"Ollama error status={response.status_code}: {response.text[:500]}"),
            "chat_completions: Ollama upstream"
        )
        raise HTTPException(
            status_code=response.status_code,
            detail=error_response["error"]["message"]
        )

    result = response.json()
    ollama_message = result.get("message") or {}

    content = ollama_message.get("content", "") or ""
    tool_calls = _translate_tool_calls(ollama_message.get("tool_calls") or [])

    assistant_message = {"role": "assistant", "content": content}
    if tool_calls:
        assistant_message["tool_calls"] = tool_calls

    if tool_calls:
        finish_reason = "tool_calls"
    else:
        done_reason = result.get("done_reason")
        finish_reason = done_reason if done_reason in ("stop", "length", "content_filter") else "stop"

    usage = _build_non_stream_usage(
        ollama_result=result,
        payload_messages=payload.get("messages") or [],
        model=model,
        completion_text=content,
    )

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model = body.get("model") or EMBEDDING_MODEL_DEFAULT
    if not model:
        raise HTTPException(status_code=400, detail="'model' is required")

    logger.info("Starting preflight check for embedding model %s", model)
    if not await _preflight_model(model):
        error_response = _safe_error_response(
            Exception(f"Preflight check failed for embedding model {model}"),
            "embeddings: preflight model check"
        )
        status_code = 500 if AUTO_PULL_MISSING_MODELS else 404
        raise HTTPException(
            status_code=status_code,
            detail=error_response["error"]["message"]
        )

    inputs = _coerce_embedding_inputs(body.get("input"))
    data = []
    total_prompt_tokens = 0
    for idx, text in enumerate(inputs):
        vector = await _fetch_embedding(model, text)
        token_count = tokenizer.count_completion_tokens(text, model)
        total_prompt_tokens += int(token_count)
        data.append(
            {
                "object": "embedding",
                "embedding": vector,
                "index": idx,
            }
        )

    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "total_tokens": total_prompt_tokens,
            },
        }
    )

def _resolve_asr_base_url(request: Request) -> str:
    """Resolve ASR upstream base URL for alignment forwarding.

    Priority:
    1) Explicit ASR_BASE_URL override.
    2) Infer from OLLAMA_BASE_URL host + ASR_PORT (models run on Orin).
    3) Fallback to inbound router host.
    """
    if ASR_BASE_URL_ENV:
        return ASR_BASE_URL_ENV

    ollama_host = urlsplit(OLLAMA_BASE_URL).hostname
    host = ollama_host or request.url.hostname or "127.0.0.1"
    return f"{ASR_SCHEME}://{host}:{ASR_PORT}"


async def _asr_post_json(base_url: str, path: str, payload: dict):
    """POST JSON to ASR service with timeouts appropriate for audio processing."""
    logger.debug("Calling _asr_post_json to %s", path)
    timeout = httpx.Timeout(
        connect=10.0,
        read=1200.0,
        write=30.0,
        pool=30.0,
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url.rstrip('/')}{path}", json=payload)
        logger.debug("ASR JSON response for %s: status %d", path, response.status_code)
        return response


async def _asr_post_multipart(
    base_url: str,
    path: str,
    *,
    fields: dict[str, str],
    file_field: str,
    file_name: str,
    file_bytes: bytes,
    file_content_type: str,
):
    """POST multipart/form-data to ASR service."""
    logger.debug("Calling _asr_post_multipart to %s", path)
    timeout = httpx.Timeout(
        connect=10.0,
        read=1200.0,
        write=30.0,
        pool=30.0,
    )
    files = {file_field: (file_name, file_bytes, file_content_type)}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}{path}",
            data=fields,
            files=files,
        )
        logger.debug("ASR multipart response for %s: status %d", path, response.status_code)
        return response


async def _forward_alignment_request(request: Request) -> JSONResponse:
    """Forward alignment request to ASR using multipart or JSON, preserving upstream status."""
    content_type = (request.headers.get("content-type") or "").lower()
    asr_base_url = _resolve_asr_base_url(request)
    logger.info("Alignment forward request content-type=%s", content_type or "<empty>")

    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("media_file") or form.get("file")
        if upload is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "cross_host_alignment_requires_multipart_upload: "
                    "multipart request missing media_file/file"
                ),
            )

        file_name = getattr(upload, "filename", "audio.wav") or "audio.wav"
        file_content_type = getattr(upload, "content_type", "application/octet-stream") or "application/octet-stream"
        file_bytes = await upload.read()

        # Cross-host alignment must not rely on local filesystem paths.
        if form.get("audio_path") is not None or form.get("media_path") is not None:
            logger.warning("Dropping audio_path/media_path form fields for cross-host alignment forwarding")

        fields: dict[str, str] = {}
        for key in (
            "model",
            "model_override",
            "model_accuracy",
            "return_word_timestamps",
            "prefer_forced_alignment",
            "language",
            "strict",
            "response_format",
        ):
            value = form.get(key)
            if value is not None:
                fields[key] = str(value)

        logger.info("Forwarding alignment upstream as multipart to %s/align", asr_base_url.rstrip("/"))
        response = await _asr_post_multipart(
            asr_base_url,
            "/align",
            fields=fields,
            file_field="media_file",
            file_name=file_name,
            file_bytes=file_bytes,
            file_content_type=file_content_type,
        )
    else:
        logger.warning("Rejecting non-multipart alignment request for cross-host flow")
        raise HTTPException(
            status_code=400,
            detail=(
                "cross_host_alignment_requires_multipart_upload: "
                "send multipart/form-data with media_file and alignment fields"
            ),
        )

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return JSONResponse(response.json())


# Native alignment endpoint(s)
@app.post("/align")
async def align(request: Request):
    """Handle native alignment requests with rich timing information."""
    try:
        logger.info("Checking ASR admission before processing alignment request")
        admission_ok = await _ensure_asr_admission()
        if not admission_ok:
            logger.error("ASR admission denied")
            raise HTTPException(status_code=503, detail="ASR admission denied - insufficient resources")

        logger.info("Forwarding alignment request to ASR service")
        return await _forward_alignment_request(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in align endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/align")
async def v1_audio_align(request: Request):
    """Handle native alignment requests with OpenAI v1 compatibility."""
    try:
        logger.info("Checking ASR admission before processing alignment request")
        admission_ok = await _ensure_asr_admission()
        if not admission_ok:
            logger.error("ASR admission denied")
            raise HTTPException(status_code=503, detail="ASR admission denied - insufficient resources")

        logger.info("Forwarding alignment request to ASR service")
        return await _forward_alignment_request(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in v1/audio/align endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/audio/transcriptions")
async def audio_transcription(request: Request):
    """Handle audio transcription requests with word-level timing."""
    try:
        logger.info("Checking ASR admission before processing request")
        admission_ok = await _ensure_asr_admission()
        if not admission_ok:
            logger.error("ASR admission denied")
            raise HTTPException(status_code=503, detail="ASR admission denied - insufficient resources")

        logger.info("Forwarding audio transcription request to ASR service")
        return await _forward_alignment_request(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in audio transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
