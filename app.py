import asyncio
import inspect
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

from policy import load_think_policy_config, parse_think_override, should_enable_think, _detect_tool_call_failure
from router_headroom import (
    apply_overflow_recovery,
    check_and_trim,
    is_context_overflow_error,
    maybe_apply_proactive_summarization,
)
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


async def _resolve_stream_context(client: Any, *args: Any, **kwargs: Any):
    """Resolve mocked or real async stream objects before entering the context."""
    stream_obj = client.stream(*args, **kwargs)
    if inspect.isawaitable(stream_obj):
        stream_obj = await stream_obj
    return stream_obj

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
try:
    HEADROOM_RECOVERY_MAX_RETRIES = max(0, int(os.getenv("HEADROOM_RECOVERY_MAX_RETRIES", "1")))
except ValueError:
    HEADROOM_RECOVERY_MAX_RETRIES = 1


ZOO_CONTEXT_CONDENSE_TRIGGER = (
    os.getenv(
        "ZOO_CONTEXT_CONDENSE_TRIGGER",
        "ZOO_CONTEXT_CONDENSE_V1: delegate context condensing to router headroom pipeline.",
    )
    .strip()
)
ZOO_CONTEXT_CONDENSE_TRIGGER_TOKEN = "zoo_context_condense_v1"

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
    """Fetch embedding from Ollama with retry logic for transient 503 errors.
    
    Handles Ollama's model eviction/reload cycle by retrying on 503 Service Unavailable
    responses with exponential backoff. This prevents client-facing errors during the
    brief window when Ollama is reloading a model into VRAM/RAM.
    
    Args:
        model: The embedding model name (e.g., 'qwen3-embedding:4b' or 'qwen3-embedding:8b')
        prompt: The text to embed
        
    Returns:
        List of float embedding values
        
    Raises:
        HTTPException: 503 if model reloading persists after retries, or other HTTP errors
    """
    payload = {"model": model, "prompt": prompt}
    max_retries = 3
    backoff_base = 1.0
    
    for attempt in range(max_retries):
        try:
            response = await _ollama_post("/api/embeddings", payload, stream=False)
            
            if response.status_code == 503:
                if attempt < max_retries - 1:
                    wait_time = backoff_base * (2 ** attempt)
                    logger.warning(
                        "Ollama 503 (model reloading) for model=%s, retry %d/%d in %.1fs",
                        model, attempt + 1, max_retries, wait_time
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "Ollama 503 (model reloading) for model=%s after %d retries",
                        model, max_retries
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="Ollama upstream unavailable - model reloading"
                    )
            
            if response.status_code >= 400:
                # Don't retry on client errors (4xx) - they indicate bad requests
                logger.error(
                    "Embedding fetch failed for model=%s with status %d: %s",
                    model, response.status_code, response.text[:200]
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama error: {response.status_code}"
                )
                
            data = response.json()
            embedding = data.get("embedding")
            if not isinstance(embedding, list):
                raise HTTPException(status_code=502, detail="Invalid embedding response from Ollama")
            return [float(v) for v in embedding]
            
        except httpx.ConnectError as e:
            # Connection errors are retryable
            if attempt < max_retries - 1:
                wait_time = backoff_base * (2 ** attempt)
                logger.warning(
                    "Ollama connection error for model=%s, retry %d/%d in %.1fs: %s",
                    model, attempt + 1, max_retries, wait_time, str(e)[:100]
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(
                    "Ollama connection error for model=%s after %d retries: %s",
                    model, max_retries, str(e)[:200]
                )
                raise HTTPException(status_code=503, detail="Ollama upstream unavailable")
                
        except httpx.ConnectTimeout as e:
            # Connection timeouts are retryable
            if attempt < max_retries - 1:
                wait_time = backoff_base * (2 ** attempt)
                logger.warning(
                    "Ollama connection timeout for model=%s, retry %d/%d in %.1fs: %s",
                    model, attempt + 1, max_retries, wait_time, str(e)[:100]
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(
                    "Ollama connection timeout for model=%s after %d retries: %s",
                    model, max_retries, str(e)[:200]
                )
                raise HTTPException(status_code=503, detail="Ollama upstream unavailable")
                
        except httpx.HTTPStatusError as e:
            # HTTP status errors - only retry on 503
            if e.response.status_code == 503 and attempt < max_retries - 1:
                wait_time = backoff_base * (2 ** attempt)
                logger.warning(
                    "Ollama 503 (model reloading) for model=%s, retry %d/%d in %.1fs",
                    model, attempt + 1, max_retries, wait_time
                )
                await asyncio.sleep(wait_time)
                continue
            raise
            
        except Exception as e:
            # Other exceptions are not retryable
            logger.error(
                "Unexpected error fetching embedding for model=%s: %s",
                model, str(e)[:200]
            )
            raise HTTPException(status_code=503, detail="Ollama upstream unavailable")
    
    # Fallback - should not reach here
    raise HTTPException(status_code=503, detail="Ollama upstream unavailable")





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
                tool_call_think = item.get("tool_call_think")
                retry_on_tool_call_failure = item.get("retry_on_tool_call_failure", False)
                retry_temperature = item.get("retry_temperature", 0.1)
                options = item.get("options") or {}
                if not isinstance(options, dict):
                    logger.warning("policy: ignoring non-dict options for %s", model_name)
                    options = {}
                table[model_name] = {
                    "keep_alive": keep_alive,
                    "think": bool(think) if think is not None else True,
                    "tool_call_think": bool(tool_call_think) if tool_call_think is not None else None,
                    "retry_on_tool_call_failure": bool(retry_on_tool_call_failure),
                    "retry_temperature": float(retry_temperature),
                    "options": {k: _parse_scalar(v) for k, v in options.items()},
                    "warmup": bool(item.get("warmup", False)),
                    "reserved_output_tokens": _parse_scalar(item.get("reserved_output_tokens", 2048)),
                    "safety_headroom_tokens": _parse_scalar(item.get("safety_headroom_tokens", 2048)),
                    "proactive_summarization_enabled": bool(item.get("proactive_summarization_enabled", False)),
                    "proactive_summarization_threshold_tokens": _parse_scalar(item.get("proactive_summarization_threshold_tokens", 0)),
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


async def _ollama_model_is_evicted(model_name: str) -> bool:
    """Return True when Ollama no longer reports the requested model as loaded.

    This detects the evicted-model case that leaves a stream stalled while the router
    is waiting for upstream output with no HTTP error from Ollama.
    """
    if not model_name:
        return False
    try:
        timeout = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/ps")
            response.raise_for_status()
            data = response.json()
            models = data.get("models") or []
            if not models:
                logger.warning("Ollama /api/ps reports no loaded models while stream for %s is stalled", model_name)
                return True
            loaded_names: list[str] = []
            for model in models:
                if isinstance(model, dict):
                    name = model.get("name") or model.get("model")
                    if name:
                        loaded_names.append(str(name))
                elif isinstance(model, str):
                    loaded_names.append(model)
            if model_name not in loaded_names:
                logger.warning(
                    "Stream for %s stalled because the model is no longer loaded; loaded models=%s",
                    model_name,
                    loaded_names,
                )
                return True
            return False
    except Exception as exc:
        logger.debug("Failed to inspect Ollama /api/ps for stalled stream on %s: %s", model_name, exc)
        try:
            timeout = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                response.raise_for_status()
                data = response.json()
                models = data.get("models") or []
                model_names: list[str] = []
                for model in models:
                    if isinstance(model, dict):
                        name = model.get("name") or model.get("model")
                        if name:
                            model_names.append(str(name))
                    elif isinstance(model, str):
                        model_names.append(model)
                if model_name not in model_names:
                    logger.warning(
                        "Ollama /api/tags does not list %s while stream is stalled; available models=%s",
                        model_name,
                        model_names,
                    )
                    return True
        except Exception as tags_exc:
            logger.debug("Failed to inspect Ollama /api/tags for stalled stream on %s: %s", model_name, tags_exc)
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
        supported_models = ", ".join(sorted(MODEL_POLICY.keys())) or "<none>"
        logger.warning(
            "Model %s not in active policy file %s; supported models=%s",
            model_name,
            POLICY_FILE,
            supported_models,
        )
        raise HTTPException(
            status_code=404,
            detail=(
                f"model '{model_name}' is not enabled by the active policy in "
                f"{os.path.basename(POLICY_FILE)}; supported models: {supported_models}"
            ),
        )

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


def _message_content_to_text(content: Any) -> str:
    """Normalize OpenAI/Anthropic-style content payloads into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_part_to_text(part) for part in content)
    if isinstance(content, dict):
        return _content_part_to_text(content)
    if content is None:
        return ""
    return str(content)


def _is_zoo_condense_trigger_text(text: str) -> bool:
    """Detect canonical Zoo context-condense trigger payloads."""
    if not text:
        return False
    lowered = text.lower()
    if ZOO_CONTEXT_CONDENSE_TRIGGER_TOKEN in lowered:
        return True
    if ZOO_CONTEXT_CONDENSE_TRIGGER and ZOO_CONTEXT_CONDENSE_TRIGGER.lower() in lowered:
        return True
    return False


def _is_zoo_condense_request(messages: list[dict[str, Any]]) -> bool:
    """Return True when a request is Zoo-native context condensing."""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role not in {"system", "user"}:
            continue
        text = _message_content_to_text(msg.get("content"))
        if _is_zoo_condense_trigger_text(text):
            return True
    return False


def _remove_zoo_condense_trigger_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip condense-trigger messages so only real conversation context is compacted."""
    filtered: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if _is_zoo_condense_trigger_text(_message_content_to_text(msg.get("content"))):
            continue
        filtered.append(msg)
    return filtered


def _render_zoo_condensed_context(messages: list[dict[str, Any]], max_chars: int = 12000) -> str:
    """Render compacted conversation into deterministic text Zoo can store as condensed context."""
    lines: list[str] = ["CONDENSED_CONTEXT_V1"]

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "unknown")
        role_header = role
        if role == "tool" and msg.get("tool_call_id"):
            role_header = f"tool[{msg.get('tool_call_id')}]"
        lines.append(f"[{role_header}]")

        content = (msg.get("content") or "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if content:
            lines.append(content)

        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, list) and tool_calls:
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                fn_name = fn.get("name", "")
                lines.append(f"tool_call: {fn_name}")

    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n[truncated]"


def _build_condense_usage(
    model: str,
    prompt_messages: list[dict[str, Any]],
    completion_text: str,
) -> dict[str, int]:
    prompt_tokens = int(tokenizer.count_prompt_tokens(prompt_messages, model) or 0)
    completion_tokens = int(tokenizer.count_completion_tokens(completion_text, model) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _build_condense_streaming_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    condensed_text: str,
    usage: dict[str, int],
    include_stream_usage: bool,
) -> StreamingResponse:
    async def _stream():
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

        content_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": condensed_text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(content_chunk)}\n\n"

        if include_stream_usage:
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage,
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n"

        end_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _build_condense_completion_response(
    *,
    completion_id: str,
    created: int,
    model: str,
    condensed_text: str,
    usage: dict[str, int],
    stream: bool,
    include_stream_usage: bool,
) -> JSONResponse | StreamingResponse:
    if stream:
        return _build_condense_streaming_response(
            completion_id=completion_id,
            created=created,
            model=model,
            condensed_text=condensed_text,
            usage=usage,
            include_stream_usage=include_stream_usage,
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
                    "message": {"role": "assistant", "content": condensed_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    )


def _run_headroom_condense_pipeline(
    *,
    model_name: str,
    messages: list[dict[str, Any]],
    policy_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the router's headroom compaction/recovery pipeline for condensing requests."""
    normalized_messages = _normalize_messages(messages)
    headroom_messages = _serialize_tool_arguments_for_headroom(normalized_messages)

    try:
        proactive_result = maybe_apply_proactive_summarization(
            headroom_messages,
            model_name,
            policy_entry,
        )
        if proactive_result.triggered:
            headroom_messages = proactive_result.messages
    except RuntimeError as exc:
        logger.warning(
            "router: proactive summarization failed during condense interception for model=%s: %s",
            model_name,
            exc,
        )

    headroom_result = check_and_trim(headroom_messages, model_name, policy_entry)
    if headroom_result.rejected:
        raise HTTPException(status_code=413, detail=headroom_result.error_response)
    condensed_messages = list(headroom_result.messages or [])

    recovery_result = apply_overflow_recovery(condensed_messages, model_name, policy_entry)
    if recovery_result.changed:
        condensed_messages = recovery_result.messages

    return condensed_messages


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


def _extract_tool_names(body: dict[str, Any]) -> list[str]:
    """Extract tool names from request body for retry detection.

    Args:
        body: Request body containing 'tools' array

    Returns:
        List of tool function names (empty list if no tools or invalid structure)
    """
    tool_names: list[str] = []
    for tool in body.get("tools", []):
        if isinstance(tool, dict) and "function" in tool:
            fn = tool["function"]
            if isinstance(fn, dict) and "name" in fn:
                tool_names.append(fn["name"])
    return tool_names


def _should_retry_tool_call(body: dict[str, Any], model_entry: dict[str, Any]) -> bool:
    """Check if retry guard should be evaluated for this request.

    This encapsulates the precondition check: retry is enabled for this model
    AND the request includes tools. The caller must still check whether tool_calls
    were actually produced in the response.

    Args:
        body: Request body
        model_entry: Model configuration from MODEL_POLICY

    Returns:
        True if retry guard preconditions are met, False otherwise
    """
    retry_enabled = model_entry.get("retry_on_tool_call_failure", False)
    has_tools = bool(body.get("tools"))
    return retry_enabled and has_tools


async def _perform_tool_call_retry(
    body: dict[str, Any],
    model_name: str,
    model_entry: dict[str, Any],
    original_payload: dict[str, Any],
) -> dict[str, Any]:
    """Perform one-shot retry with think:false and low temperature.
    
    This is called when a model returns assistant text mentioning a tool but fails
    to produce structured tool_calls. The retry removes the invalid assistant response
    and forces think:false with low temperature to encourage proper tool call formatting.
    
    Args:
        body: Original request body
        model_name: Model name
        model_entry: Model configuration from MODEL_POLICY
        original_payload: Original Ollama payload (for reference)
    
    Returns:
        dict with keys:
        - success: bool indicating if retry succeeded
        - result: dict containing the retry response (if success=True)
        - error: str containing error message (if success=False)
    """
    logger.info("router: initiating tool-call retry for model=%s", model_name)
    
    # Remove the invalid assistant response from message history
    retry_messages = []
    for msg in body.get("messages", []):
        if msg.get("role") == "assistant":
            # Skip the last assistant message that failed to produce tool_calls
            continue
        retry_messages.append(msg)
    
    # Build retry body with cleaned messages
    retry_body = {
        "model": model_name,
        "messages": retry_messages,
        "tools": body.get("tools"),
        "tool_choice": body.get("tool_choice"),
        "stream": False,  # Retry is always non-streaming
    }
    
    # Force think:false and low temperature for retry
    retry_temperature = model_entry.get("retry_temperature", 0.1)
    
    try:
        # Build payload with forced settings
        retry_payload = _build_ollama_payload(retry_body, think=False)
        
        # Override temperature in options
        if "options" not in retry_payload:
            retry_payload["options"] = {}
        retry_payload["options"]["temperature"] = retry_temperature
        
        logger.info(
            "router: retry payload model=%s think=false temperature=%.2f messages=%d",
            model_name,
            retry_temperature,
            len(retry_messages),
        )
        
        # Make the retry request
        response = await _ollama_post("/api/chat", retry_payload, stream=False)
        
        if response.status_code >= 400:
            error_text = response.text[:500]
            logger.error(
                "router: tool-call retry failed model=%s status=%d error=%s",
                model_name,
                response.status_code,
                error_text,
            )
            return {
                "success": False,
                "result": None,
                "error": f"Retry request failed with status {response.status_code}: {error_text}",
            }
        
        result = response.json()
        
        # Check if retry produced tool_calls
        ollama_message = result.get("message") or {}
        tool_calls = _translate_tool_calls(ollama_message.get("tool_calls") or [])
        
        if tool_calls:
            logger.info(
                "router: tool-call retry succeeded model=%s tool_calls=%d",
                model_name,
                len(tool_calls),
            )
            return {
                "success": True,
                "result": result,
                "error": None,
            }
        else:
            logger.warning(
                "router: tool-call retry did not produce tool_calls model=%s",
                model_name,
            )
            return {
                "success": False,
                "result": result,
                "error": "Retry did not produce tool_calls",
            }
    
    except Exception as e:
        logger.error(
            "router: tool-call retry exception model=%s error=%s",
            model_name,
            str(e),
            exc_info=True,
        )
        return {
            "success": False,
            "result": None,
            "error": f"Retry exception: {str(e)}",
        }


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

    body = dict(body)
    body_messages = body.get("messages") or []
    stream_options = body.get("stream_options") or {}
    include_stream_usage = bool(stream_options.get("include_usage", False))

    if isinstance(body_messages, list) and _is_zoo_condense_request(body_messages):
        if not model_name:
            raise HTTPException(status_code=400, detail="'model' is required")

        logger.info(
            "router: intercepted zoo context condense request for model=%s using trigger=%s",
            model_name,
            ZOO_CONTEXT_CONDENSE_TRIGGER_TOKEN,
        )
        policy_entry = MODEL_POLICY.get(model_name, {})
        source_messages = _remove_zoo_condense_trigger_messages(body_messages)
        if not source_messages:
            source_messages = body_messages

        condensed_messages = _run_headroom_condense_pipeline(
            model_name=model_name,
            messages=source_messages,
            policy_entry=policy_entry,
        )
        condensed_text = _render_zoo_condensed_context(condensed_messages)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        usage = _build_condense_usage(model_name, condensed_messages, condensed_text)
        return _build_condense_completion_response(
            completion_id=completion_id,
            created=created,
            model=model_name,
            condensed_text=condensed_text,
            usage=usage,
            stream=bool(body.get("stream", False)),
            include_stream_usage=include_stream_usage,
        )
    
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

    model_entry = MODEL_POLICY.get(model_name, {})
    model_default_think = model_entry.get("think", True)
    tool_call_think = model_entry.get("tool_call_think")
    think = should_enable_think(
        body=body,
        override=think_override,
        config=THINK_POLICY_CONFIG,
        default_think=model_default_think,
        model_name=model_name or "",
        tool_call_think=tool_call_think,
    )
    if isinstance(body_messages, list):
        body["messages"] = await _inject_retrieval_context(body, body_messages)

    payload = _build_ollama_payload(body, think=think)

    policy_entry = MODEL_POLICY.get(model_name, {})
    
    # Serialize tool arguments for Headroom compression (converts dicts to JSON strings)
    headroom_messages = _serialize_tool_arguments_for_headroom(payload["messages"])

    try:
        proactive_result = maybe_apply_proactive_summarization(headroom_messages, model_name, policy_entry)
        if proactive_result.triggered:
            logger.info(
                "router: proactive summarization model=%s before=%s after=%s reason=%s",
                model_name,
                proactive_result.tokens_before,
                proactive_result.tokens_after,
                proactive_result.reason,
            )
            headroom_messages = proactive_result.messages
    except RuntimeError as exc:
        logger.warning("router: proactive summarization failed for model=%s: %s", model_name, exc)

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

                    stream_state = {
                        "request_id": completion_id,
                        "model_name": model_name,
                        "opened_at": time.time(),
                        "last_progress_at": None,
                        "retry_count": 0,
                        "stream_state": "opening",
                        "finalized": False,
                    }

                    def mark_progress(state: str) -> None:
                        stream_state["stream_state"] = state
                        stream_state["last_progress_at"] = time.time()
                    
                    # Context overflow recovery: bounded retries after successful reductions.
                    max_context_retries = HEADROOM_RECOVERY_MAX_RETRIES
                    context_retry_count = 0
                    eviction_retry_count = 0
                    empty_stream_retry_count = 0

                    while True:
                        async with httpx.AsyncClient(timeout=timeout) as client:
                            stream_obj = await _resolve_stream_context(
                                client,
                                "POST",
                                f"{OLLAMA_BASE_URL}/api/chat",
                                json=payload,
                            )
                            async with stream_obj as resp:
                                logger.debug("Ollama /api/chat streaming response status: %d", resp.status_code)

                                # Handle 404 for policy models (existing logic)
                                if resp.status_code == 404:
                                    try:
                                        error_text = (await resp.aread()).decode("utf-8", errors="ignore")
                                        if "not found" in error_text and model_name in MODEL_POLICY:
                                            logger.info("Received 404 for model %s in streaming, attempting one-time recovery pull", model_name)
                                            if await _preflight_model(model_name):
                                                logger.info("Recovery pull successful, retrying Ollama request for %s", model_name)
                                    except Exception:
                                        pass

                                # Handle context overflow with recovery trim (new pipeline)
                                if resp.status_code >= 400:
                                    error_text = (await resp.aread()).decode("utf-8", errors="ignore")
                                    if is_context_overflow_error(resp.status_code, error_text):
                                        context_retry_count += 1
                                        if context_retry_count <= max_context_retries:
                                            logger.warning(
                                                "Context overflow detected in streaming for model %s (status=%d), applying recovery trim and retrying (attempt %d/%d)",
                                                model_name,
                                                resp.status_code,
                                                context_retry_count,
                                                max_context_retries,
                                            )
                                            try:
                                                recovery_result = apply_overflow_recovery(
                                                    payload["messages"], model_name, policy_entry
                                                )
                                                trimmed_messages = recovery_result.messages
                                                was_trimmed = recovery_result.changed
                                                logger.info(
                                                    "Overflow recovery streaming model=%s stage=%s attempts=%d changed=%s success_candidate=%s",
                                                    model_name,
                                                    recovery_result.final_stage,
                                                    len(recovery_result.attempts),
                                                    recovery_result.changed,
                                                    recovery_result.success_candidate,
                                                )
                                                if was_trimmed:
                                                    logger.info("Overflow recovery applied for streaming model %s, retrying with %d messages", model_name, len(trimmed_messages))
                                                    payload["messages"] = trimmed_messages
                                                    continue  # retry the while loop
                                                logger.warning("Overflow recovery did not reduce messages for streaming model %s; propagating upstream error", model_name)
                                            except Exception as trim_exc:
                                                logger.error("Overflow recovery failed for streaming model %s: %s", model_name, trim_exc, exc_info=True)

                                # Any other error (or context retry exhausted): emit error chunk and return
                                if resp.status_code >= 400:
                                    error_text = (await resp.aread()).decode("utf-8", errors="ignore")
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
                                    logger.debug("streaming: emitted error chunk (early error path)")
                                    if not stream_state.get("finalized"):
                                        await queue.put("data: [DONE]\n\n")
                                        stream_state["finalized"] = True
                                        logger.debug("streaming: emitted [DONE] terminator (early error path)")
                                    return

                                retry_stream = False
                                saw_tool_calls = False
                                saw_assistant_payload = False
                                first_line_seen = False
                                streaming_content_chunks = []  # Track content for retry detection
                                line_iter = resp.aiter_lines()
                                while True:
                                    try:
                                        line = await anext(line_iter)
                                    except StopAsyncIteration as exc:
                                        if await _ollama_model_is_evicted(model_name):
                                            stream_state["stream_state"] = "ended_evicted_before_completion"
                                            if not first_line_seen and eviction_retry_count < 1:
                                                eviction_retry_count += 1
                                                retry_stream = True
                                                logger.warning(
                                                    "Detected silent-start eviction for model %s before any output; retrying once after preflight",
                                                    model_name,
                                                )
                                                await _preflight_model(model_name)
                                                break
                                            logger.warning(
                                                "Ollama stream for model %s ended before completion because the model was evicted or unloaded",
                                                model_name,
                                            )
                                            raise TimeoutError(
                                                f"Ollama stream for model {model_name} ended without completion because the model was evicted or unloaded"
                                            ) from exc
                                        stream_state["stream_state"] = (
                                            "ended_before_progress"
                                            if stream_state["stream_state"] == "opening"
                                            else "ended_without_completion"
                                        )
                                        raise RuntimeError(
                                            f"Ollama stream for model {model_name} ended without producing a completion or error"
                                        ) from exc

                                    if not line or not str(line).strip():
                                        continue
                                    if str(line).startswith(":"):
                                        continue
                                    if not first_line_seen:
                                        mark_progress("generating")
                                        await queue.put(f"data: {json.dumps(first_chunk)}\n\n")
                                        first_line_seen = True
                                    data = json.loads(line)
                                    message = data.get("message") or {}

                                    # Tool-call delta translation.
                                    raw_tool_calls = message.get("tool_calls") or []
                                    tc_deltas = _streaming_tool_call_deltas(raw_tool_calls)
                                    if tc_deltas:
                                        saw_tool_calls = True
                                        saw_assistant_payload = True
                                        mark_progress("tool_calls")
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
                                        saw_assistant_payload = True
                                        mark_progress("content")
                                        streaming_content_chunks.append(token)  # Track for retry detection
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

                                    # Thinking/reasoning delta (for thinking models).
                                    # Ollama emits thinking tokens in a separate field for thinking models.
                                    # We acknowledge these as valid assistant payload to prevent false
                                    # "no content" errors, but do not forward them to clients by default.
                                    thinking_token = message.get("thinking", "")
                                    if thinking_token:
                                        saw_assistant_payload = True
                                        # Note: thinking tokens are not forwarded to clients unless
                                        # explicitly configured. This prevents the RuntimeError when
                                        # models emit only thinking tokens without content/tool_calls.

                                    if data.get("done"):
                                        logger.debug("streaming: received done:true from upstream model=%s", model_name)
                                        if not saw_assistant_payload:
                                            # Diagnostic logging for empty stream failure mode
                                            logger.error(
                                                "Empty stream for model %s: no thinking/content/tool_calls received. "
                                                "Last data chunk: %s",
                                                model_name,
                                                json.dumps(data)[:500]
                                            )
                                            
                                            # Empty stream retry logic: attempt once with context recovery
                                            empty_stream_retry_count += 1
                                            if empty_stream_retry_count <= 1:
                                                logger.warning(
                                                    "Empty stream detected for model %s, attempting recovery with context trim (attempt %d/1)",
                                                    model_name,
                                                    empty_stream_retry_count
                                                )
                                                try:
                                                    recovery_result = apply_overflow_recovery(
                                                        payload["messages"], model_name, policy_entry
                                                    )
                                                    trimmed_messages = recovery_result.messages
                                                    was_trimmed = recovery_result.changed
                                                    logger.info(
                                                        "Empty stream recovery model=%s stage=%s attempts=%d changed=%s success_candidate=%s",
                                                        model_name,
                                                        recovery_result.final_stage,
                                                        len(recovery_result.attempts),
                                                        recovery_result.changed,
                                                        recovery_result.success_candidate,
                                                    )
                                                    if was_trimmed:
                                                        logger.info(
                                                            "Empty stream recovery applied for model %s, retrying with %d messages",
                                                            model_name,
                                                            len(trimmed_messages)
                                                        )
                                                        payload["messages"] = trimmed_messages
                                                        retry_stream = True
                                                        break  # Break from inner while loop to retry outer while loop
                                                    else:
                                                        logger.warning(
                                                            "Empty stream recovery did not reduce messages for model %s; raising error",
                                                            model_name
                                                        )
                                                except Exception as recovery_exc:
                                                    logger.error(
                                                        "Empty stream recovery failed for model %s: %s",
                                                        model_name,
                                                        recovery_exc,
                                                        exc_info=True
                                                    )
                                            
                                            raise RuntimeError(
                                                f"Ollama stream for model {model_name} completed without any assistant content or tool calls"
                                            )
                                        
                                        # Check for tool-call retry condition
                                        model_entry = MODEL_POLICY.get(model_name, {})
                                        
                                        if _should_retry_tool_call(body, model_entry) and not saw_tool_calls:
                                            # Accumulate content and check for tool mentions
                                            streaming_content = "".join(streaming_content_chunks)
                                            tool_names = _extract_tool_names(body)
                                            
                                            if _detect_tool_call_failure(streaming_content, tool_names, False):
                                                logger.info(
                                                    "router: streaming tool-call failure detected model=%s content_mentions_tool=true",
                                                    model_name,
                                                )
                                                
                                                # Perform retry
                                                retry_result = await _perform_tool_call_retry(
                                                    body=body,
                                                    model_name=model_name,
                                                    model_entry=model_entry,
                                                    original_payload=payload,
                                                )
                                                
                                                if retry_result["success"]:
                                                    # Emit retry result as streaming chunks
                                                    retry_message = retry_result["result"].get("message", {})
                                                    retry_content = retry_message.get("content", "")
                                                    retry_tool_calls = _translate_tool_calls(retry_message.get("tool_calls") or [])
                                                    
                                                    # Emit content chunk
                                                    if retry_content:
                                                        content_chunk = {
                                                            "id": completion_id,
                                                            "object": "chat.completion.chunk",
                                                            "created": created,
                                                            "model": model,
                                                            "choices": [{
                                                                "index": 0,
                                                                "delta": {"content": retry_content},
                                                                "finish_reason": None,
                                                            }],
                                                        }
                                                        await queue.put(f"data: {json.dumps(content_chunk)}\n\n")
                                                    
                                                    # Emit tool calls chunk
                                                    if retry_tool_calls:
                                                        tc_deltas = _streaming_tool_call_deltas(retry_tool_calls)
                                                        if tc_deltas:
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
                                                            await queue.put(f"data: {json.dumps(tc_chunk)}\n\n")
                                                            saw_tool_calls = True
                                                    
                                                    # Update finish reason
                                                    finish_reason = "tool_calls" if saw_tool_calls else "stop"
                                                else:
                                                    # Retry failed, emit error
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
                                                            "message": "Model failed to produce structured tool_calls after retry",
                                                            "type": "tool_call_retry_failed",
                                                            "code": "retry_failed",
                                                        },
                                                    }
                                                    await queue.put(f"data: {json.dumps(err_chunk)}\n\n")
                                                    finish_reason = "error"
                                            else:
                                                # No retry needed, use original finish reason
                                                if saw_tool_calls:
                                                    finish_reason = "tool_calls"
                                                else:
                                                    dr = data.get("done_reason")
                                                    finish_reason = (
                                                        dr if dr in ("stop", "length", "content_filter")
                                                        else "stop"
                                                    )
                                        else:
                                            # No retry condition, use original finish reason
                                            if saw_tool_calls:
                                                finish_reason = "tool_calls"
                                            else:
                                                dr = data.get("done_reason")
                                                finish_reason = (
                                                    dr if dr in ("stop", "length", "content_filter")
                                                    else "stop"
                                                )
                                        
                                        mark_progress("done")
                                        usage_chunk = _build_stream_usage_chunk(
                                            include_usage=include_stream_usage,
                                            completion_id=completion_id,
                                            created=created,
                                            model=model,
                                            done_data=data,
                                        )
                                        if usage_chunk is not None:
                                            await queue.put(f"data: {json.dumps(usage_chunk)}\n\n")

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
                                        if not stream_state.get("finalized"):
                                            await queue.put("data: [DONE]\n\n")
                                            stream_state["finalized"] = True
                                            logger.debug("streaming: emitted [DONE] terminator (normal completion)")
                                        stream_state["stream_state"] = "completed"
                                        return

                                if retry_stream:
                                    continue

                                # Stream completed successfully; exit retry loop.
                                stream_state["stream_state"] = "completed"
                                return
                except Exception as exc:
                    logger.warning("streaming: exception in reader task: %s", exc, exc_info=True)
                    error_response = _safe_error_response(exc, "streaming: reader task")
                    if stream_state["stream_state"] == "opening":
                        stream_state["stream_state"] = "failed_before_progress"
                    error_message = str(exc) if isinstance(exc, TimeoutError) else error_response["error"]["message"]
                    error_type = "ollama_model_evicted" if isinstance(exc, TimeoutError) else error_response["error"]["type"]
                    error_code = "ollama_model_evicted" if isinstance(exc, TimeoutError) else error_response["error"]["code"]
                    if isinstance(exc, RuntimeError) and "completed without any assistant content or tool calls" in str(exc):
                        error_message = str(exc)
                        error_type = "empty_assistant_response"
                        error_code = "empty_assistant_response"
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
                            "message": error_message,
                            "type": error_type,
                            "code": error_code,
                        },
                    }
                    await queue.put(f"data: {json.dumps(err_chunk)}\n\n")
                    logger.debug("streaming: emitted error chunk (exception path)")
                    if not stream_state.get("finalized"):
                        await queue.put("data: [DONE]\n\n")
                        stream_state["finalized"] = True
                        logger.debug("streaming: emitted [DONE] terminator (exception path)")
                finally:
                    logger.debug("streaming: final stream state=%s", stream_state)
                    await queue.put(SENTINEL)
                    logger.debug("streaming: queued sentinel")

            hb_task = asyncio.create_task(heartbeat())
            rd_task = asyncio.create_task(reader())
            try:
                while True:
                    item = await queue.get()
                    if item is SENTINEL:
                        logger.debug("streaming: downstream stream closed")
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
        # Context overflow recovery pipeline: detect context error and retry with bounded attempts.
        context_retry_count = 0
        while response.status_code >= 400 and is_context_overflow_error(response.status_code, response.text):
            if context_retry_count >= HEADROOM_RECOVERY_MAX_RETRIES:
                break
            context_retry_count += 1
            logger.warning(
                "Context overflow detected for model %s (status=%d), applying recovery trim and retrying (attempt %d/%d)",
                model_name,
                response.status_code,
                context_retry_count,
                HEADROOM_RECOVERY_MAX_RETRIES,
            )
            try:
                recovery_result = apply_overflow_recovery(
                    payload["messages"], model_name, policy_entry
                )
                trimmed_messages = recovery_result.messages
                was_trimmed = recovery_result.changed
                logger.info(
                    "Overflow recovery nonstream model=%s stage=%s attempts=%d changed=%s success_candidate=%s",
                    model_name,
                    recovery_result.final_stage,
                    len(recovery_result.attempts),
                    recovery_result.changed,
                    recovery_result.success_candidate,
                )
                if not was_trimmed:
                    logger.warning("Overflow recovery did not reduce messages for model %s; propagating upstream error", model_name)
                    break

                logger.info("Overflow recovery applied for model %s, retrying with %d messages", model_name, len(trimmed_messages))
                payload["messages"] = trimmed_messages
                response = await _ollama_post("/api/chat", payload, stream=False)
                logger.debug("Ollama /api/chat retry response for non-streaming: status %d", response.status_code)
            except Exception as trim_exc:
                logger.error("Overflow recovery failed for model %s: %s", model_name, trim_exc, exc_info=True)
                break

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

    # Check for tool-call retry condition
    model_entry = MODEL_POLICY.get(model_name, {})
    
    if _should_retry_tool_call(body, model_entry):
        content_check = ollama_message.get("content", "") or ""
        tool_calls_check = _translate_tool_calls(ollama_message.get("tool_calls") or [])
        
        if not tool_calls_check:
            # Extract tool names from request
            tool_names = _extract_tool_names(body)
            
            # Check if assistant content mentions any tool
            if _detect_tool_call_failure(content_check, tool_names, False):
                logger.info(
                    "router: tool-call failure detected model=%s content_mentions_tool=true",
                    model_name,
                )
                
                # Perform retry
                retry_result = await _perform_tool_call_retry(
                    body=body,
                    model_name=model_name,
                    model_entry=model_entry,
                    original_payload=payload,
                )
                
                if retry_result["success"]:
                    result = retry_result["result"]
                    ollama_message = result.get("message") or {}
                else:
                    # Retry failed, return protocol error
                    raise HTTPException(
                        status_code=502,
                        detail={
                            "error": "tool_call_retry_failed",
                            "message": "Model failed to produce structured tool_calls after retry",
                            "retry_attempted": True,
                            "retry_error": retry_result.get("error"),
                        },
                    )

    content = ollama_message.get("content", "") or ""
    tool_calls = _translate_tool_calls(ollama_message.get("tool_calls") or [])

    if not content and not tool_calls:
        logger.warning(
            "Ollama returned an empty assistant response for model %s",
            model,
        )
        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty assistant response",
        )

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
