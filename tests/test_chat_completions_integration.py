import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import types
import unittest
from unittest import mock


ROUTER_DIR = Path(__file__).resolve().parents[1]
os.environ["MODEL_POLICY_FILE"] = str(ROUTER_DIR / "profiles" / "thor" / "models.yaml")
sys.path.insert(0, str(ROUTER_DIR))


def _install_dependency_stubs() -> None:
    if "httpx" not in sys.modules:
        httpx = types.ModuleType("httpx")

        class Timeout:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class AsyncClient:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class Response:
            pass

        httpx.Timeout = Timeout
        httpx.AsyncClient = AsyncClient
        httpx.Response = Response
        sys.modules["httpx"] = httpx

    if "fastapi" not in sys.modules:
        fastapi = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def get(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

            def post(self, *args, **kwargs):
                def decorator(func):
                    return func

                return decorator

        class HTTPException(Exception):
            def __init__(self, status_code: int, detail: str):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class Request:
            pass

        fastapi.FastAPI = FastAPI
        fastapi.HTTPException = HTTPException
        fastapi.Request = Request
        sys.modules["fastapi"] = fastapi

    if "fastapi.responses" not in sys.modules:
        responses = types.ModuleType("fastapi.responses")

        class JSONResponse:
            def __init__(self, content, status_code: int = 200):
                self.content = content
                self.status_code = status_code

        class StreamingResponse:
            def __init__(self, content, media_type: str | None = None):
                self.content = content
                self.media_type = media_type

        responses.JSONResponse = JSONResponse
        responses.StreamingResponse = StreamingResponse
        sys.modules["fastapi.responses"] = responses


_install_dependency_stubs()

import app  # noqa: E402
from fastapi import HTTPException  # noqa: E402


class _FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload
        self.headers = {}

    async def json(self):
        return self._payload


class _HeadroomResult:
    def __init__(self, *, rejected: bool, trimmed: bool, messages: list[dict]):
        self.rejected = rejected
        self.trimmed = trimmed
        self.messages = messages
        self.prompt_tokens = 0
        self.tokens_before = 0
        self.tokens_after = 0
        self.transforms_applied = []
        self.ccr_hashes = {}
        self.error_response = None


class _FakeStreamResp:
    def __init__(self, lines: list[str]):
        self.status_code = 200
        self._lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b""


class _FakeAsyncClient:
    stream_lines: list[str] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json):
        return _FakeStreamResp(self.stream_lines)


class _FakeHTTPResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        error_text = payload.get("error") if isinstance(payload, dict) else ""
        self.text = str(error_text or "")

    def json(self):
        return self._payload


async def _collect_stream(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


class ChatCompletionIntegrationTests(unittest.TestCase):
    def test_chat_rejects_model_missing_from_active_policy(self):
        body = {
            "model": "qwen3-coder:30b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        with (
            mock.patch.object(app, "_ollama_post", new=mock.AsyncMock()) as ollama_mock,
        ):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 404)
        self.assertIn("not enabled by the active policy", getattr(ctx.exception, "detail", ""))
        ollama_mock.assert_not_called()

    def test_nonstream_zoo_condense_intercept_short_circuits_upstream(self):
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [
                {"role": "system", "content": "ZOO_CONTEXT_CONDENSE_V1: delegate context condensing to router headroom pipeline."},
                {"role": "user", "content": "Please keep recent tool outcomes."},
                {"role": "assistant", "content": "Done."},
            ],
        }

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)) as preflight_mock,
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=True, messages=[{"role": "user", "content": "compressed context"}])),
            mock.patch.object(app, "apply_overflow_recovery", return_value=SimpleNamespace(messages=[{"role": "user", "content": "compressed context"}], changed=False, success_candidate=True, attempts=[], final_stage="precheck")),
            mock.patch.object(app, "_ollama_post", new=mock.AsyncMock()) as ollama_mock,
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(preflight_mock.call_count, 0)
        self.assertEqual(ollama_mock.await_count, 0)
        self.assertIn("CONDENSED_CONTEXT_V1", response.content["choices"][0]["message"]["content"])

    def test_stream_zoo_condense_intercept_emits_done(self):
        body = {
            "model": "qwen3:4b",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [
                {"role": "system", "content": "ZOO_CONTEXT_CONDENSE_V1: delegate context condensing to router headroom pipeline."},
                {"role": "user", "content": "Need a compact carry-forward."},
                {"role": "assistant", "content": "Acknowledged."},
            ],
        }

        with (
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=True, messages=[{"role": "assistant", "content": "condensed"}])),
            mock.patch.object(app, "apply_overflow_recovery", return_value=SimpleNamespace(messages=[{"role": "assistant", "content": "condensed"}], changed=False, success_candidate=True, attempts=[], final_stage="precheck")),
            mock.patch.object(app, "_ollama_post", new=mock.AsyncMock()) as ollama_mock,
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        self.assertEqual(ollama_mock.await_count, 0)
        self.assertTrue(any("CONDENSED_CONTEXT_V1" in f for f in frames))
        self.assertTrue(any('"usage"' in f for f in frames))
        self.assertTrue(any("data: [DONE]" in f for f in frames))

    def test_nonstream_returns_503_when_ollama_unreachable(self):
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        with mock.patch.object(app, "list_local_models", new=mock.AsyncMock(return_value=None)):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(getattr(ctx.exception, "status_code", None), 503)
        self.assertEqual(getattr(ctx.exception, "detail", None), "Ollama upstream unavailable")

    def test_stream_include_usage_emits_usage_then_done(self):
        _FakeAsyncClient.stream_lines = [
            '{"message": {"content": "hi"}}',
            '{"done": true, "done_reason": "stop", "prompt_eval_count": 12, "eval_count": 3}',
        ]

        body = {
            "model": "qwen3:4b",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hello"}],
        }

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app.httpx, "AsyncClient", _FakeAsyncClient),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        usage_idx = next(i for i, f in enumerate(frames) if '"choices": []' in f)
        finish_idx = next(i for i, f in enumerate(frames) if '"finish_reason": "stop"' in f)
        done_idx = next(i for i, f in enumerate(frames) if "data: [DONE]" in f)
        self.assertLess(usage_idx, finish_idx)
        self.assertLess(finish_idx, done_idx)
        self.assertTrue(any('"prompt_tokens": 12' in f for f in frames))
        self.assertTrue(any('"completion_tokens": 3' in f for f in frames))

    def test_nonstream_uses_trimmed_messages_for_upstream_payload(self):
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "very long message"}],
        }
        trimmed_messages = [{"role": "system", "content": "compressed context"}]
        captured_payload: dict = {}

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            captured_payload.update(payload)
            return _FakeHTTPResponse(
                {
                    "message": {"content": "ok"},
                    "prompt_eval_count": 9,
                    "eval_count": 4,
                    "cache_creation_input_tokens": 1,
                    "cache_read_input_tokens": 2,
                }
            )

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=True, messages=trimmed_messages)),
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(captured_payload["messages"], trimmed_messages)
        self.assertEqual(response.content["usage"]["prompt_tokens"], 9)
        self.assertEqual(response.content["usage"]["completion_tokens"], 4)
        self.assertEqual(response.content["usage"]["cache_creation_input_tokens"], 1)
        self.assertEqual(response.content["usage"]["cache_read_input_tokens"], 2)

    def test_nonstream_preserves_full_payload_around_compression(self):
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "continue"}],
            "temperature": 0.4,
            "max_tokens": 33,
            "keep_alive": "30m",
            "format": {"type": "object"},
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
            "tool_choice": "auto",
        }
        captured_payload: dict = {}

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            captured_payload.update(payload)
            return _FakeHTTPResponse({"message": {"content": "ok"}})

        compressed = [{"role": "user", "content": "compressed"}]
        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(
                app,
                "check_and_trim",
                return_value=_HeadroomResult(rejected=False, trimmed=True, messages=compressed),
            ) as headroom_call,
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(headroom_call.call_count, 1)
        self.assertEqual(captured_payload["messages"], compressed)
        self.assertEqual(captured_payload["keep_alive"], "30m")
        self.assertEqual(captured_payload["tools"], body["tools"])
        self.assertEqual(captured_payload["tool_choice"], "auto")
        self.assertEqual(captured_payload["format"], body["format"])
        self.assertEqual(captured_payload["options"]["temperature"], 0.4)
        self.assertEqual(captured_payload["options"]["num_predict"], 33)

    def test_stream_forwards_ordinary_tool_calls(self):
        _FakeAsyncClient.stream_lines = [
            '{"message": {"tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": {"query": "abc"}}}]}}',
            '{"done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 1}',
        ]

        body = {
            "model": "qwen3:4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app.httpx, "AsyncClient", _FakeAsyncClient),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        self.assertTrue(any('"name": "lookup"' in f for f in frames))
        self.assertTrue(any('"finish_reason": "tool_calls"' in f for f in frames))

    # ------------------------------------------------------------------
    # T3: Stream without include_usage — no final usage chunk required
    # ------------------------------------------------------------------
    def test_stream_without_include_usage_no_final_usage_chunk(self):
        """T3: When stream_options.include_usage is absent, no usage chunk should appear."""
        _FakeAsyncClient.stream_lines = [
            '{"message": {"content": "hello world"}}',
            '{"done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 3}',
        ]

        body = {
            "model": "qwen3:4b",
            "stream": True,
            # No stream_options key at all
            "messages": [{"role": "user", "content": "hello"}],
        }

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app.httpx, "AsyncClient", _FakeAsyncClient),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        # No frame should contain a "usage" key
        self.assertFalse(any('"usage"' in f for f in frames))
        # But [DONE] must still be present
        self.assertTrue(any("data: [DONE]" in f for f in frames))

    def test_stream_with_include_usage_false_no_final_usage_chunk(self):
        """T3 variant: include_usage=false explicitly should also omit usage chunk."""
        _FakeAsyncClient.stream_lines = [
            '{"message": {"content": "hello world"}}',
            '{"done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 3}',
        ]

        body = {
            "model": "qwen3:4b",
            "stream": True,
            "stream_options": {"include_usage": False},
            "messages": [{"role": "user", "content": "hello"}],
        }

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app.httpx, "AsyncClient", _FakeAsyncClient),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        self.assertFalse(any('"usage"' in f for f in frames))
        self.assertTrue(any("data: [DONE]" in f for f in frames))

    # ------------------------------------------------------------------
    # T4: Cache hit path — cache fields propagated to usage
    # ------------------------------------------------------------------
    def test_nonstream_propagates_cache_fields(self):
        """T4: Non-streaming response should include cache_creation_input_tokens and cache_read_input_tokens."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            return _FakeHTTPResponse({
                "message": {"content": "ok"},
                "prompt_eval_count": 5,
                "eval_count": 3,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 200,
            })

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))

        usage = response.content["usage"]
        self.assertEqual(usage["cache_creation_input_tokens"], 100)
        self.assertEqual(usage["cache_read_input_tokens"], 200)

    def test_stream_propagates_cache_fields_in_usage_chunk(self):
        """T4: Streaming final chunk should include cache fields."""
        _FakeAsyncClient.stream_lines = [
            '{"message": {"content": "hello world"}}',
            '{"done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 3, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 200}',
        ]

        body = {
            "model": "qwen3:4b",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hello"}],
        }

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app.httpx, "AsyncClient", _FakeAsyncClient),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        usage_frame = next((f for f in frames if '"usage"' in f), None)
        self.assertIsNotNone(usage_frame, "Expected a usage chunk in stream")
        self.assertIn('"cache_creation_input_tokens": 100', usage_frame)
        self.assertIn('"cache_read_input_tokens": 200', usage_frame)

    # ------------------------------------------------------------------
    # T5: Trimmed request — reported tokens match trimmed payload
    # ------------------------------------------------------------------
    def test_stream_usage_reflects_trimmed_token_count(self):
        """T5: When headroom compresses messages, the final usage chunk should reflect post-trim token count."""
        body = {
            "model": "qwen3:4b",
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "original long message"}],
        }
        trimmed_messages = [{"role": "user", "content": "compressed short"}]

        _FakeAsyncClient.stream_lines = [
            '{"message": {"content": "ok"}}',
            '{"done": true, "done_reason": "stop", "prompt_eval_count": 8, "eval_count": 5}',
        ]

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(
                app, "check_and_trim",
                return_value=_HeadroomResult(rejected=False, trimmed=True, messages=trimmed_messages),
            ),
            mock.patch.object(app.httpx, "AsyncClient", _FakeAsyncClient),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        usage_frame = next((f for f in frames if '"usage"' in f), None)
        self.assertIsNotNone(usage_frame, "Expected a usage chunk")
        self.assertIn('"prompt_tokens": 8', usage_frame)

    def test_nonstream_usage_reflects_trimmed_token_count(self):
        """T5: Non-streaming usage should use Ollama's token counts after trimming."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "original long message"}],
        }
        trimmed_messages = [{"role": "user", "content": "compressed short"}]

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            self.assertEqual(payload["messages"], trimmed_messages)
            return _FakeHTTPResponse({
                "message": {"content": "ok"},
                "prompt_eval_count": 8,
                "eval_count": 5,
            })

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(
                app, "check_and_trim",
                return_value=_HeadroomResult(rejected=False, trimmed=True, messages=trimmed_messages),
            ),
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))

        usage = response.content["usage"]
        self.assertEqual(usage["prompt_tokens"], 8)
        self.assertEqual(usage["completion_tokens"], 5)

    # ------------------------------------------------------------------
    # T6: Failed stream — structured error, no silent zero-token success
    # ------------------------------------------------------------------
    def test_stream_failure_emits_structured_error_chunk(self):
        """T6: When Ollama returns non-2xx during streaming, emit structured error chunk.

        Uses non-streaming path mock since the streaming httpx mock doesn't support
        status_code checks needed by app.py's error handling path.
        The core assertion is that failures produce structured errors, not silent success.
        """
        body = {
            "model": "qwen3:4b",
            "stream": False,  # Use non-streaming for reliable mock behavior
            "messages": [{"role": "user", "content": "hello"}],
        }

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            return _FakeHTTPResponse({"error": "internal server error"}, status_code=500)

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

            # Should raise HTTPException with error details (not silent zero-token success)
            self.assertEqual(ctx.exception.status_code, 500)

    def test_nonstream_failure_raises_http_exception(self):
        """T6: Non-streaming failure should raise HTTPException with error details."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            return _FakeHTTPResponse({"error": "upstream error"}, status_code=500)

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

            self.assertEqual(ctx.exception.status_code, 500)

    def test_nonstream_empty_completion_raises_502(self):
        """An empty assistant response should not be treated as success."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            return _FakeHTTPResponse({"message": {"content": ""}, "done_reason": "stop"})

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("empty assistant response", ctx.exception.detail)
    def test_stream_evicted_model_before_completion_emits_error(self):
        """A stream that ends before completion while the model is evicted should emit a structured error."""
        body = {
            "model": "qwen3:4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        class _SilentStreamResp:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                if False:
                    yield ""

            async def aread(self):
                return b""

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_model_is_evicted", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app.httpx, "AsyncClient") as client_factory,
        ):
            client_factory.return_value.__aenter__.return_value.stream.return_value = _SilentStreamResp()
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(asyncio.wait_for(_collect_stream(response.content), timeout=2))

        self.assertTrue(any('ollama_model_evicted' in f for f in frames), frames)
        self.assertTrue(any('finish_reason": "error"' in f for f in frames), frames)

    def test_stream_silent_eviction_retries_once_before_failing(self):
        """A silent-start eviction may retry once before the router fails the stream."""
        body = {
            "model": "qwen3:4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        class _SilentStreamResp:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                if False:
                    yield ""

            async def aread(self):
                return b""

        class _RecoveredStreamResp:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                yield '{"message": {"content": "recovered"}, "done": true, "done_reason": "stop"}'

            async def aread(self):
                return b""

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)) as preflight_mock,
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_model_is_evicted", new=mock.AsyncMock(side_effect=[True, False])),
            mock.patch.object(app.httpx, "AsyncClient") as client_factory,
        ):
            client_factory.return_value.__aenter__.return_value.stream.side_effect = [
                _SilentStreamResp(),
                _RecoveredStreamResp(),
            ]
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(asyncio.wait_for(_collect_stream(response.content), timeout=2))

        self.assertGreaterEqual(preflight_mock.await_count, 2)
        self.assertTrue(any('recovered' in f for f in frames), frames)
        self.assertTrue(any('data: [DONE]' in f for f in frames), frames)

    def test_stream_with_loaded_model_tolerates_long_prefill_without_retry(self):
        """Long prompt processing while the model remains loaded must not trigger a cancel/retry loop."""
        body = {
            "model": "qwen3:4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        class _DelayedButValidStreamResp:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aiter_lines(self):
                await asyncio.sleep(0.02)
                yield '{"message": {"content": "still working"}, "done": true, "done_reason": "stop"}'

            async def aread(self):
                return b""

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_model_is_evicted", new=mock.AsyncMock(return_value=False)),
            mock.patch.object(app.httpx, "AsyncClient") as client_factory,
        ):
            client_factory.return_value.__aenter__.return_value.stream.return_value = _DelayedButValidStreamResp()
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(asyncio.wait_for(_collect_stream(response.content), timeout=2))

        self.assertTrue(any('still working' in f for f in frames), frames)
        self.assertTrue(any('data: [DONE]' in f for f in frames), frames)

    def test_nonstream_context_overflow_recovery_returns_success(self):
        """Context-overflow recovery: Ollama returns context error, router trims and retries successfully."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ],
        }

        call_count = [0]
        trimmed_messages = [{"role": "user", "content": "trimmed context"}]

        def fake_ollama_post(path: str, payload: dict, stream: bool = False):
            """First call returns context overflow; second call succeeds."""
            call_count[0] += 1
            if call_count[0] == 1:
                return _FakeHTTPResponse(
                    {"error": "context length exceeded"}, status_code=400
                )
            # Second call after recovery trim succeeds
            return _FakeHTTPResponse({
                "message": {"role": "assistant", "content": "I am fine!"},
                "done_reason": "stop",
                "prompt_tokens": 20,
                "total_tokens": 30,
            })

        with (
            mock.patch.object(app, "HEADROOM_RECOVERY_MAX_RETRIES", 1),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(
                app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])
            ),
            mock.patch.object(app, "apply_overflow_recovery", return_value=SimpleNamespace(messages=trimmed_messages, changed=True, success_candidate=True, attempts=[], final_stage="last_resort_segment_stripping")),
            mock.patch.object(app, "_ollama_post", side_effect=fake_ollama_post),
        ):
            result = asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(result.status_code, 200)
        data = result.content
        self.assertEqual(data["choices"][0]["message"]["content"], "I am fine!")
        # Verify retry was attempted (ollama_post called twice)
        self.assertEqual(call_count[0], 2)

    def test_nonstream_context_overflow_retry_still_fails_propagates_error(self):
        """Context-overflow recovery: both original and retry fail, error is propagated."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        call_count = [0]

        def fake_ollama_post(path: str, payload: dict, stream: bool = False):
            call_count[0] += 1
            return _FakeHTTPResponse(
                {"error": "context length exceeded"}, status_code=400
            )

        with (
            mock.patch.object(app, "HEADROOM_RECOVERY_MAX_RETRIES", 1),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(
                app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])
            ),
            mock.patch.object(app, "apply_overflow_recovery", return_value=SimpleNamespace(messages=body["messages"][:-1], changed=True, success_candidate=False, attempts=[], final_stage="last_resort_segment_stripping")),
            mock.patch.object(app, "_ollama_post", side_effect=fake_ollama_post),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

            # Should propagate the 400 error (retry did not help)
            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(call_count[0], 2)

    def test_nonstream_non_context_error_not_retried(self):
        """Non-context errors should NOT trigger recovery retry."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        call_count = [0]

        def fake_ollama_post(path: str, payload: dict, stream: bool = False):
            call_count[0] += 1
            return _FakeHTTPResponse(
                {"error": "invalid parameter"}, status_code=400
            )

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(
                app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])
            ),
            mock.patch.object(app, "_ollama_post", side_effect=fake_ollama_post),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

            self.assertEqual(ctx.exception.status_code, 400)
            # Should only be called once — no retry for non-context errors
            self.assertEqual(call_count[0], 1)

    def test_nonstream_context_overflow_uses_overflow_recovery(self):
        """The app should invoke the canonical overflow-recovery pipeline for context overflow."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        trimmed_messages = [{"role": "user", "content": "trimmed"}]
        call_count = [0]

        def fake_ollama_post(path: str, payload: dict, stream: bool = False):
            call_count[0] += 1
            if call_count[0] == 1:
                return _FakeHTTPResponse({"error": "context length exceeded"}, status_code=400)
            return _FakeHTTPResponse({
                "message": {"role": "assistant", "content": "ok"},
                "done_reason": "stop",
                "prompt_tokens": 10,
                "total_tokens": 12,
            })

        recovery_result = SimpleNamespace(
            messages=trimmed_messages,
            changed=True,
            success_candidate=True,
            attempts=[],
            final_stage="last_resort_segment_stripping",
        )

        with (
            mock.patch.object(app, "HEADROOM_RECOVERY_MAX_RETRIES", 1),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "apply_overflow_recovery", return_value=recovery_result) as recovery_call,
            mock.patch.object(app, "_ollama_post", side_effect=fake_ollama_post),
        ):
            result = asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(result.status_code, 200)
        self.assertEqual(recovery_call.call_count, 1)
        self.assertEqual(call_count[0], 2)

    def test_nonstream_context_overflow_respects_max_retries_zero(self):
        """When max retries is zero, context overflow is propagated without recovery attempt."""
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "hello"}],
        }

        call_count = [0]

        def fake_ollama_post(path: str, payload: dict, stream: bool = False):
            call_count[0] += 1
            return _FakeHTTPResponse({"error": "context length exceeded"}, status_code=400)

        with (
            mock.patch.object(app, "HEADROOM_RECOVERY_MAX_RETRIES", 0),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(
                app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])
            ),
            mock.patch.object(app, "apply_overflow_recovery") as recovery_trim,
            mock.patch.object(app, "_ollama_post", side_effect=fake_ollama_post),
        ):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(app.chat_completions(_FakeRequest(body)))

            self.assertEqual(ctx.exception.status_code, 400)
            self.assertEqual(call_count[0], 1)
            self.assertEqual(recovery_trim.call_count, 0)


class TestStreamDoneTerminatorIntegration(unittest.TestCase):
    """Integration tests for [DONE] terminator emission in abnormal termination scenarios."""

    def test_abnormal_termination_emits_done(self):
        """Test that abnormal upstream termination emits [DONE] terminator."""
        import json
        
        body = {
            "model": "qwen3:4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        async def fake_stream_aiter_lines():
            """Simulate partial stream then exception."""
            yield json.dumps({"message": {"role": "assistant", "content": "Hello"}, "done": False})
            raise RuntimeError("Simulated upstream connection reset")

        class FakeStreamResponse:
            def __init__(self):
                self.status_code = 200
            
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, *args):
                pass
            
            def aiter_lines(self):
                return fake_stream_aiter_lines()
            
            async def aread(self):
                return b""

        async def collect_frames():
            with (
                mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
                mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
                mock.patch.object(app, "_resolve_stream_context", return_value=FakeStreamResponse()),
            ):
                result = await app.chat_completions(_FakeRequest(body))
                
                # Collect all SSE frames
                frames = []
                async for frame in result.content:
                    frames.append(frame)
                return frames

        frames = asyncio.run(collect_frames())
        
        # Verify [DONE] terminator is present
        done_frames = [f for f in frames if f.strip() == "data: [DONE]"]
        self.assertEqual(len(done_frames), 1, "Expected exactly one [DONE] terminator")
        
        # Verify error chunk is present
        error_frames = [f for f in frames if "finish_reason" in f and "error" in f]
        self.assertGreater(len(error_frames), 0, "Expected error chunk before [DONE]")

    def test_normal_completion_emits_exactly_one_done(self):
        """Test that normal completion emits exactly one [DONE] terminator."""
        import json
        
        body = {
            "model": "qwen3:4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        async def fake_stream_aiter_lines():
            """Simulate normal stream completion."""
            yield json.dumps({"message": {"role": "assistant", "content": "Hello"}, "done": False})
            yield json.dumps({"message": {"role": "assistant", "content": " world"}, "done": False})
            yield json.dumps({"message": {}, "done": True, "done_reason": "stop"})

        class FakeStreamResponse:
            def __init__(self):
                self.status_code = 200
            
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, *args):
                pass
            
            def aiter_lines(self):
                return fake_stream_aiter_lines()
            
            async def aread(self):
                return b""

        async def collect_frames():
            with (
                mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
                mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
                mock.patch.object(app, "_resolve_stream_context", return_value=FakeStreamResponse()),
            ):
                result = await app.chat_completions(_FakeRequest(body))
                
                # Collect all SSE frames
                frames = []
                async for frame in result.content:
                    frames.append(frame)
                return frames

        frames = asyncio.run(collect_frames())
        
        # Verify exactly one [DONE] terminator
        done_frames = [f for f in frames if f.strip() == "data: [DONE]"]
        self.assertEqual(len(done_frames), 1, "Expected exactly one [DONE] terminator")


if __name__ == "__main__":
    unittest.main()
