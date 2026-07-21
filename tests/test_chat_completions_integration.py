import asyncio
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROUTER_DIR = Path(__file__).resolve().parents[1]
os.environ["MODEL_POLICY_FILE"] = str(ROUTER_DIR / "model_policy.yml")
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

        httpx.Timeout = Timeout
        httpx.AsyncClient = AsyncClient
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
        self.usable_prompt_budget = 0
        self.trim_reason = None
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
        self.text = ""

    def json(self):
        return self._payload


async def _collect_stream(gen):
    out = []
    async for item in gen:
        out.append(item)
    return out


class ChatCompletionIntegrationTests(unittest.TestCase):
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

    def test_nonstream_resolves_headroom_retrieve_transparently(self):
        body = {
            "model": "qwen3:4b",
            "stream": False,
            "messages": [{"role": "user", "content": "continue"}],
        }
        first = _FakeHTTPResponse(
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_hr",
                            "type": "function",
                            "function": {"name": "headroom_retrieve", "arguments": {"hash": "abc123"}},
                        }
                    ],
                }
            }
        )
        second = {
            "message": {"content": "resolved answer"},
            "prompt_eval_count": 10,
            "eval_count": 4,
        }
        calls = {"n": 0}

        async def _fake_ollama_post(path: str, payload: dict, stream: bool = False):
            calls["n"] += 1
            if calls["n"] == 1:
                return first
            return _FakeHTTPResponse(second)

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app, "_ollama_post", new=_fake_ollama_post),
            mock.patch.object(app, "_continue_after_headroom_retrieve", new=mock.AsyncMock(return_value=second)),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content["choices"][0]["message"]["content"], "resolved answer")

    def test_stream_resolves_headroom_retrieve_without_leak(self):
        _FakeAsyncClient.stream_lines = [
            '{"message": {"tool_calls": [{"id": "call_hr", "function": {"name": "headroom_retrieve", "arguments": {"hash": "abc123"}}}]}}',
            '{"done": true, "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 1}',
        ]

        body = {
            "model": "qwen3:4b",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        }

        continued = {
            "message": {"content": "resolved stream"},
            "done_reason": "stop",
            "prompt_eval_count": 6,
            "eval_count": 2,
        }

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "check_and_trim", return_value=_HeadroomResult(rejected=False, trimmed=False, messages=body["messages"])),
            mock.patch.object(app.httpx, "AsyncClient", _FakeAsyncClient),
            mock.patch.object(app, "_continue_after_headroom_retrieve", new=mock.AsyncMock(return_value=continued)),
        ):
            response = asyncio.run(app.chat_completions(_FakeRequest(body)))
            frames = asyncio.run(_collect_stream(response.content))

        self.assertTrue(any("resolved stream" in f for f in frames))
        self.assertFalse(any("headroom_retrieve" in f for f in frames))


if __name__ == "__main__":
    unittest.main()
