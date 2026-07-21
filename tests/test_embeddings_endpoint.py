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

    async def json(self):
        return self._payload


class EmbeddingsEndpointTests(unittest.TestCase):
    def test_single_input_embedding_shape(self):
        async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
            self.assertEqual(model, "qwen3-embedding:4b")
            self.assertEqual(prompt, "hello")
            return [0.1, 0.2]

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)) as preflight,
            mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
        ):
            response = asyncio.run(
                app.embeddings(_FakeRequest({"model": "qwen3-embedding:4b", "input": "hello"}))
            )

        preflight.assert_awaited_once_with("qwen3-embedding:4b")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content["object"], "list")
        self.assertEqual(response.content["model"], "qwen3-embedding:4b")
        self.assertEqual(len(response.content["data"]), 1)
        self.assertEqual(response.content["data"][0]["index"], 0)
        self.assertEqual(response.content["data"][0]["embedding"], [0.1, 0.2])

    def test_batch_input_embedding_shape(self):
        async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
            return [float(len(prompt))]

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
        ):
            response = asyncio.run(
                app.embeddings(
                    _FakeRequest(
                        {
                            "model": "qwen3-embedding:4b",
                            "input": ["hello", "world"],
                        }
                    )
                )
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.content["data"]), 2)
        self.assertEqual(response.content["data"][0]["index"], 0)
        self.assertEqual(response.content["data"][1]["index"], 1)

    def test_default_embedding_model_used_when_model_missing(self):
        async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
            self.assertEqual(model, app.EMBEDDING_MODEL_DEFAULT)
            return [0.3]

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
        ):
            response = asyncio.run(app.embeddings(_FakeRequest({"input": "hello"})))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content["model"], app.EMBEDDING_MODEL_DEFAULT)

    def test_embedding_preflight_failure_auto_pull_disabled(self):
        with (
            mock.patch.object(app, "AUTO_PULL_MISSING_MODELS", False),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=False)),
        ):
            with self.assertRaises(Exception) as ctx:
                asyncio.run(
                    app.embeddings(
                        _FakeRequest({"model": "qwen3-embedding:4b", "input": "hello"})
                    )
                )
        self.assertEqual(getattr(ctx.exception, "status_code", None), 404)


if __name__ == "__main__":
    unittest.main()
