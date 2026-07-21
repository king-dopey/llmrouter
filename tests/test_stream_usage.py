import os
from pathlib import Path
import sys
import types
import unittest


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

from app import _build_stream_usage_chunk  # noqa: E402


class StreamUsageChunkTests(unittest.TestCase):
    def test_stream_usage_chunk_included(self):
        chunk = _build_stream_usage_chunk(
            include_usage=True,
            completion_id="chatcmpl-test",
            created=123,
            model="qwen",
            done_data={"prompt_eval_count": 10, "eval_count": 5},
        )
        self.assertIsNotNone(chunk)
        self.assertEqual(chunk["choices"], [])
        self.assertEqual(chunk["usage"]["prompt_tokens"], 10)
        self.assertEqual(chunk["usage"]["completion_tokens"], 5)
        self.assertEqual(chunk["usage"]["total_tokens"], 15)
        self.assertEqual(chunk["usage"]["cache_creation_input_tokens"], 0)
        self.assertEqual(chunk["usage"]["cache_read_input_tokens"], 0)

    def test_stream_usage_chunk_omitted_when_not_requested(self):
        chunk = _build_stream_usage_chunk(
            include_usage=False,
            completion_id="chatcmpl-test",
            created=123,
            model="qwen",
            done_data={"prompt_eval_count": 10, "eval_count": 5},
        )
        self.assertIsNone(chunk)


if __name__ == "__main__":
    unittest.main()
