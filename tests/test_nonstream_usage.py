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

import app  # noqa: E402


class NonStreamUsageTests(unittest.TestCase):
    def test_uses_ollama_counts_and_cache_fields(self):
        usage = app._build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 11,
                "eval_count": 7,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        self.assertEqual(usage["prompt_tokens"], 11)
        self.assertEqual(usage["completion_tokens"], 7)
        self.assertEqual(usage["total_tokens"], 18)
        self.assertEqual(usage["cache_creation_input_tokens"], 3)
        self.assertEqual(usage["cache_read_input_tokens"], 2)

    def test_falls_back_to_tokenizer_when_ollama_counts_missing(self):
        usage = app._build_non_stream_usage(
            ollama_result={},
            payload_messages=[{"role": "user", "content": "abcd" * 10}],
            model="qwen3:4b",
            completion_text="abcd" * 5,
        )
        self.assertGreaterEqual(usage["prompt_tokens"], 0)
        self.assertGreaterEqual(usage["completion_tokens"], 0)
        self.assertEqual(usage["total_tokens"], usage["prompt_tokens"] + usage["completion_tokens"])
        self.assertEqual(usage["cache_creation_input_tokens"], 0)
        self.assertEqual(usage["cache_read_input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
