import asyncio
import os
from pathlib import Path
import sys
import tempfile
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
from ingest_repo import collect_repo_chunks  # noqa: E402
from retrieval import format_retrieval_context, RetrievedChunk  # noqa: E402


class RetrievalTests(unittest.TestCase):
    def test_format_retrieval_context_renders_chunks(self):
        ctx = format_retrieval_context([
            RetrievedChunk(
                repo="repo",
                branch="main",
                path="src/app.py",
                symbol="foo",
                score=0.9,
                chunk_text="print('hello')",
                line_start=1,
                line_end=2,
                language="python",
            )
        ])
        self.assertIn("src/app.py::foo [1-2]", ctx)
        self.assertIn("print('hello')", ctx)

    def test_inject_retrieval_context_prepends_system_message(self):
        body = {"retrieval": {"repo": "repo", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        async def _fake_retrieve_context(**kwargs):
            return [
                {
                    "repo": "repo",
                    "branch": "main",
                    "path": "src/app.py",
                    "symbol": "foo",
                    "score": 0.9,
                    "chunk_text": "retrieved text",
                    "line_start": 1,
                    "line_end": 2,
                    "language": "python",
                }
            ]

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context", new=_fake_retrieve_context),
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        self.assertEqual(injected[0]["role"], "system")
        self.assertIn("Retrieved context:", injected[0]["content"])
        self.assertEqual(injected[1], messages[0])

    def test_collect_repo_chunks_extracts_python_functions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir)
            (repo_dir / "demo.py").write_text(
                """
import os

def alpha():
    return 1

class Beta:
    def gamma(self):
        return 2
""".strip(),
                encoding="utf-8",
            )
            chunks = collect_repo_chunks(repo_dir)

        symbols = {chunk.symbol for chunk in chunks if chunk.path.endswith("demo.py")}
        self.assertIn("alpha", symbols)
        self.assertIn("Beta", symbols)

    def test_qdrant_disabled_passthrough(self):
        """Verify that when ENABLE_QDRANT_RETRIEVAL=false, messages are returned unchanged."""
        body = {"retrieval": {"repo": "repo", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", False),
            mock.patch.object(app, "retrieve_context") as mock_retrieve,
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        self.assertEqual(injected, messages)
        mock_retrieve.assert_not_called()

    def test_qdrant_unavailable_fails_open(self):
        """Verify that when Qdrant is unavailable, messages are returned unchanged."""
        body = {"retrieval": {"repo": "repo", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context") as mock_retrieve,
        ):
            mock_retrieve.side_effect = Exception("Connection refused")
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        self.assertEqual(injected, messages)

    def test_retrieval_empty_repo_returns_unchanged(self):
        """Verify that when repo is empty, messages are returned unchanged."""
        body = {"retrieval": {"repo": "", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context") as mock_retrieve,
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        self.assertEqual(injected, messages)
        mock_retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
