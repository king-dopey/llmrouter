"""Tests for retrieval module with pytest markers."""

import asyncio
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

import pytest


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


@pytest.mark.retrieval
class TestRetrievalChunkFormatting:
    """Tests for format_retrieval_context() output."""

    def test_format_retrieval_context_renders_chunks(self):
        """Test that format_retrieval_context renders chunk information."""
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
        assert "src/app.py::foo [1-2]" in ctx
        assert "print('hello')" in ctx

    def test_format_retrieval_context_with_multiple_chunks(self):
        """Test formatting with multiple chunks."""
        chunks = [
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
            ),
            RetrievedChunk(
                repo="repo",
                branch="main",
                path="src/utils.py",
                symbol="bar",
                score=0.8,
                chunk_text="def bar(): pass",
                line_start=5,
                line_end=6,
                language="python",
            ),
        ]
        ctx = format_retrieval_context(chunks)
        assert "src/app.py::foo [1-2]" in ctx
        assert "src/utils.py::bar [5-6]" in ctx

    def test_format_retrieval_context_without_branch(self):
        """Test formatting without branch information."""
        ctx = format_retrieval_context([
            RetrievedChunk(
                repo="repo",
                branch=None,
                path="src/app.py",
                symbol="foo",
                score=0.9,
                chunk_text="print('hello')",
                line_start=1,
                line_end=2,
                language="python",
            )
        ])
        assert "src/app.py::foo [1-2]" in ctx

    def test_format_retrieval_context_without_symbol(self):
        """Test formatting without symbol information."""
        ctx = format_retrieval_context([
            RetrievedChunk(
                repo="repo",
                branch="main",
                path="src/app.py",
                symbol=None,
                score=0.9,
                chunk_text="print('hello')",
                line_start=1,
                line_end=2,
                language="python",
            )
        ])
        assert "src/app.py" in ctx


@pytest.mark.retrieval
class TestRetrievalContextInjection:
    """Tests for _inject_retrieval_context() message manipulation."""

    def test_inject_retrieval_context_prepends_system_message(self):
        """Test that retrieval context is prepended as system message."""
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

        assert injected[0]["role"] == "system"
        assert "Retrieved context:" in injected[0]["content"]
        assert injected[1] == messages[0]

    def test_inject_retrieval_context_preserves_user_message(self):
        """Test that user message is preserved after retrieval context."""
        body = {"retrieval": {"repo": "repo", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        async def _fake_retrieve_context(**kwargs):
            return []

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context", new=_fake_retrieve_context),
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        assert len(injected) == 1
        assert injected[0] == messages[0]


@pytest.mark.retrieval
class TestRetrievalQdrantIntegration:
    """Tests for Qdrant client operations (enabled/disabled)."""

    def test_qdrant_disabled_passthrough(self):
        """Verify that when ENABLE_QDRANT_RETRIEVAL=false, messages are returned unchanged."""
        body = {"retrieval": {"repo": "repo", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", False),
            mock.patch.object(app, "retrieve_context") as mock_retrieve,
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        assert injected == messages
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

        assert injected == messages

    def test_retrieval_empty_repo_returns_unchanged(self):
        """Verify that when repo is empty, messages are returned unchanged."""
        body = {"retrieval": {"repo": "", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context") as mock_retrieve,
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        assert injected == messages
        mock_retrieve.assert_not_called()


@pytest.mark.retrieval
class TestRetrievalRepoChunks:
    """Tests for collect_repo_chunks() function."""

    def test_collect_repo_chunks_extracts_python_functions(self):
        """Test that Python functions are extracted from repo chunks."""
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
        assert "alpha" in symbols
        assert "Beta" in symbols

    def test_collect_repo_chunks_extracts_classes(self):
        """Test that Python classes are extracted from repo chunks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir)
            (repo_dir / "demo.py").write_text(
                """
class Alpha:
    pass

class Beta:
    class Gamma:
        pass
""".strip(),
                encoding="utf-8",
            )
            chunks = collect_repo_chunks(repo_dir)

        symbols = {chunk.symbol for chunk in chunks if chunk.path.endswith("demo.py")}
        assert "Alpha" in symbols
        assert "Beta" in symbols

    def test_collect_repo_chunks_empty_directory(self):
        """Test that empty directory returns no chunks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_dir = Path(tmp_dir)
            chunks = collect_repo_chunks(repo_dir)

        assert len(chunks) == 0


@pytest.mark.retrieval
class TestRetrievalEdgeCases:
    """Tests for edge cases: empty repos, unavailable Qdrant, missing embeddings."""

    def test_retrieval_context_with_no_chunks(self):
        """Test retrieval context with no chunks returns original messages."""
        body = {"retrieval": {"repo": "repo", "query": "find the thing"}}
        messages = [{"role": "user", "content": "question"}]

        async def _fake_retrieve_context(**kwargs):
            return []

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context", new=_fake_retrieve_context),
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        assert injected == messages

    def test_retrieval_context_with_none_messages(self):
        """Test retrieval context with None messages."""
        body = {"retrieval": {"repo": "repo", "query": "find the thing"}}

        async def _fake_retrieve_context(**kwargs):
            return []

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context", new=_fake_retrieve_context),
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, None))

        # When messages is None, _inject_retrieval_context returns None
        assert injected is None

    def test_retrieval_empty_query(self):
        """Test retrieval with empty query."""
        body = {"retrieval": {"repo": "repo", "query": ""}}
        messages = [{"role": "user", "content": "question"}]

        async def _fake_retrieve_context(**kwargs):
            return []

        with (
            mock.patch.object(app, "ENABLE_QDRANT_RETRIEVAL", True),
            mock.patch.object(app, "retrieve_context", new=_fake_retrieve_context),
        ):
            injected = asyncio.run(app._inject_retrieval_context(body, messages))

        assert injected == messages
