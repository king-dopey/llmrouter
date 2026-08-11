"""Tests for embeddings endpoint module with pytest markers."""

import asyncio
import os
from pathlib import Path
import sys
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


class _FakeRequest:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self):
        return self._payload


@pytest.mark.embeddings
class TestEmbeddingsSingleInput:
    """Tests for single input embedding shape and response."""

    def test_single_input_embedding_shape(self):
        """Test single input returns correct embedding shape."""
        async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
            assert model == "qwen3-embedding:4b"
            assert prompt == "hello"
            return [0.1, 0.2]

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)) as preflight,
            mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
        ):
            response = asyncio.run(
                app.embeddings(_FakeRequest({"model": "qwen3-embedding:4b", "input": "hello"}))
            )

        preflight.assert_awaited_once_with("qwen3-embedding:4b")

        assert response.status_code == 200
        assert response.content["object"] == "list"
        assert response.content["model"] == "qwen3-embedding:4b"
        assert len(response.content["data"]) == 1
        assert response.content["data"][0]["index"] == 0
        assert response.content["data"][0]["embedding"] == [0.1, 0.2]

    def test_single_input_embedding_returns_floats(self):
        """Test that embedding values are floats."""
        async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
            return [0.5, -0.3, 0.8]

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
        ):
            response = asyncio.run(
                app.embeddings(_FakeRequest({"model": "qwen3-embedding:4b", "input": "test"}))
            )

        embedding = response.content["data"][0]["embedding"]
        assert all(isinstance(v, float) for v in embedding)


@pytest.mark.embeddings
class TestEmbeddingsBatchInput:
    """Tests for batch input (list of strings) processing."""

    def test_batch_input_embedding_shape(self):
        """Test batch input returns correct shape for multiple embeddings."""
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

        assert response.status_code == 200
        assert len(response.content["data"]) == 2
        assert response.content["data"][0]["index"] == 0
        assert response.content["data"][1]["index"] == 1

    def test_batch_input_different_lengths(self):
        """Test batch input with different length strings."""
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
                            "input": ["a", "bb", "ccc"],
                        }
                    )
                )
            )

        embeddings = [d["embedding"][0] for d in response.content["data"]]
        assert embeddings == [1.0, 2.0, 3.0]


@pytest.mark.embeddings
class TestEmbeddingsDefaultModel:
    """Tests for EMBEDDING_MODEL_DEFAULT fallback."""

    def test_default_embedding_model_used_when_model_missing(self):
        """Test that default model is used when model is not specified."""
        async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
            assert model == app.EMBEDDING_MODEL_DEFAULT
            return [0.3]

        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
        ):
            response = asyncio.run(app.embeddings(_FakeRequest({"input": "hello"})))

        assert response.status_code == 200
        assert response.content["model"] == app.EMBEDDING_MODEL_DEFAULT

    def test_default_model_from_env(self):
        """Test default model from environment variable."""
        original = os.environ.get("EMBEDDING_MODEL_DEFAULT")
        try:
            os.environ["EMBEDDING_MODEL_DEFAULT"] = "custom-embedding:latest"
            # Reload app to pick up new env var
            import importlib
            importlib.reload(app)

            async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
                return [0.1]

            with (
                mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
                mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
            ):
                response = asyncio.run(app.embeddings(_FakeRequest({"input": "hello"})))

            assert response.content["model"] == "custom-embedding:latest"
        finally:
            if original is None:
                del os.environ["EMBEDDING_MODEL_DEFAULT"]
            else:
                os.environ["EMBEDDING_MODEL_DEFAULT"] = original
            importlib.reload(app)


@pytest.mark.embeddings
class TestEmbeddingsPreflight:
    """Tests for model availability checking and auto-pull."""

    def test_embedding_preflight_failure_auto_pull_disabled(self):
        """Test 404 when model unavailable and auto-pull is disabled."""
        with (
            mock.patch.object(app, "AUTO_PULL_MISSING_MODELS", False),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=False)),
        ):
            with pytest.raises(Exception) as exc_info:
                asyncio.run(
                    app.embeddings(
                        _FakeRequest({"model": "qwen3-embedding:4b", "input": "hello"})
                    )
                )
            assert exc_info.value.status_code == 404

    def test_embedding_preflight_succeeds_with_auto_pull(self):
        """Test that preflight succeeds when auto-pull is enabled."""
        async def _fake_fetch_embedding(model: str, prompt: str) -> list[float]:
            return [0.1]

        with (
            mock.patch.object(app, "AUTO_PULL_MISSING_MODELS", True),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "_fetch_embedding", new=_fake_fetch_embedding),
        ):
            response = asyncio.run(
                app.embeddings(_FakeRequest({"model": "new-model:latest", "input": "hello"}))
            )

        assert response.status_code == 200


@pytest.mark.embeddings
class TestEmbeddingsErrorHandling:
    """Tests for 404 when model unavailable, preflight failure."""

    def test_embedding_model_not_found(self):
        """Test 404 when requested model is not available."""
        with (
            mock.patch.object(app, "AUTO_PULL_MISSING_MODELS", False),
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=False)),
        ):
            with pytest.raises(Exception) as exc_info:
                asyncio.run(
                    app.embeddings(
                        _FakeRequest({"model": "unknown-model:latest", "input": "hello"})
                    )
                )
            assert exc_info.value.status_code == 404

    def test_embedding_preflight_connection_error(self):
        """Test handling of preflight connection errors."""
        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(side_effect=Exception("Connection refused"))),
        ):
            with pytest.raises(Exception) as exc_info:
                asyncio.run(
                    app.embeddings(
                        _FakeRequest({"model": "qwen3-embedding:4b", "input": "hello"})
                    )
                )
            # Should fail gracefully
            assert "Connection refused" in str(exc_info.value)

    def test_embedding_fetch_error(self):
        """Test handling of fetch embedding errors."""
        with (
            mock.patch.object(app, "_preflight_model", new=mock.AsyncMock(return_value=True)),
            mock.patch.object(app, "_fetch_embedding", new=mock.AsyncMock(side_effect=Exception("Embedding failed"))),
        ):
            with pytest.raises(Exception) as exc_info:
                asyncio.run(
                    app.embeddings(
                        _FakeRequest({"model": "qwen3-embedding:4b", "input": "hello"})
                    )
                )
            assert "Embedding failed" in str(exc_info.value)


@pytest.mark.embeddings
class TestEmbeddingsCoerceInputs:
    """Tests for input coercion functions."""

    def test_coerce_embedding_inputs_single_string(self):
        """Test coercing single string to list."""
        result = app._coerce_embedding_inputs("hello")
        assert result == ["hello"]

    def test_coerce_embedding_inputs_list(self):
        """Test passing through list."""
        result = app._coerce_embedding_inputs(["hello", "world"])
        assert result == ["hello", "world"]

    def test_coerce_embedding_inputs_empty_list(self):
        """Test empty list handling."""
        result = app._coerce_embedding_inputs([])
        assert result == []
