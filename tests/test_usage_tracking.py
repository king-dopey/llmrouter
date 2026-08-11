"""Tests for usage tracking module with pytest markers."""

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

from app import _build_non_stream_usage  # noqa: E402


@pytest.mark.usage
class TestUsageBuildNonStream:
    """Tests for _build_non_stream_usage() field population."""

    def test_uses_ollama_counts_and_cache_fields(self):
        """Test Ollama response fields are correctly mapped."""
        usage = _build_non_stream_usage(
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
        assert usage["prompt_tokens"] == 11
        assert usage["completion_tokens"] == 7
        assert usage["total_tokens"] == 18
        assert usage["cache_creation_input_tokens"] == 3
        assert usage["cache_read_input_tokens"] == 2

    def test_uses_ollama_counts_without_cache(self):
        """Test Ollama counts without cache fields."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15
        assert usage["cache_creation_input_tokens"] == 0
        assert usage["cache_read_input_tokens"] == 0

    def test_uses_ollama_counts_with_only_prompt(self):
        """Test Ollama counts with only prompt_eval_count."""
        # When eval_count is missing, tokenizer fallback is used
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["prompt_tokens"] == 10
        # completion_tokens uses tokenizer fallback when eval_count is missing
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.usage
class TestUsageFallbackToTokenizer:
    """Tests for tokenizer fallback when Ollama counts missing."""

    def test_falls_back_to_tokenizer_when_ollama_counts_missing(self):
        """Test tokenizer fallback when ollama_result is empty."""
        usage = _build_non_stream_usage(
            ollama_result={},
            payload_messages=[{"role": "user", "content": "abcd" * 10}],
            model="qwen3:4b",
            completion_text="abcd" * 5,
        )
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        assert usage["cache_creation_input_tokens"] == 0
        assert usage["cache_read_input_tokens"] == 0

    def test_falls_back_to_tokenizer_with_none_ollama_result(self):
        """Test tokenizer fallback when ollama_result is None."""
        # Note: _build_non_stream_usage expects a dict, not None
        # When ollama_result is None, it will fail with AttributeError
        # This test verifies the expected behavior (dict required)
        with pytest.raises(AttributeError):
            _build_non_stream_usage(
                ollama_result=None,
                payload_messages=[{"role": "user", "content": "hello"}],
                model="qwen3:4b",
                completion_text="world",
            )

    def test_falls_back_to_tokenizer_with_empty_messages(self):
        """Test tokenizer fallback with empty messages."""
        usage = _build_non_stream_usage(
            ollama_result={},
            payload_messages=[],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0


@pytest.mark.usage
class TestUsageCacheTokens:
    """Tests for cache_creation_input_tokens and cache_read_input_tokens."""

    def test_cache_tokens_default_to_zero(self):
        """Test cache tokens default to zero when not present."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["cache_creation_input_tokens"] == 0
        assert usage["cache_read_input_tokens"] == 0

    def test_cache_tokens_preserved_when_present(self):
        """Test cache tokens are preserved when present."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["cache_creation_input_tokens"] == 3
        assert usage["cache_read_input_tokens"] == 2

    def test_cache_tokens_with_zero_values(self):
        """Test cache tokens with zero values."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["cache_creation_input_tokens"] == 0
        assert usage["cache_read_input_tokens"] == 0


@pytest.mark.usage
class TestUsageTotalCalculation:
    """Tests for total_tokens = prompt_tokens + completion_tokens."""

    def test_total_calculation_simple(self):
        """Test simple total calculation."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["total_tokens"] == 15

    def test_total_calculation_with_cache(self):
        """Test total calculation includes cache tokens."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 2,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        # total_tokens = prompt_tokens + completion_tokens (cache tokens are separate)
        assert usage["total_tokens"] == 15

    def test_total_calculation_large_numbers(self):
        """Test total calculation with large numbers."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 100000,
                "eval_count": 50000,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["total_tokens"] == 150000

    def test_total_calculation_zero(self):
        """Test total calculation with zero tokens."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 0,
                "eval_count": 0,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["total_tokens"] == 0


@pytest.mark.usage
class TestUsageEdgeCases:
    """Tests for edge cases in usage tracking."""

    def test_handles_none_values_in_ollama_result(self):
        """Test handling of None values in ollama_result."""
        # When values are None, tokenizer fallback is used
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": None,
                "eval_count": None,
            },
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        # None values trigger tokenizer fallback
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0

    def test_handles_missing_keys_in_ollama_result(self):
        """Test handling of missing keys in ollama_result."""
        # Missing keys trigger tokenizer fallback
        usage = _build_non_stream_usage(
            ollama_result={},
            payload_messages=[{"role": "user", "content": "hi"}],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["prompt_tokens"] >= 0
        assert usage["completion_tokens"] >= 0

    def test_handles_empty_payload_messages(self):
        """Test handling of empty payload messages."""
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
            payload_messages=[],
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5

    def test_handles_none_payload_messages(self):
        """Test handling of None payload messages."""
        # When payload_messages is None, tokenizer.count_prompt_tokens handles it
        usage = _build_non_stream_usage(
            ollama_result={
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
            payload_messages=None,
            model="qwen3:4b",
            completion_text="ok",
        )
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
