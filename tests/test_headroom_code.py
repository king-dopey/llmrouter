"""Tests for code compression ([code] extra with tree-sitter)."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

import router_headroom as headroom

from conftest import CODE_POLICY, MESSAGES, BASE_POLICY, MODEL


@pytest.mark.code
@pytest.mark.compression
class TestCodeCompressionConfig:
    """Tests for code compression configuration."""

    @pytest.mark.compression
    def test_code_compression_uses_valid_api_params(self, patch_headroom_compress, patch_token_count):
        """Test code compression uses only valid API parameters (model, headroom_query)."""
        compressed_messages = [{"role": "user", "content": "code"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=100,
            tokens_after=50,
            transforms_applied=["code_compressor"],
        )

        with (
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[100, 50]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

        # Verify only valid parameters are passed (model is always passed)
        call_args = call.call_args
        assert call_args.kwargs.get("model") == MODEL
        # Invalid parameters should NOT be passed
        assert "compress_user_messages" not in call_args.kwargs
        assert "target_ratio" not in call_args.kwargs
        assert "protect_recent" not in call_args.kwargs

    @pytest.mark.compression
    def test_code_compression_reduces_tokens(self, patch_headroom_compress, patch_token_count):
        """Test code compression actually reduces token count."""
        compressed_messages = [{"role": "user", "content": "reduced"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=200,
            tokens_after=80,
            transforms_applied=["code_compressor"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[200, 80]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

        assert result.tokens_before == 200
        assert result.tokens_after == 80
        assert "code_compressor" in result.transforms_applied


@pytest.mark.code
@pytest.mark.error_handling
class TestCodeCompressionEdgeCases:
    """Tests for code compression edge cases."""

    @pytest.mark.error_handling
    def test_code_compression_with_empty_transforms(self, patch_headroom_compress, patch_token_count):
        """Test code compression with empty transforms list."""
        # Use same object reference to ensure trimmed=False
        compressed = SimpleNamespace(
            messages=MESSAGES,
            tokens_before=10,
            tokens_after=10,
            transforms_applied=[],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 10]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

        assert result.trimmed is False
        assert result.transforms_applied == []

    @pytest.mark.error_handling
    def test_code_compression_with_dict_result(self, patch_headroom_compress_dict, patch_token_count):
        """Test code compression with dict result."""
        compressed_messages = [{"role": "user", "content": "dict"}]
        compressed = {
            "messages": compressed_messages,
            "tokens_before": 100,
            "tokens_after": 50,
            "transforms_applied": ["code_compressor"],
        }

        with (
            patch_headroom_compress_dict(compressed),
            patch_token_count(side_effect=[100, 50]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_before == 100
        assert result.tokens_after == 50


@pytest.mark.code
@pytest.mark.error_handling
class TestCodeCompressionErrorHandling:
    """Tests for code compression error handling."""

    @pytest.mark.error_handling
    def test_code_compression_missing_dependency_fails_explicitly(self, patch_token_count):
        """Test RuntimeError when headroom.compress is unavailable."""
        with (
            mock.patch.object(headroom, "_headroom_compress", None),
            patch_token_count(return_value=10),
            mock.patch.object(headroom, "_extract_query", return_value=None),
            pytest.raises(RuntimeError, match="compression API is unavailable"),
        ):
            headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

    @pytest.mark.error_handling
    def test_code_compression_type_error_fallback_with_minimal_params(self, patch_headroom_compress, patch_token_count):
        """Test TypeError fallback retries with minimal params (messages and model only)."""
        compressed_messages = [{"role": "user", "content": "fallback"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=5,
            transforms_applied=["content_router"],
        )

        with (
            mock.patch.object(headroom, "_headroom_compress") as mock_compress,
            patch_token_count(side_effect=[10, 5]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            # First call raises TypeError, second succeeds
            mock_compress.side_effect = [TypeError("old API"), compressed]
            result = headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_after == 5
        # Verify fallback was called with only messages and model
        assert mock_compress.call_count == 2
        fallback_call = mock_compress.call_args_list[1]
        assert fallback_call.kwargs.get("model") == MODEL

    @pytest.mark.error_handling
    def test_code_compression_general_exception_raises_runtime_error(self, patch_headroom_compress, patch_token_count):
        """Test general exceptions raise RuntimeError with message."""
        with (
            mock.patch.object(headroom, "_headroom_compress", side_effect=ValueError("network error")),
            patch_token_count(return_value=10),
            mock.patch.object(headroom, "_extract_query", return_value=None),
            pytest.raises(RuntimeError, match="Headroom compression failed"),
        ):
            headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

    @pytest.mark.error_handling
    def test_code_compression_fallback_exception_raises_runtime_error(self, patch_headroom_compress, patch_token_count):
        """Test fallback exception raises RuntimeError with message."""
        with (
            mock.patch.object(headroom, "_headroom_compress") as mock_compress,
            patch_token_count(return_value=10),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            mock_compress.side_effect = [TypeError("old API"), ValueError("fallback error")]

            with pytest.raises(RuntimeError, match="Headroom compression failed"):
                headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)


@pytest.mark.code
@pytest.mark.compression
class TestCodeCompressionBudget:
    """Tests for code compression budget enforcement."""

    @pytest.mark.compression
    def test_code_compression_rejects_when_exceeds_budget(self, patch_headroom_compress, patch_token_count):
        """Test rejection when compressed prompt exceeds budget."""
        compressed_messages = [{"role": "user", "content": "still too long"}]
        with (
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[900, 850]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

        assert result.rejected is True
        assert result.prompt_tokens == 850
        assert result.usable_prompt_budget == 800

    @pytest.mark.compression
    def test_code_compression_accepts_when_within_budget(self, patch_headroom_compress, patch_token_count):
        """Test acceptance when compressed prompt is within budget."""
        compressed_messages = [{"role": "user", "content": "short"}]
        with (
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[900, 700]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, CODE_POLICY)

        assert result.rejected is False
        assert result.prompt_tokens == 700
        assert result.usable_prompt_budget == 800


@pytest.mark.code
@pytest.mark.integration
class TestCodeCompressionIntegration:
    """Tests for code compression integration scenarios."""

    @pytest.mark.integration
    def test_code_compression_full_pipeline(self, patch_headroom_compress, patch_token_count):
        """Test full code compression pipeline end-to-end."""
        compressed_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
        ]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=150,
            tokens_after=80,
            transforms_applied=["code_compressor"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[150, 80]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is AI?"},
                    {"role": "assistant", "content": "AI is a broad field of computer science..."},
                ],
                MODEL,
                CODE_POLICY,
            )

        assert result.rejected is False
        assert result.trimmed is True
        assert len(result.messages) == 3
        assert result.tokens_before == 150
        assert result.tokens_after == 80
        assert "code_compressor" in result.transforms_applied
