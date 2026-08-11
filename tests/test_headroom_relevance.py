"""Tests for relevance scoring ([relevance] extra with fastembed)."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

import router_headroom as headroom

from conftest import CODE_POLICY, MESSAGES, BASE_POLICY, MODEL


@pytest.mark.relevance
@pytest.mark.compression
class TestRelevanceScoringConfig:
    """Tests for relevance scoring configuration."""

    @pytest.mark.compression
    def test_relevance_enabled_passes_headroom_query(self, patch_headroom_compress, patch_token_count):
        """Test headroom_query is passed when HEADROOM_RELEVANCE_ENABLED defaults to 1."""
        compressed_messages = [{"role": "user", "content": "relevant"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=8,
            transforms_applied=["relevance_filter"],
        )

        messages_with_query = [{"role": "user", "content": "What is AI?"}]

        with (
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[10, 8]),
        ):
            result = headroom.check_and_trim(messages_with_query, MODEL, BASE_POLICY)

        # Verify headroom_query is passed (extracted from user message)
        call_args = call.call_args
        assert "headroom_query" in call_args.kwargs
        assert call_args.kwargs["headroom_query"] == "What is AI?"

    @pytest.mark.compression
    def test_relevance_disabled_excludes_headroom_query(self, patch_headroom_compress, patch_token_count):
        """Test headroom_query is NOT passed when HEADROOM_RELEVANCE_ENABLED=0."""
        compressed_messages = [{"role": "user", "content": "all"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=10,
            transforms_applied=[],
        )

        messages_with_query = [{"role": "user", "content": "What is AI?"}]

        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "0"}),
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[10, 10]),
        ):
            result = headroom.check_and_trim(messages_with_query, MODEL, BASE_POLICY)

        # Verify headroom_query is NOT in the call
        call_args = call.call_args
        assert "headroom_query" not in call_args.kwargs

    @pytest.mark.compression
    def test_relevance_with_no_extractable_query(self, patch_headroom_compress, patch_token_count):
        """Test compression works when no query can be extracted (e.g., only assistant messages)."""
        compressed_messages = [{"role": "assistant", "content": "response"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=10,
            transforms_applied=[],
        )

        messages_no_user = [{"role": "assistant", "content": "response"}]

        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[10, 10]),
        ):
            result = headroom.check_and_trim(messages_no_user, MODEL, BASE_POLICY)

        # Verify headroom_query is NOT passed when no query can be extracted
        call_args = call.call_args
        assert "headroom_query" not in call_args.kwargs or call_args.kwargs["headroom_query"] is None


@pytest.mark.relevance
@pytest.mark.error_handling
class TestRelevanceScoringEdgeCases:
    """Tests for relevance scoring edge cases."""

    @pytest.mark.error_handling
    def test_relevance_with_empty_transforms(self, patch_headroom_compress, patch_token_count):
        """Test relevance scoring with empty transforms list."""
        # Use same object reference to ensure trimmed=False
        compressed = SimpleNamespace(
            messages=MESSAGES,
            tokens_before=10,
            tokens_after=10,
            transforms_applied=[],
        )

        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 10]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.trimmed is False
        assert result.transforms_applied == []

    @pytest.mark.error_handling
    def test_relevance_with_dict_result(self, patch_headroom_compress_dict, patch_token_count):
        """Test relevance scoring with dict result."""
        compressed_messages = [{"role": "user", "content": "dict"}]
        compressed = {
            "messages": compressed_messages,
            "tokens_before": 100,
            "tokens_after": 50,
            "transforms_applied": ["relevance_filter"],
        }

        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            patch_headroom_compress_dict(compressed),
            patch_token_count(side_effect=[100, 50]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_before == 100
        assert result.tokens_after == 50


@pytest.mark.relevance
@pytest.mark.error_handling
class TestRelevanceScoringErrorHandling:
    """Tests for relevance scoring error handling."""

    @pytest.mark.error_handling
    def test_relevance_missing_dependency_fails_explicitly(self, patch_token_count):
        """Test RuntimeError when headroom.compress is unavailable."""
        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            mock.patch.object(headroom, "_headroom_compress", None),
            patch_token_count(return_value=10),
            pytest.raises(RuntimeError, match="compression API is unavailable"),
        ):
            headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

    @pytest.mark.error_handling
    def test_relevance_type_error_fallback_with_minimal_params(self, patch_headroom_compress, patch_token_count):
        """Test TypeError fallback retries with minimal params (messages and model only)."""
        compressed_messages = [{"role": "user", "content": "fallback"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=5,
            transforms_applied=["relevance_filter"],
        )

        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            mock.patch.object(headroom, "_headroom_compress") as mock_compress,
            patch_token_count(side_effect=[10, 5]),
        ):
            # First call raises TypeError, second succeeds
            mock_compress.side_effect = [TypeError("old API"), compressed]
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_after == 5
        # Verify fallback was called with only messages and model
        assert mock_compress.call_count == 2
        fallback_call = mock_compress.call_args_list[1]
        assert "messages" in fallback_call.kwargs or len(fallback_call.args) > 0
        assert fallback_call.kwargs.get("model") == MODEL

    @pytest.mark.error_handling
    def test_relevance_general_exception_raises_runtime_error(self, patch_headroom_compress, patch_token_count):
        """Test general exceptions raise RuntimeError with message."""
        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            mock.patch.object(headroom, "_headroom_compress", side_effect=ValueError("network error")),
            patch_token_count(return_value=10),
            pytest.raises(RuntimeError, match="Headroom compression failed"),
        ):
            headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)


@pytest.mark.relevance
@pytest.mark.compression
class TestRelevanceScoringBudget:
    """Tests for relevance scoring budget enforcement."""

    @pytest.mark.compression
    def test_relevance_rejects_when_exceeds_budget(self, patch_headroom_compress, patch_token_count):
        """Test rejection when compressed prompt exceeds budget."""
        compressed_messages = [{"role": "user", "content": "still too long"}]
        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[900, 850]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is True
        assert result.prompt_tokens == 850
        assert result.usable_prompt_budget == 800

    @pytest.mark.compression
    def test_relevance_accepts_when_within_budget(self, patch_headroom_compress, patch_token_count):
        """Test acceptance when compressed prompt is within budget."""
        compressed_messages = [{"role": "user", "content": "short"}]
        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[900, 700]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.prompt_tokens == 700
        assert result.usable_prompt_budget == 800


@pytest.mark.relevance
@pytest.mark.integration
class TestRelevanceScoringIntegration:
    """Tests for relevance scoring integration scenarios."""

    @pytest.mark.integration
    def test_relevance_full_pipeline(self, patch_headroom_compress, patch_token_count):
        """Test full relevance scoring pipeline end-to-end."""
        compressed_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
        ]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=150,
            tokens_after=80,
            transforms_applied=["relevance_filter"],
        )

        with (
            mock.patch.dict(os.environ, {"HEADROOM_RELEVANCE_ENABLED": "1"}),
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[150, 80]),
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
        assert "relevance_filter" in result.transforms_applied
        # Verify headroom_query was passed
        call_args = call.call_args
        assert call_args.kwargs.get("headroom_query") == "What is AI?"


@pytest.mark.relevance
@pytest.mark.compression
class TestNonStandardContent:
    """Tests for handling non-string content in messages."""

    def test_compress_with_dict_content(self, patch_headroom_compress, patch_token_count):
        """Test compression when a message contains dict content (e.g., file read)."""
        # Simulate the input that caused the failure
        messages = [
            {"role": "user", "content": "Check this file"},
            {"role": "assistant", "content": {"path": "Dockerfile", "mode": "slice", "offset": 1, "limit": 100}},
        ]

        compressed_messages = [{"role": "user", "content": "Compressed content"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=20,
            tokens_after=10,
            transforms_applied=[],
        )

        with (
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[20, 10]),
        ):
            # This should extract the query from the user message and handle it gracefully
            result = headroom.check_and_trim(messages, MODEL, CODE_POLICY)

        # Verify that headroom_query was extracted from the user message
        call_args = call.call_args
        assert call_args.kwargs.get("headroom_query") == "Check this file"
        assert result.rejected is False

    def test_compress_with_list_content(self, patch_headroom_compress, patch_token_count):
        """Test compression when a message contains list content."""
        messages = [
            {"role": "user", "content": "List content"},
            {"role": "assistant", "content": ["item1", "item2", "item3"]},
        ]

        compressed_messages = [{"role": "user", "content": "Compressed"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=15,
            tokens_after=8,
            transforms_applied=[],
        )

        with (
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[15, 8]),
        ):
            result = headroom.check_and_trim(messages, MODEL, CODE_POLICY)

        # Verify that headroom_query was extracted from the user message
        call_args = call.call_args
        assert call_args.kwargs.get("headroom_query") == "List content"
        assert result.rejected is False

    def test_compress_with_multimodal_content(self, patch_headroom_compress, patch_token_count):
        """Test compression when a message contains multimodal content (list of dicts)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]

        compressed_messages = [{"role": "user", "content": "Compressed"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=15,
            tokens_after=8,
            transforms_applied=[],
        )

        with (
            patch_headroom_compress(compressed) as call,
            patch_token_count(side_effect=[15, 8]),
        ):
            result = headroom.check_and_trim(messages, MODEL, CODE_POLICY)

        # Verify that headroom_query was extracted from the multimodal content
        call_args = call.call_args
        assert call_args.kwargs.get("headroom_query") == "What is in this image?"
        assert result.rejected is False
