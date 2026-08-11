"""Tests for base compression (headroom-ai without extras)."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

import router_headroom as headroom

from conftest import CODE_POLICY, MESSAGES, BASE_POLICY, MODEL


@pytest.mark.base
class TestExtractQuery:
    """Tests for _extract_query helper function."""

    def test_extract_query_from_single_user_message(self):
        """Test extracting query from a single user message."""
        messages = [{"role": "user", "content": "What is AI?"}]
        query = headroom._extract_query(messages)
        assert query == "What is AI?"

    def test_extract_query_from_multi_turn_conversation(self):
        """Test extracting the last user query from multi-turn conversation."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What is AI?"},
        ]
        query = headroom._extract_query(messages)
        assert query == "What is AI?"

    def test_extract_query_from_multimodal_content(self):
        """Test extracting query from multimodal content (list of dicts)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        query = headroom._extract_query(messages)
        assert query == "What is in this image?"

    def test_extract_query_returns_none_for_empty_messages(self):
        """Test that _extract_query returns None for empty messages."""
        query = headroom._extract_query([])
        assert query is None

    def test_extract_query_returns_none_for_no_user_messages(self):
        """Test that _extract_query returns None when no user messages exist."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": "Hi there!"},
        ]
        query = headroom._extract_query(messages)
        assert query is None

    def test_extract_query_skips_non_string_content(self):
        """Test that _extract_query skips messages with non-string/list content."""
        messages = [
            {"role": "user", "content": {"path": "file.py"}},  # dict content
            {"role": "user", "content": "What is AI?"},  # valid string content
        ]
        query = headroom._extract_query(messages)
        assert query == "What is AI?"

    def test_extract_query_handles_empty_text_parts(self):
        """Test that _extract_query handles multimodal content with empty text parts."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        query = headroom._extract_query(messages)
        assert query is None  # No text parts found


@pytest.mark.base
@pytest.mark.compression
class TestBaseCompression:
    """Tests for base compression functionality."""

    @pytest.mark.compression
    def test_compression_runs_once_for_small_prompt(self, patch_headroom_compress, patch_token_count):
        """Test base compression with CompressResult object."""
        compressed_messages = [{"role": "user", "content": "hi"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=3,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 3]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.messages == compressed_messages
        assert result.tokens_before == 10
        assert result.tokens_after == 3
        assert result.transforms_applied == ["content_router"]

    @pytest.mark.compression
    def test_plain_message_list_is_supported(self, patch_headroom_compress_list, patch_token_count):
        """Test base compression with plain list result."""
        compressed_messages = [{"role": "user", "content": "short"}]

        with (
            patch_headroom_compress_list(compressed_messages),
            patch_token_count(side_effect=[10, 2]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_before == 10
        assert result.tokens_after == 2

    def test_dict_result_is_supported(self, patch_headroom_compress_dict, patch_token_count):
        """Test base compression with dict result."""
        compressed_messages = [{"role": "user", "content": "dict"}]
        compressed = {
            "messages": compressed_messages,
            "tokens_before": 10,
            "tokens_after": 3,
            "transforms_applied": ["code_compressor"],
        }

        with (
            patch_headroom_compress_dict(compressed),
            patch_token_count(side_effect=[10, 3]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_before == 10
        assert result.tokens_after == 3

    def test_empty_transforms_preserves_original(self, patch_headroom_compress, patch_token_count):
        """Test compression with empty transforms list."""
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
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.trimmed is False
        assert result.transforms_applied == []

    def test_compression_reduces_tokens(self, patch_headroom_compress, patch_token_count):
        """Test compression actually reduces token count."""
        compressed_messages = [{"role": "user", "content": "reduced"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=200,
            tokens_after=80,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[200, 80]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.tokens_before == 200
        assert result.tokens_after == 80


@pytest.mark.base
@pytest.mark.error_handling
class TestBaseCompressionEdgeCases:
    """Tests for base compression edge cases."""

    def test_empty_messages_returns_zero_tokens(self, patch_headroom_compress):
        """Test empty messages list returns zero tokens without calling compression."""
        with mock.patch.object(headroom, "_headroom_compress") as call:
            result = headroom.check_and_trim([], MODEL, BASE_POLICY)

        call.assert_not_called()
        assert result.messages == []
        assert result.prompt_tokens == 0
        assert result.tokens_before == 0
        assert result.tokens_after == 0

    def test_single_turn_conversation(self, patch_headroom_compress, patch_token_count):
        """Test single turn conversation (user message only)."""
        single_message = [{"role": "user", "content": "hello"}]
        compressed_messages = [{"role": "user", "content": "hi"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=5,
            tokens_after=2,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[5, 2]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(single_message, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_before == 5
        assert result.tokens_after == 2

    def test_multi_turn_conversation(self, patch_headroom_compress, patch_token_count):
        """Test multi-turn conversation with system, user, and assistant messages."""
        multi_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        compressed_messages = [
            {"role": "system", "content": "Helpful assistant."},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=30,
            tokens_after=15,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[30, 15]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(multi_messages, MODEL, BASE_POLICY)

        assert len(result.messages) == 3
        assert result.tokens_before == 30
        assert result.tokens_after == 15

    def test_unexpected_result_type_logs_warning(self, patch_headroom_compress, patch_token_count):
        """Test unexpected result type logs warning and returns original messages."""
        with (
            mock.patch.object(headroom, "_headroom_compress", return_value="unexpected_string"),
            mock.patch.object(headroom, "_token_count", side_effect=[10, 10]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
            mock.patch("router_headroom.logger") as logger_mock,
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        logger_mock.warning.assert_called_once()
        assert result.messages == MESSAGES  # Original messages returned
        assert result.trimmed is False


@pytest.mark.base
@pytest.mark.compression
class TestBaseCompressionBudget:
    """Tests for base compression budget enforcement."""

    def test_rejects_when_compressed_exceeds_budget(self, patch_headroom_compress, patch_token_count):
        """Test rejection when compressed prompt exceeds budget."""
        compressed_messages = [{"role": "user", "content": "still too long"}]
        with (
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[900, 850]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is True
        assert result.prompt_tokens == 850
        assert result.usable_prompt_budget == 800
        assert result.error_response["error"]["details"]["over_by_tokens"] == 50

    def test_accepts_when_compressed_within_budget(self, patch_headroom_compress, patch_token_count):
        """Test acceptance when compressed prompt is within budget."""
        compressed_messages = [{"role": "user", "content": "short"}]
        with (
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[900, 700]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.prompt_tokens == 700
        assert result.usable_prompt_budget == 800

    def test_accepts_when_compressed_equals_budget(self, patch_headroom_compress, patch_token_count):
        """Test acceptance when compressed prompt equals budget (not exceeded)."""
        compressed_messages = [{"role": "user", "content": "exact"}]
        with (
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[800, 800]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.prompt_tokens == 800
        assert result.usable_prompt_budget == 800


@pytest.mark.base
@pytest.mark.compression
class TestBaseCompressionTrimReason:
    """Tests for base compression trim reason."""

    def test_trim_reason_when_compressed(self, patch_headroom_compress, patch_token_count):
        """Test trim_reason is 'compressed' when messages are actually compressed."""
        compressed_messages = [{"role": "user", "content": "hi"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=3,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 3]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.trim_reason == "compressed"

    def test_trim_reason_none_when_not_compressed(self, patch_headroom_compress, patch_token_count):
        """Test trim_reason is None when messages are not compressed."""
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
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.trim_reason is None


@pytest.mark.base
@pytest.mark.compression
class TestBaseCompressionConfig:
    """Tests for base compression configuration."""

    def test_headroom_disabled_with_false(self, patch_headroom_compress, patch_token_count):
        """Test HEADROOM_ENABLED=false disables compression."""
        with (
            mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "false"}),
            mock.patch.object(headroom, "_headroom_compress") as call,
            patch_token_count(return_value=10),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        call.assert_not_called()
        assert result.trimmed is False
        assert result.messages == MESSAGES

    def test_headroom_disabled_with_off(self, patch_headroom_compress, patch_token_count):
        """Test HEADROOM_ENABLED=off disables compression."""
        with (
            mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "off"}),
            mock.patch.object(headroom, "_headroom_compress") as call,
            patch_token_count(return_value=10),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        call.assert_not_called()
        assert result.trimmed is False
        assert result.messages == MESSAGES

    def test_headroom_enabled_with_1(self, patch_headroom_compress, patch_token_count):
        """Test HEADROOM_ENABLED=1 enables compression."""
        compressed_messages = [{"role": "user", "content": "hi"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=10,
            tokens_after=3,
            transforms_applied=["content_router"],
        )

        with (
            mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1"}),
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 3]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.trimmed is True
        assert result.messages == compressed_messages


@pytest.mark.base
@pytest.mark.compression
class TestBaseCompressionPolicy:
    """Tests for base compression policy resolution."""

    def test_policy_resolution_from_options(self):
        """Test policy resolution extracts options correctly."""
        policy_entry = {
            "options": {"num_ctx": 2000},
            "reserved_output_tokens": 512,
            "safety_headroom_tokens": 256,
        }

        policy = headroom.resolve_headroom(MODEL, policy_entry)

        assert policy.model == MODEL
        assert policy.max_num_ctx == 2000
        assert policy.reserved_output_tokens == 512
        assert policy.safety_headroom_tokens == 256

    def test_policy_resolution_with_default_options(self):
        """Test policy resolution uses defaults when options missing."""
        policy_entry = {
            "reserved_output_tokens": 100,
            "safety_headroom_tokens": 100,
        }

        policy = headroom.resolve_headroom(MODEL, policy_entry)

        assert policy.max_num_ctx == 65536  # Default
        assert policy.reserved_output_tokens == 100
        assert policy.safety_headroom_tokens == 100

    def test_policy_resolution_with_empty_options(self):
        """Test policy resolution with empty options dict."""
        policy_entry = {
            "options": {},
            "reserved_output_tokens": 100,
            "safety_headroom_tokens": 100,
        }

        policy = headroom.resolve_headroom(MODEL, policy_entry)

        assert policy.max_num_ctx == 65536  # Default
        assert policy.reserved_output_tokens == 100
        assert policy.safety_headroom_tokens == 100

    def test_usable_prompt_budget_calculation(self):
        """Test usable prompt budget calculation."""
        policy = headroom.HeadroomPolicy(
            model=MODEL,
            max_num_ctx=10000,
            reserved_output_tokens=2000,
            safety_headroom_tokens=1000,
        )

        budget = headroom._usable_prompt_budget(policy)

        assert budget == 7000  # 10000 - 2000 - 1000

    def test_usable_prompt_budget_zero_when_overdraw(self):
        """Test usable prompt budget is zero when reserved > max."""
        policy = headroom.HeadroomPolicy(
            model=MODEL,
            max_num_ctx=1000,
            reserved_output_tokens=2000,
            safety_headroom_tokens=1000,
        )

        budget = headroom._usable_prompt_budget(policy)

        assert budget == 0  # Should not go negative


@pytest.mark.base
@pytest.mark.compression
class TestBaseCompressionTelemetry:
    """Tests for base compression telemetry extraction."""

    def test_telemetry_with_dict_result(self):
        """Test telemetry extraction from dict result."""
        compressed = {
            "tokens_before": 100,
            "tokens_after": 50,
            "transforms_applied": ["code_compressor"],
        }

        tokens_before, tokens_after, transforms = headroom._telemetry(compressed)

        assert tokens_before == 100
        assert tokens_after == 50
        assert transforms == ["code_compressor"]

    def test_telemetry_with_simple_namespace(self):
        """Test telemetry extraction from SimpleNamespace result."""
        compressed = SimpleNamespace(
            tokens_before=100,
            tokens_after=50,
            transforms_applied=["content_router"],
        )

        tokens_before, tokens_after, transforms = headroom._telemetry(compressed)

        assert tokens_before == 100
        assert tokens_after == 50
        assert transforms == ["content_router"]

    def test_telemetry_with_missing_fields(self):
        """Test telemetry extraction with missing fields defaults to 0."""
        compressed = SimpleNamespace(
            tokens_before=0,
            tokens_after=0,
            transforms_applied=[],
        )

        tokens_before, tokens_after, transforms = headroom._telemetry(compressed)

        assert tokens_before == 0
        assert tokens_after == 0
        assert transforms == []


@pytest.mark.base
@pytest.mark.compression
class TestBaseCompressionNormalizeOutput:
    """Tests for base compression _normalize_output function."""

    def test_normalize_output_with_list(self):
        """Test _normalize_output returns list directly."""
        messages = [{"role": "user", "content": "hi"}]
        result = headroom._normalize_output(messages, MESSAGES)

        assert result == messages

    def test_normalize_output_with_compress_result(self):
        """Test _normalize_output extracts messages from CompressResult."""
        compressed = SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            tokens_before=10,
            tokens_after=3,
            transforms_applied=[],
        )

        result = headroom._normalize_output(compressed, MESSAGES)

        assert result == compressed.messages

    def test_normalize_output_with_dict(self):
        """Test _normalize_output extracts messages from dict."""
        compressed = {
            "messages": [{"role": "user", "content": "hi"}],
            "tokens_before": 10,
        }

        result = headroom._normalize_output(compressed, MESSAGES)

        assert result == compressed["messages"]

    def test_normalize_output_with_unexpected_type(self):
        """Test _normalize_output logs warning for unexpected type."""
        with mock.patch("router_headroom.logger") as logger_mock:
            result = headroom._normalize_output("unexpected", MESSAGES)

        logger_mock.warning.assert_called_once()
        assert result == MESSAGES  # Returns original messages


@pytest.mark.base
@pytest.mark.integration
class TestBaseCompressionIntegration:
    """Tests for base compression integration scenarios."""

    def test_full_compression_pipeline(self, patch_headroom_compress, patch_token_count):
        """Test full compression pipeline end-to-end."""
        compressed_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."},
        ]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=150,
            tokens_after=80,
            transforms_applied=["content_router", "code_compressor"],
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
        assert "content_router" in result.transforms_applied
        assert "code_compressor" in result.transforms_applied

    def test_compression_with_error_response(self, patch_headroom_compress, patch_token_count):
        """Test error response when compression fails budget."""
        compressed_messages = [{"role": "user", "content": "still too long"}]
        with (
            patch_headroom_compress(compressed_messages),
            patch_token_count(side_effect=[900, 850]),
            mock.patch.object(headroom, "_extract_query", return_value=None),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is True
        assert result.error_response is not None
        assert result.error_response["error"]["type"] == "context_length_exceeded"
        assert "over_by_tokens" in result.error_response["error"]["details"]
