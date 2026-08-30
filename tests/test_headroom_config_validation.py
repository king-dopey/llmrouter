"""Tests for _HEADROOM_CONFIG singleton and _validate_tool_integrity function."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

import router_headroom as headroom


@pytest.mark.base
class TestHeadroomConfig:
    """Tests for _HEADROOM_CONFIG module-level singleton."""

    def test_headroom_config_exists_at_module_level(self):
        """Test that _HEADROOM_CONFIG is defined at module level."""
        assert hasattr(headroom, "_HEADROOM_CONFIG")
        assert headroom._HEADROOM_CONFIG is not None

    def test_headroom_config_has_compress_system_messages_false(self):
        """Test that _HEADROOM_CONFIG has compress_system_messages=False."""
        config = headroom._HEADROOM_CONFIG
        assert hasattr(config, "compress_system_messages")
        assert config.compress_system_messages is False

    def test_headroom_config_has_compress_user_messages_false(self):
        """Test that _HEADROOM_CONFIG has compress_user_messages=False."""
        config = headroom._HEADROOM_CONFIG
        assert hasattr(config, "compress_user_messages")
        assert config.compress_user_messages is False

    def test_headroom_config_has_protect_recent(self):
        """Test that _HEADROOM_CONFIG has protect_recent=4."""
        config = headroom._HEADROOM_CONFIG
        assert hasattr(config, "protect_recent")
        assert config.protect_recent == 4


@pytest.mark.base
class TestValidateToolIntegrity:
    """Tests for _validate_tool_integrity function."""

    def test_validate_tool_integrity_returns_true_for_identical_messages(self):
        """Test that identical messages pass validation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert headroom._validate_tool_integrity(messages, messages) is True

    def test_validate_tool_integrity_returns_false_for_length_mismatch(self):
        """Test that different message lengths fail validation."""
        original = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        compressed = [
            {"role": "user", "content": "Hello"},
        ]
        assert headroom._validate_tool_integrity(original, compressed) is False

    def test_validate_tool_integrity_returns_false_when_assistant_has_tool_calls_in_original_but_not_compressed(self):
        """Test that missing tool_calls in compressed message fails validation."""
        original = [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_123"}],
            },
        ]
        compressed = [
            {"role": "user", "content": "Search"},
            {"role": "assistant", "content": None},  # tool_calls removed!
        ]
        assert headroom._validate_tool_integrity(original, compressed) is False

    def test_validate_tool_integrity_returns_false_when_compressed_has_tool_calls_but_original_does_not(self):
        """Test that unexpected tool_calls in compressed message fails validation."""
        original = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},  # No tool_calls
        ]
        compressed = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "tool_calls": [{"id": "call_123"}],  # Unexpected tool_calls!
            },
        ]
        assert headroom._validate_tool_integrity(original, compressed) is False

    def test_validate_tool_integrity_returns_false_when_system_message_content_altered(self):
        """Test that altered system message content fails validation."""
        original = [
            {"role": "system", "content": "You are a helpful assistant. Use tools."},
            {"role": "user", "content": "Hello"},
        ]
        compressed = [
            {"role": "system", "content": "You are helpful."},  # Content altered!
            {"role": "user", "content": "Hello"},
        ]
        assert headroom._validate_tool_integrity(original, compressed) is False

    def test_validate_tool_integrity_returns_false_when_tool_message_lost_tool_call_id(self):
        """Test that tool message losing tool_call_id fails validation."""
        original = [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123"}],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]
        compressed = [
            {"role": "user", "content": "Search"},
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_123"}],
            },
            {"role": "tool", "content": "result"},  # tool_call_id missing!
        ]
        assert headroom._validate_tool_integrity(original, compressed) is False

    def test_validate_tool_integrity_passes_valid_multi_turn_with_tool_calls(self):
        """Test that valid multi-turn conversation with tool calls passes."""
        original = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Search for Python"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "Python"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "results"},
        ]
        # Same messages (simulating no compression)
        assert headroom._validate_tool_integrity(original, original) is True

    def test_validate_tool_integrity_passes_when_no_tool_calls_anywhere(self):
        """Test that messages without any tool calls pass validation."""
        original = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        compressed = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        assert headroom._validate_tool_integrity(original, compressed) is True


@pytest.mark.base
class TestIntegrityFallbackPath:
    """Tests for integrity fallback path in _compress()."""

    def test_compress_falls_back_to_original_messages_on_system_message_alteration(self):
        """Test that _compress returns original messages when system message content changes."""
        original_messages = [
            {"role": "system", "content": "You are a helpful assistant. Use tools."},
            {"role": "user", "content": "Hello"},
        ]

        # Simulate compression that alters system message (will fail validation)
        altered_compressed = [
            {"role": "system", "content": "You are helpful."},  # Content changed!
            {"role": "user", "content": "Hello"},
        ]
        result = SimpleNamespace(
            messages=altered_compressed,
            tokens_before=50,
            tokens_after=30,
            transforms_applied=["content_router"],
        )

        with (
            mock.patch.object(headroom, "_headroom_compress", return_value=result),
            mock.patch.object(headroom, "_token_count", side_effect=[50, 50]),
        ):
            output_messages, tokens_before, tokens_after, transforms = headroom._compress(
                original_messages, "test-model", {"options": {}, "reserved_output_tokens": 2048, "safety_headroom_tokens": 2048}
            )

        # Should return original messages (not compressed)
        assert output_messages == original_messages
        # Tokens should be original count
        assert tokens_before == 50
        assert tokens_after == 50
        # Transform should indicate fallback
        assert "integrity_guard:fallback" in transforms

    def test_compress_returns_compressed_messages_on_validation_success(self):
        """Test that _compress returns compressed messages when validation passes."""
        original_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Simulate compression that preserves structure (passes validation)
        compressed_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},  # Compressed but valid
        ]
        result = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=50,
            tokens_after=30,
            transforms_applied=["content_router"],
        )

        with (
            mock.patch.object(headroom, "_headroom_compress", return_value=result),
            mock.patch.object(headroom, "_token_count", side_effect=[50, 30]),
        ):
            output_messages, tokens_before, tokens_after, transforms = headroom._compress(
                original_messages, "test-model", {"options": {}, "reserved_output_tokens": 2048, "safety_headroom_tokens": 2048}
            )

        # Should return compressed messages
        assert output_messages == compressed_messages
        assert tokens_before == 50
        assert tokens_after == 30
        # Transform should NOT indicate fallback
        assert "integrity_guard:fallback" not in transforms

    def test_compress_logs_warning_on_validation_failure(self):
        """Test that _compress logs a warning when falling back."""
        original_messages = [
            {"role": "system", "content": "You are a helpful assistant. Use tools."},
            {"role": "user", "content": "Hello"},
        ]

        altered_compressed = [
            {"role": "system", "content": "You are helpful."},  # Content changed!
            {"role": "user", "content": "Hello"},
        ]
        result = SimpleNamespace(
            messages=altered_compressed,
            tokens_before=50,
            tokens_after=30,
            transforms_applied=["content_router"],
        )

        with (
            mock.patch.object(headroom, "_headroom_compress", return_value=result),
            mock.patch.object(headroom, "_token_count", side_effect=[50, 50]),
            mock.patch("router_headroom.logger") as logger_mock,
        ):
            headroom._compress(
                original_messages, "test-model", {"options": {}, "reserved_output_tokens": 2048, "safety_headroom_tokens": 2048}
            )

        # Two warnings fire: system message altered + integrity fallback
        assert logger_mock.warning.call_count == 2
        call_strings = [str(c) for c in logger_mock.warning.call_args_list]
        assert any("system message altered" in c for c in call_strings)
        assert any("integrity check FAILED" in c for c in call_strings)

    def test_compress_falls_back_on_message_length_mismatch(self):
        """Test that _compress falls back when compressed message count differs."""
        original_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        # Simulate compression that changes message count (will fail validation)
        altered_compressed = [
            {"role": "user", "content": "Hello"},  # assistant removed!
        ]
        result = SimpleNamespace(
            messages=altered_compressed,
            tokens_before=50,
            tokens_after=20,
            transforms_applied=["content_router"],
        )

        with (
            mock.patch.object(headroom, "_headroom_compress", return_value=result),
            mock.patch.object(headroom, "_token_count", side_effect=[50, 50]),
        ):
            output_messages, tokens_before, tokens_after, transforms = headroom._compress(
                original_messages, "test-model", {"options": {}, "reserved_output_tokens": 2048, "safety_headroom_tokens": 2048}
            )

        # Should return original messages (not compressed)
        assert output_messages == original_messages
        assert tokens_before == 50
        assert tokens_after == 50
        assert "integrity_guard:fallback" in transforms
