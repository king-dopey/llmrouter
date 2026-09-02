"""Tests for base compression (unconditional compaction + recovery trim)."""

import os
from types import SimpleNamespace
from unittest import mock

import pytest

import router_headroom as headroom

from conftest import CODE_POLICY, MESSAGES, BASE_POLICY, MODEL, HIGH_BUDGET_POLICY


@pytest.mark.base
@pytest.mark.compression
class TestUnconditionalCompaction:
    """Tests for unconditional compaction — HEADROOM's core purpose."""

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
            patch_token_count(side_effect=[10, 3, 3]),
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
            patch_token_count(side_effect=[10, 2, 2]),
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
            patch_token_count(side_effect=[10, 3, 3]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_before == 10
        assert result.tokens_after == 3

    def test_empty_transforms_preserves_original(self, patch_headroom_compress, patch_token_count):
        """Test compression with empty transforms list."""
        compressed = SimpleNamespace(
            messages=MESSAGES,
            tokens_before=10,
            tokens_after=10,
            transforms_applied=[],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 10, 10]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
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
            patch_token_count(side_effect=[200, 80, 80]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.tokens_before == 200
        assert result.tokens_after == 80
        assert result.rejected is False


@pytest.mark.base
@pytest.mark.error_handling
class TestCompactionEdgeCases:
    """Tests for compaction edge cases."""

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
            patch_token_count(side_effect=[5, 2, 2]),
        ):
            result = headroom.check_and_trim(single_message, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.tokens_before == 5

    def test_multi_turn_conversation(self, patch_headroom_compress, patch_token_count):
        """Test multi-turn conversation with system, user, and assistant messages."""
        multi_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        compressed_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=50,
            tokens_after=30,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[50, 30, 30]),
        ):
            result = headroom.check_and_trim(multi_messages, MODEL, BASE_POLICY)

        assert result.messages == compressed_messages
        assert result.rejected is False

    def test_unexpected_result_type_logs_warning(self, patch_headroom_compress, patch_token_count):
        """Test unexpected result type logs warning and returns original messages."""
        with (
            mock.patch.object(headroom, "_headroom_compress", return_value="unexpected"),
            patch_token_count(side_effect=[10, 10]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.messages == MESSAGES


@pytest.mark.base
class TestCompactionConfig:
    """Tests for compaction configuration."""

    def test_headroom_disabled_with_false(self, patch_headroom_compress, patch_headroom_disabled, patch_token_count):
        """Test HEADROOM_ENABLED=false disables compression."""
        with (
            mock.patch.object(headroom, "_headroom_compress") as call,
            patch_headroom_disabled,
            patch_token_count(return_value=10),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        call.assert_not_called()
        assert result.rejected is False
        assert result.messages == MESSAGES

    def test_headroom_disabled_with_off(self, patch_headroom_compress, patch_token_count):
        """Test HEADROOM_ENABLED=off disables compression."""
        with (
            mock.patch.object(headroom, "_headroom_compress") as call,
            mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "off"}, clear=True),
            patch_token_count(return_value=10),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        call.assert_not_called()
        assert result.rejected is False

    def test_headroom_enabled_with_1(self, patch_headroom_compress, patch_token_count):
        """Test HEADROOM_ENABLED=1 enables compression."""
        compressed = SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            tokens_before=10,
            tokens_after=5,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            mock.patch.dict(os.environ, {"HEADROOM_ENABLED": "1"}, clear=True),
            patch_token_count(side_effect=[10, 5, 5]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False


@pytest.mark.base
class TestCompactionPolicy:
    """Tests for compaction policy resolution."""

    def test_policy_resolution_from_options(self):
        """Test policy options are correctly parsed."""
        policy = {"options": {"num_ctx": 8192}, "reserved_output_tokens": 100, "safety_headroom_tokens": 50}
        result = headroom.resolve_headroom("test", policy)
        assert result.max_num_ctx == 8192
        assert result.reserved_output_tokens == 100
        assert result.safety_headroom_tokens == 50

    def test_policy_resolution_with_default_options(self):
        """Test default options when none provided."""
        result = headroom.resolve_headroom("test", {})
        assert result.max_num_ctx == 65536
        assert result.reserved_output_tokens == 2048
        assert result.safety_headroom_tokens == 2048

    def test_policy_resolution_with_empty_options(self):
        """Test empty options dict."""
        result = headroom.resolve_headroom("test", {"options": {}})
        assert result.max_num_ctx == 65536

    def test_usable_prompt_budget_calculation(self):
        """Test usable prompt budget calculation."""
        policy = headroom.HeadroomPolicy(model="test", max_num_ctx=8192, reserved_output_tokens=100, safety_headroom_tokens=50)
        assert headroom._usable_prompt_budget(policy) == 8192 - 100 - 50

    def test_usable_prompt_budget_zero_when_overdraw(self):
        """Test usable prompt budget is zero when reserved + safety > num_ctx."""
        policy = headroom.HeadroomPolicy(model="test", max_num_ctx=100, reserved_output_tokens=50, safety_headroom_tokens=60)
        assert headroom._usable_prompt_budget(policy) == 0


@pytest.mark.base
class TestCompactionTelemetry:
    """Tests for compaction telemetry extraction."""

    def test_telemetry_with_dict_result(self):
        """Test telemetry extraction from dict result."""
        result = {
            "messages": MESSAGES,
            "tokens_before": 10,
            "tokens_after": 5,
            "transforms_applied": ["content_router"],
        }
        tokens_before, tokens_after, transforms = headroom._telemetry(result)
        assert tokens_before == 10
        assert tokens_after == 5
        assert transforms == ["content_router"]

    def test_telemetry_with_simple_namespace(self):
        """Test telemetry extraction from SimpleNamespace result."""
        result = SimpleNamespace(
            messages=MESSAGES,
            tokens_before=20,
            tokens_after=10,
            transforms_applied=["code_compressor"],
        )
        tokens_before, tokens_after, transforms = headroom._telemetry(result)
        assert tokens_before == 20
        assert tokens_after == 10

    def test_telemetry_with_missing_fields(self):
        """Test telemetry with missing fields uses defaults."""
        result = SimpleNamespace(messages=MESSAGES)
        tokens_before, tokens_after, transforms = headroom._telemetry(result)
        assert tokens_before == 0
        assert tokens_after == 0
        assert transforms == []


@pytest.mark.base
class TestNormalizeOutput:
    """Tests for _normalize_output function."""

    def test_normalize_output_with_list(self):
        """Test normalization with list input."""
        result = headroom._normalize_output(MESSAGES, MESSAGES)
        assert result == MESSAGES

    def test_normalize_output_with_compress_result(self):
        """Test normalization with CompressResult-like object."""
        result_obj = SimpleNamespace(messages=[{"role": "user", "content": "hi"}])
        normalized = headroom._normalize_output(result_obj, MESSAGES)
        assert normalized == [{"role": "user", "content": "hi"}]

    def test_normalize_output_with_dict(self):
        """Test normalization with dict result."""
        result_obj = {"messages": [{"role": "user", "content": "dict"}]}
        normalized = headroom._normalize_output(result_obj, MESSAGES)
        assert normalized == [{"role": "user", "content": "dict"}]

    def test_normalize_output_with_unexpected_type(self):
        """Test normalization with unexpected type returns original."""
        result_obj = "unexpected"
        normalized = headroom._normalize_output(result_obj, MESSAGES)
        assert normalized == MESSAGES


@pytest.mark.base
class TestCompactionIntegration:
    """Tests for compaction integration scenarios."""

    @pytest.mark.compression
    def test_full_compression_pipeline(self, patch_headroom_compress, patch_token_count):
        """Test full compression pipeline end-to-end."""
        compressed_messages = [{"role": "user", "content": "compressed"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=100,
            tokens_after=50,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[100, 50, 50]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.trimmed is True
        assert result.messages == compressed_messages
        assert result.prompt_tokens == 50
        assert result.trim_reason == "compressed"

    @pytest.mark.compression
    def test_full_compression_pipeline_with_tool_calls(self, patch_headroom_compress, patch_token_count):
        """Test full compression pipeline with tool calls."""
        messages_with_tools = [
            {"role": "user", "content": "Search for weather"},
            {
                "role": "user",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "web_search", "arguments": '{"query": "weather"}'}}],
            },
        ]
        compressed_messages = [
            {"role": "user", "content": "Search for weather"},
            {
                "role": "user",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "web_search", "arguments": '{"query": "weather"}'}}],
            },
        ]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=50,
            tokens_after=30,
            transforms_applied=[],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[50, 30, 30]),
        ):
            result = headroom.check_and_trim(messages_with_tools, MODEL, BASE_POLICY)

        assert result.rejected is False


@pytest.mark.base
class TestLongSessionCompaction:
    """Tests for synthetic long-session compaction (C1)."""

    @pytest.mark.compression
    def test_compaction_fires_before_proxy_trim(self, patch_headroom_compress, patch_token_count):
        """C1: Compaction happens before proxy emergency trimming."""
        compressed_messages = [{"role": "user", "content": "compacted"}]
        compressed = SimpleNamespace(
            messages=compressed_messages,
            tokens_before=500,
            tokens_after=200,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[500, 200, 200]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.trimmed is True

    @pytest.mark.compression
    def test_rejection_never_occurs_at_compaction_stage(self, patch_headroom_compress, patch_token_count):
        """C5: Compaction never rejects — recovery handles overflow."""
        compressed = SimpleNamespace(
            messages=[{"role": "user", "content": "x"}],
            tokens_before=10000,
            tokens_after=9000,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10000, 9000, 9000]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.error_response is None

    @pytest.mark.compression
    def test_messages_under_budget_not_modified_when_no_transforms(self, patch_headroom_compress, patch_token_count):
        """Regression: Messages already under budget should not be modified if no transforms."""
        compressed = SimpleNamespace(
            messages=MESSAGES,  # Same object reference
            tokens_before=10,
            tokens_after=10,
            transforms_applied=[],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 10, 10]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.rejected is False
        assert result.trimmed is False


@pytest.mark.base
class TestStructuredErrorResponse:
    """Tests for structured error response (should never occur at compaction stage)."""

    @pytest.mark.compression
    def test_error_response_is_none_at_compaction(self, patch_headroom_compress, patch_token_count):
        """Verify HeadroomCheckResult.error_response is None — no rejection at compaction."""
        compressed = SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            tokens_before=10,
            tokens_after=5,
            transforms_applied=["content_router"],
        )

        with (
            patch_headroom_compress(compressed),
            patch_token_count(side_effect=[10, 5, 5]),
        ):
            result = headroom.check_and_trim(MESSAGES, MODEL, BASE_POLICY)

        assert result.error_response is None


@pytest.mark.base
class TestOverflowRecovery:
    """Tests for apply_overflow_recovery() function."""

    @pytest.mark.compression
    def test_apply_overflow_recovery_reduces_overflow(self, patch_headroom_compress, patch_token_count):
        """The overflow-recovery flow reduces oversized prompts using the real recovery pipeline."""
        long_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
        policy = {"options": {"num_ctx": 1000}, "reserved_output_tokens": 100, "safety_headroom_tokens": 100}

        with mock.patch.object(headroom, "_token_count", side_effect=lambda ms, _m: 5000 if ms == long_messages else 2000):
            result = headroom.apply_overflow_recovery(long_messages, MODEL, policy)
        assert result.changed is True
        assert len(result.messages) <= len(long_messages)

    @pytest.mark.compression
    def test_apply_overflow_recovery_handles_compaction_safely(self, patch_headroom_compress, patch_token_count):
        """Recovery should still produce a valid message list without relying on a trim strategy name."""
        long_messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]
        policy = {"options": {"num_ctx": 1000}, "reserved_output_tokens": 100, "safety_headroom_tokens": 100}

        with mock.patch.object(headroom, "_token_count", side_effect=lambda ms, _m: 5000 if ms == long_messages else 2500):
            result = headroom.apply_overflow_recovery(long_messages, MODEL, policy)
        assert result.changed is True

    @pytest.mark.compression
    def test_overflow_recovery_keeps_tool_result_units_intact(self):
        """Recovery must not orphan tool call results when it trims conversation history."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Results: ..."},
            {"role": "user", "content": "What did you find?"},
            {"role": "assistant", "content": "I found Y."},
        ]
        policy = {"options": {"num_ctx": 50}, "reserved_output_tokens": 10, "safety_headroom_tokens": 10}

        result = headroom.apply_overflow_recovery(messages, MODEL, policy)

        assistant_tool_ids = set()
        for msg in result.messages:
            for tc in (msg.get("tool_calls") or []):
                if isinstance(tc, dict) and tc.get("id"):
                    assistant_tool_ids.add(tc["id"])

        for msg in result.messages:
            if msg.get("role") == "tool":
                assert msg.get("tool_call_id") in assistant_tool_ids

    @pytest.mark.compression
    def test_overflow_recovery_preserves_orphan_tool_results(self):
        """Tool results without matching calls are preserved while recovery trims older context."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "tool", "tool_call_id": "call_orphan", "content": "Orphaned result"},
        ]
        policy = {"options": {"num_ctx": 50}, "reserved_output_tokens": 10, "safety_headroom_tokens": 10}

        result = headroom.apply_overflow_recovery(messages, MODEL, policy)

        remaining_roles = [m.get("role") for m in result.messages]
        assert "tool" in remaining_roles

    @pytest.mark.compression
    def test_overflow_recovery_returns_message_list(self, patch_token_count):
        """The recovery flow returns a message list in the canonical shape."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "M1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "M2"},
            {"role": "assistant", "content": "A2"},
        ]
        policy = {"options": {"num_ctx": 50}, "reserved_output_tokens": 10, "safety_headroom_tokens": 10}

        with mock.patch.object(headroom, "_token_count", side_effect=lambda ms, _m: 5000 if ms == messages else 2500):
            result = headroom.apply_overflow_recovery(messages, MODEL, policy)

        assert isinstance(result.messages, list)
        assert result.changed is True

    @pytest.mark.compression
    def test_overflow_recovery_handles_large_history_without_typeerror(self):
        """The recovery function remains robust when the prompt is larger than the budget."""
        messages = [{"role": "system", "content": "You are helpful"}] + [
            {"role": "user", "content": f"Message {i}"} for i in range(20)
        ]
        policy = {"options": {"num_ctx": 50}, "reserved_output_tokens": 10, "safety_headroom_tokens": 10}

        result = headroom.apply_overflow_recovery(messages, MODEL, policy)
        assert isinstance(result.messages, list)

    def test_proactive_summarization_triggers_on_threshold(self):
        """Proactive summarization runs before upstream call when the configured context threshold is reached."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Please summarize a long history."},
        ]
        policy = {
            "options": {"num_ctx": 1000},
            "reserved_output_tokens": 100,
            "safety_headroom_tokens": 100,
            "proactive_summarization_enabled": True,
            "proactive_summarization_threshold_tokens": 250,
        }

        with (
            mock.patch.object(headroom, "_token_count", return_value=300),
            mock.patch.object(headroom, "_compress_with_config", return_value=(messages, 300, 200, ["proactive_summary"])) as compress_mock,
        ):
            result = headroom.maybe_apply_proactive_summarization(messages, MODEL, policy)

        assert result.triggered is True
        assert result.messages == messages
        assert compress_mock.call_count == 1

    def test_proactive_summarization_keeps_tool_integrity(self):
        """Proactive summarization can compress safely without stripping tool-call state."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Search and report back."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{\"q\":\"python\"}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        policy = {
            "options": {"num_ctx": 1000},
            "reserved_output_tokens": 100,
            "safety_headroom_tokens": 100,
            "proactive_summarization_enabled": True,
            "proactive_summarization_threshold_tokens": 50,
        }

        with (
            mock.patch.object(headroom, "_token_count", return_value=200),
            mock.patch.object(headroom, "_compress_with_config", return_value=(messages, 200, 160, ["proactive_summary"])),
        ):
            result = headroom.maybe_apply_proactive_summarization(messages, MODEL, policy)

        assert result.triggered is True
        assert len(result.messages) == 4
        assert result.messages[2]["tool_calls"][0]["id"] == "call_1"
        assert result.messages[3]["tool_call_id"] == "call_1"

    def test_policy_entry_strips_legacy_strategy_names(self):
        """Legacy strategy names are removed because they do not affect the runtime flow."""
        normalized = headroom.normalize_policy_entry({
            "reserved_output_tokens": 128,
            "safety_headroom_tokens": 64,
            "trim_strategy": "drop_oldest",
            "recovery_strategy": "summarize_history",
        })

        assert "trim_strategy" not in normalized
        assert "recovery_strategy" not in normalized

    def test_policy_entry_ignores_allow_auto_pull(self):
        """allow_auto_pull is not a supported router config key and must be discarded."""
        normalized = headroom.normalize_policy_entry({
            "reserved_output_tokens": 128,
            "safety_headroom_tokens": 64,
            "allow_auto_pull": True,
        })

        assert "allow_auto_pull" not in normalized


@pytest.mark.base
class TestOverflowRecoveryFlow:
    """Tests for the canonical overflow-recovery flow."""

    def test_apply_overflow_recovery_uses_recovery_pipeline(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        policy = {"options": {"num_ctx": 1000}, "reserved_output_tokens": 100, "safety_headroom_tokens": 100}

        result = headroom.apply_overflow_recovery(messages, MODEL, policy)

        assert isinstance(result, headroom.RecoveryResult)
        assert result.messages == messages or result.changed is True

    def test_strip_oldest_segments_removes_oldest_complete_segment(self):
        messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]

        with mock.patch.object(headroom, "_token_count", side_effect=lambda ms, _m: len(ms) * 100):
            stripped = headroom._strip_oldest_segments(
                messages=messages,
                budget=250,
                model_name=MODEL,
                min_segments_to_keep=1,
            )

        assert stripped == [
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]

    def test_apply_overflow_recovery_respects_material_change_guard(self):
        messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]
        policy = {
            "options": {"num_ctx": 180},
            "reserved_output_tokens": 0,
            "safety_headroom_tokens": 0,
        }

        with (
            mock.patch.object(headroom, "RECOVERY_ENABLE_STRONGER_COMPRESSION", True),
            mock.patch.object(headroom, "RECOVERY_ENABLE_TEXT_COMPRESSION", False),
            mock.patch.object(headroom, "RECOVERY_ENABLE_SEGMENT_STRIPPING", False),
            mock.patch.object(headroom, "RECOVERY_MIN_TOKEN_DELTA", 50),
            mock.patch.object(headroom, "_headroom_compress", object()),
            mock.patch.object(
                headroom,
                "_compress_with_config",
                return_value=(messages[:], 200, 180, ["stronger_compression_test"]),
            ),
            mock.patch.object(
                headroom,
                "_token_count",
                side_effect=lambda ms, _m: 200 if ms == messages else 180,
            ),
        ):
            result = headroom.apply_overflow_recovery(messages, MODEL, policy)

        assert result.changed is False
        assert result.success_candidate is False
        assert any(a.stage.startswith("stronger_headroom_compression_") for a in result.attempts)

    def test_apply_overflow_recovery_stage_d_applies_last_resort_strip(self):
        messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]
        policy = {
            "options": {"num_ctx": 250},
            "reserved_output_tokens": 0,
            "safety_headroom_tokens": 0,
        }

        with (
            mock.patch.object(headroom, "RECOVERY_ENABLE_STRONGER_COMPRESSION", False),
            mock.patch.object(headroom, "RECOVERY_ENABLE_TEXT_COMPRESSION", False),
            mock.patch.object(headroom, "RECOVERY_ENABLE_SEGMENT_STRIPPING", True),
            mock.patch.object(headroom, "RECOVERY_MIN_STRIP_TOKEN_DELTA", 1),
            mock.patch.object(headroom, "RECOVERY_MIN_SEGMENTS_TO_KEEP", 1),
            mock.patch.object(headroom, "_token_count", side_effect=lambda ms, _m: len(ms) * 100),
        ):
            result = headroom.apply_overflow_recovery(messages, MODEL, policy)

        assert result.changed is True
        assert result.success_candidate is True
        assert result.final_stage == "last_resort_segment_stripping"
        assert result.messages == [
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
        ]

    def test_strip_oldest_segments_blocks_cross_segment_tool_orphans(self):
        messages = [
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "next turn"},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "assistant", "content": "used result"},
        ]

        with mock.patch.object(headroom, "_token_count", side_effect=lambda ms, _m: len(ms) * 100):
            stripped = headroom._strip_oldest_segments(
                messages=messages,
                budget=150,
                model_name=MODEL,
                min_segments_to_keep=1,
            )

        # Oldest segment should not be removed because its tool_call result lives in a later segment.
        assert stripped == messages

    def test_stage_d_respects_min_segments_to_keep(self):
        messages = [
            {"role": "user", "content": "turn1"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "turn3"},
            {"role": "assistant", "content": "resp3"},
        ]
        policy = {
            "options": {"num_ctx": 250},
            "reserved_output_tokens": 0,
            "safety_headroom_tokens": 0,
        }

        with (
            mock.patch.object(headroom, "RECOVERY_ENABLE_STRONGER_COMPRESSION", False),
            mock.patch.object(headroom, "RECOVERY_ENABLE_TEXT_COMPRESSION", False),
            mock.patch.object(headroom, "RECOVERY_ENABLE_SEGMENT_STRIPPING", True),
            mock.patch.object(headroom, "RECOVERY_MIN_STRIP_TOKEN_DELTA", 1),
            mock.patch.object(headroom, "RECOVERY_MIN_SEGMENTS_TO_KEEP", 2),
            mock.patch.object(headroom, "_token_count", side_effect=lambda ms, _m: len(ms) * 100),
        ):
            result = headroom.apply_overflow_recovery(messages, MODEL, policy)

        assert result.changed is True
        # Keep the newest 2 segments (turn2/resp2 and turn3/resp3).
        assert result.messages == [
            {"role": "user", "content": "turn2"},
            {"role": "assistant", "content": "resp2"},
            {"role": "user", "content": "turn3"},
            {"role": "assistant", "content": "resp3"},
        ]


@pytest.mark.base
class TestContextOverflowDetection:
    """Tests for is_context_overflow_error() function."""

    def test_detects_http_400_with_context_length(self):
        """Test detection of HTTP 400 with 'context length exceeded' message."""
        assert headroom.is_context_overflow_error(400, "context length exceeded") is True

    def test_detects_http_400_with_prompt_too_long(self):
        """Test detection of HTTP 400 with 'prompt too long' message."""
        assert headroom.is_context_overflow_error(400, "prompt too long for model") is True

    def test_detects_http_400_with_input_length_exceeds(self):
        """Test detection of HTTP 400 with 'input length exceeds' message."""
        assert headroom.is_context_overflow_error(400, "input length exceeds maximum context length") is True

    def test_detects_http_400_with_context_length_snake_case(self):
        """Test detection of HTTP 400 with 'context_length' in message."""
        assert headroom.is_context_overflow_error(400, '{"error": "context_length_exceeded"}') is True

    def test_returns_false_for_non_400_status(self):
        """Test that non-400 status codes return False."""
        assert headroom.is_context_overflow_error(500, "internal server error") is False
        assert headroom.is_context_overflow_error(200, "success") is False

    def test_returns_false_for_unrelated_400_message(self):
        """Test that unrelated 400 messages return False."""
        assert headroom.is_context_overflow_error(400, "invalid parameter") is False
        assert headroom.is_context_overflow_error(400, "model not found") is False

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        assert headroom.is_context_overflow_error(400, "CONTEXT LENGTH EXCEEDED") is True
        assert headroom.is_context_overflow_error(400, "Prompt Too Long") is True
